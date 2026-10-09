"""
NeuroLoop-Q Full Model Assembly
================================
Assembles EEG encoder, MRI encoder, CMSSF fusion, Looped Transformer core,
and classification heads into the complete model.

Forward pass:
  1. EEG tokens  -> EEGEncoder  -> H_eeg (B, N_eeg, d)
  2. MRI volumes+FC -> MRIEncoder -> H_mri (B, N_mri, d)
  3. For each loop t:
     a. CMSSF: H_eeg, H_mri -> H_fused, H_eeg', H_mri'
     b. LoopBlock: H_fused -> H_fused'
  4. Pool + classify

Total parameters: ~2.1M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from .eeg_encoder import EEGEncoder
from .mri_encoder import MRIEncoder
from .cmssf import CMSSFBlock
from .looped_transformer import LoopedTransformerCore


class ClassificationHead(nn.Module):
    """Lightweight linear classification head with optional temperature."""

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) -> (B, n_classes)"""
        return self.proj(self.dropout(x))


class NeuroLoopQ(nn.Module):
    """Full NeuroLoop-Q model.

    Config params:
        d_model:        256
        n_heads:          8
        n_core_layers:    4 (weight-tied)
        max_loops:       12 (adaptive)
        min_loops:        3
        ffn_expansion:    2 (reduced from 4)
        tau:            0.15 (confidence halting threshold)
        n_classes_ad:     3 (CN, MCI, AD)
        n_classes_asd:    2 (TD, ASD)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_core_layers: int = 4,
        max_loops: int = 12,
        min_loops: int = 3,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        tau: float = 0.15,
        n_classes_ad: int = 3,
        n_classes_asd: int = 2,
        n_eeg_channels: int = 128,
        n_eeg_freq_bins: int = 23,
        n_eeg_time_windows: int = 15,
        n_eeg_bands: int = 5,
        n_mri_rois: int = 116,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_loops = max_loops
        self.min_loops = min_loops

        # ── Encoders ──
        self.eeg_encoder = EEGEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_bands=n_eeg_bands,
            n_channels=n_eeg_channels,
            n_freq_bins=n_eeg_freq_bins,
            n_time_windows=n_eeg_time_windows,
            dropout=dropout,
        )

        self.mri_encoder = MRIEncoder(
            d_model=d_model,
            n_rois=n_mri_rois,
            n_gc_layers=2,
            dropout=dropout,
        )

        # ── CMSSF (shared across loops, weight-tied) ──
        self.cmssf = CMSSFBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            ffn_expansion=ffn_expansion,
        )

        # ── Looped Transformer Core ──
        self.looped_core = LoopedTransformerCore(
            d_model=d_model,
            n_heads=n_heads,
            n_core_layers=n_core_layers,
            max_loops=max_loops,
            min_loops=min_loops,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            tau=tau,
        )

        # ── Classification Heads ──
        self.ad_head = ClassificationHead(d_model, n_classes_ad, dropout)
        self.asd_head = ClassificationHead(d_model, n_classes_asd, dropout)

        # ── Auxiliary Heads (multi-task regularization) ──
        self.age_head = nn.Linear(d_model, 1)      # age regression
        self.sex_head = nn.Linear(d_model, 2)       # sex classification

        # ── Global token (CLS) for pooling ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pool(self, x: torch.Tensor,
              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean pooling over sequence dimension."""
        if mask is not None:
            return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return x.mean(dim=1)

    def forward(
        self,
        eeg_tokens: Optional[torch.Tensor] = None,
        eeg_connectivity: Optional[torch.Tensor] = None,
        eeg_bandpower: Optional[torch.Tensor] = None,
        eeg_mask: Optional[torch.Tensor] = None,
        mri_volumes: Optional[torch.Tensor] = None,
        mri_fc: Optional[torch.Tensor] = None,
        mri_mask: Optional[torch.Tensor] = None,
        loop_depth: Optional[int] = None,
        return_all_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns dict with:
            'logits_ad':   (B, n_classes_ad)
            'logits_asd':  (B, n_classes_asd)
            'age_pred':    (B, 1)
            'sex_pred':    (B, 2)
            'loops_used':   int
            'fused_repr':  (B, d_model) — for contrastive loss
            'all_states':  List[torch.Tensor] (if return_all_states)
        """
        B = max(
            eeg_tokens.shape[0] if eeg_tokens is not None else 0,
            mri_volumes.shape[0] if mri_volumes is not None else 0,
            mri_fc.shape[0] if mri_fc is not None else 0,
        )
        device = next(self.parameters()).device

        # ── Encode EEG ──
        if eeg_tokens is not None:
            h_eeg, eeg_global = self.eeg_encoder(
                eeg_tokens, eeg_connectivity, eeg_bandpower, eeg_mask
            )
            # Prepend global token
            if eeg_global is not None:
                h_eeg = torch.cat([eeg_global, h_eeg], dim=1)
                if eeg_mask is not None:
                    eeg_mask = torch.cat([
                        torch.ones(B, 1, device=device, dtype=torch.bool),
                        eeg_mask
                    ], dim=1)
        else:
            # Create zero placeholder
            n_ch = self.eeg_encoder.n_channels
            h_eeg = torch.zeros(B, n_ch + 1, self.d_model, device=device)
            eeg_mask = torch.zeros(B, n_ch + 1, device=device, dtype=torch.bool)

        # ── Encode MRI ──
        if mri_fc is not None or mri_volumes is not None:
            if mri_fc is None:
                mri_fc = torch.eye(self.mri_encoder.n_rois, device=device).unsqueeze(0).expand(B, -1, -1)
            h_mri = self.mri_encoder(mri_volumes, mri_fc)
            if mri_mask is None:
                mri_mask = torch.ones(B, h_mri.shape[1], device=device, dtype=torch.bool)
        else:
            h_mri = torch.zeros(B, self.mri_encoder.n_rois, self.d_model, device=device)
            mri_mask = torch.zeros(B, self.mri_encoder.n_rois, device=device, dtype=torch.bool)

        # ── CMSSF + Looped Transformer ──
        all_loop_states = []
        L = loop_depth if loop_depth is not None else self.max_loops

        # Initial fusion
        h_fused, h_eeg, h_mri = self.cmssf(h_eeg, h_mri, eeg_mask, mri_mask)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        h_fused = torch.cat([cls, h_fused], dim=1)
        fused_mask = torch.ones(B, h_fused.shape[1], device=device, dtype=torch.bool)

        # Looped transformer core
        h_fused, loops_used, all_states = self.looped_core(
            h_fused, fused_mask, loop_depth=loop_depth,
            return_all_states=return_all_states,
        )

        # ── Pool (using CLS token) ──
        pooled = h_fused[:, 0, :]  # CLS token representation

        # ── Classify ──
        logits_ad = self.ad_head(pooled)
        logits_asd = self.asd_head(pooled)
        age_pred = self.age_head(pooled)
        sex_pred = self.sex_head(pooled)

        result = {
            "logits_ad": logits_ad,
            "logits_asd": logits_asd,
            "age_pred": age_pred,
            "sex_pred": sex_pred,
            "loops_used": loops_used,
            "fused_repr": pooled,
        }

        if return_all_states:
            result["all_states"] = all_states

        return result

    # ── Utility methods ──

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_module(self) -> Dict[str, int]:
        """Count parameters per submodule."""
        counts = {}
        for name, module in [
            ("eeg_encoder", self.eeg_encoder),
            ("mri_encoder", self.mri_encoder),
            ("cmssf", self.cmssf),
            ("looped_core", self.looped_core),
            ("ad_head", self.ad_head),
            ("asd_head", self.asd_head),
            ("age_head", self.age_head),
            ("sex_head", self.sex_head),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters() if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts

    @torch.no_grad()
    def get_attention_weights(
        self, eeg_tokens=None, eeg_connectivity=None, eeg_bandpower=None,
        eeg_mask=None, mri_volumes=None, mri_fc=None, mri_mask=None,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability.
        Returns per-loop attention maps showing how the model
        refines its diagnosis iteratively.
        """
        # This would require modifying the forward pass to return attention weights
        # For now, return the loop states as a proxy
        result = self.forward(
            eeg_tokens=eeg_tokens, eeg_connectivity=eeg_connectivity,
            eeg_bandpower=eeg_bandpower, eeg_mask=eeg_mask,
            mri_volumes=mri_volumes, mri_fc=mri_fc, mri_mask=mri_mask,
            return_all_states=True,
        )
        return {"loop_states": result.get("all_states", []),
                "loops_used": result["loops_used"]}
