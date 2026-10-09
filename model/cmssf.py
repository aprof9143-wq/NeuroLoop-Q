"""
NeuroLoop-Q Cross-Modal Spectral-Spatial Fusion (CMSSF)
========================================================
The novel fusion mechanism that iteratively exchanges information
between EEG and MRI streams at every loop iteration of the
Looped Transformer core.

Mechanism (per loop iteration t):
  H_eeg^t  = LoopBlock(H_eeg^{t-1}, CrossAttn(Q=H_eeg, K=V=H_mri))
  H_mri^t  = LoopBlock(H_mri^{t-1}, CrossAttn(Q=H_mri, K=V=H_eeg))
  H_fused  = GatedFusion(H_eeg^t, H_mri^t)

The gated fusion learns modality-specific reliability dynamically:
  g = sigmoid(W_g * [H_eeg; H_mri; H_eeg (x) H_mri])
  H_fused = g * H_eeg + (1 - g) * H_mri

This is critical because:
  - AD: structural atrophy in MRI is the dominant biomarker
  - ASD: EEG spectral atypicality is more discriminative
  - The gate adapts per-subject, not just per-dataset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossModalAttention(nn.Module):
    """Cross-attention: one modality attends to the other.

    Q from modality A, K/V from modality B.
    Standard multi-head cross-attention with optional masking.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_a:  (B, N_a, d_model) — query source
        x_b:  (B, N_b, d_model) — key/value source
        mask: (B, N_b) — True for valid tokens in x_b
        """
        B, N_a, _ = x_a.shape
        N_b = x_b.shape[1]

        q = self.q_proj(x_a).view(B, N_a, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_b).view(B, N_b, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_b).view(B, N_b, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N_a, N_b)

        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, h, N_a, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N_a, self.d_model)
        return self.o_proj(out)


class GatedFusion(nn.Module):
    """Learned gated fusion of two modality representations.

    g = sigmoid(W_g * [H_eeg; H_mri; H_eeg * H_mri] + b_g)
    H_fused = g * H_eeg + (1 - g) * H_mri

    The gate coefficient is per-token and per-dimension, allowing
    fine-grained modality balancing. Gate temperature is a learnable
    parameter that the Q-RLO can adjust.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Input: [H_eeg; H_mri; H_eeg * H_mri] -> 3 * d_model
        self.gate_proj = nn.Linear(3 * d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_eeg: torch.Tensor, h_mri: torch.Tensor,
                eeg_mask: Optional[torch.Tensor] = None,
                mri_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h_eeg: (B, N_eeg, d_model)
        h_mri: (B, N_mri, d_model) — may have different N
        Returns: (B, N_fused, d_model) where N_fused = max(N_eeg, N_mri)

        If sequence lengths differ, we pad the shorter one and mask.
        """
        B = h_eeg.shape[0]
        N_eeg = h_eeg.shape[1]
        N_mri = h_mri.shape[1]
        d = h_eeg.shape[2]

        if N_eeg != N_mri:
            # Pad shorter sequence to match
            if N_eeg < N_mri:
                pad = torch.zeros(B, N_mri - N_eeg, d, device=h_eeg.device)
                h_eeg = torch.cat([h_eeg, pad], dim=1)
                if eeg_mask is not None:
                    eeg_mask = torch.cat([
                        eeg_mask,
                        torch.zeros(B, N_mri - N_eeg, device=eeg_mask.device, dtype=torch.bool)
                    ], dim=1)
            else:
                pad = torch.zeros(B, N_eeg - N_mri, d, device=h_mri.device)
                h_mri = torch.cat([h_mri, pad], dim=1)
                if mri_mask is not None:
                    mri_mask = torch.cat([
                        mri_mask,
                        torch.zeros(B, N_eeg - N_mri, device=mri_mask.device, dtype=torch.bool)
                    ], dim=1)

        N = max(N_eeg, N_mri)

        # Compute gate
        interaction = h_eeg * h_mri  # element-wise
        gate_input = torch.cat([h_eeg, h_mri, interaction], dim=-1)  # (B, N, 3*d)
        gate = torch.sigmoid(self.gate_proj(gate_input) / self.temperature)
        gate = self.dropout(gate)

        # Weighted fusion
        fused = gate * h_eeg + (1 - gate) * h_mri

        return fused


class CMSSFBlock(nn.Module):
    """One iteration of Cross-Modal Spectral-Spatial Fusion.

    This block is applied at every loop iteration of the
    Looped Transformer core. It performs:
      1. Cross-attention: EEG attends to MRI
      2. Cross-attention: MRI attends to EEG
      3. Gated fusion of updated representations
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 dropout: float = 0.1, ffn_expansion: int = 2):
        super().__init__()
        self.d_model = d_model

        # Cross-attention layers
        self.eeg_to_mri_attn = CrossModalAttention(d_model, n_heads, dropout)
        self.mri_to_eeg_attn = CrossModalAttention(d_model, n_heads, dropout)

        # Layer norms (pre-norm)
        self.eeg_norm = nn.LayerNorm(d_model)
        self.mri_norm = nn.LayerNorm(d_model)

        # FFN after cross-attention
        self.eeg_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
        )
        self.mri_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
        )
        self.eeg_ffn_norm = nn.LayerNorm(d_model)
        self.mri_ffn_norm = nn.LayerNorm(d_model)

        # Gated fusion
        self.fusion = GatedFusion(d_model, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h_eeg: torch.Tensor, h_mri: torch.Tensor,
                eeg_mask: Optional[torch.Tensor] = None,
                mri_mask: Optional[torch.Tensor] = None) -> tuple:
        """
        h_eeg: (B, N_eeg, d_model)
        h_mri: (B, N_mri, d_model)
        Returns:
            h_fused: (B, N_fused, d_model)
            h_eeg:   (B, N_eeg, d_model) — updated
            h_mri:   (B, N_mri, d_model) — updated
        """
        # ── Cross-attention: EEG queries MRI ──
        eeg_residual = h_eeg
        h_eeg_normed = self.eeg_norm(h_eeg)
        eeg_cross = self.eeg_to_mri_attn(h_eeg_normed, h_mri, mask=mri_mask)
        h_eeg = eeg_residual + self.dropout(eeg_cross)

        # FFN for EEG
        h_eeg = h_eeg + self.dropout(self.eeg_ffn(self.eeg_ffn_norm(h_eeg)))

        # ── Cross-attention: MRI queries EEG ──
        mri_residual = h_mri
        h_mri_normed = self.mri_norm(h_mri)
        mri_cross = self.mri_to_eeg_attn(h_mri_normed, h_eeg, mask=eeg_mask)
        h_mri = mri_residual + self.dropout(mri_cross)

        # FFN for MRI
        h_mri = h_mri + self.dropout(self.mri_ffn(self.mri_ffn_norm(h_mri)))

        # ── Gated fusion ──
        h_fused = self.fusion(h_eeg, h_mri, eeg_mask, mri_mask)

        return h_fused, h_eeg, h_mri
