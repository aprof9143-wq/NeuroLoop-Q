"""
NeuroLoop-Q Looped Recurrent Transformer Core
==============================================
The weight-tied recurrent transformer that provides deep reasoning
with minimal parameters.

Key insight from Saunshi et al. (ICLR 2025): A k-layer transformer
looped L times nearly matches the performance of a kL-layer non-looped
model, while requiring only k layers worth of parameters.

Design:
  - n_core_layers = 4 transformer blocks (weight-tied)
  - max_loops = 12 (adaptive, confidence-based halting)
  - min_loops = 3 (minimum depth for all inputs)
  - Learned loop-step embedding injected at each iteration
  - Adaptive compute: easy cases exit early, hard cases use more loops

References:
  - Saunshi et al., "Reasoning with Latent Thoughts", ICLR 2025
  - Gatmiry et al., "Looped Transformers Implement Multi-step GD", 2024
  - Nguyen & Lin, "Intra-Layer Recurrence in Transformers", 2025
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with multi-head self-attention
    and feed-forward network. Uses reduced FFN expansion (2x instead of 4x)
    for compute efficiency.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, N, d_model)
        mask: (B, N) — True for valid tokens
        """
        B, N, _ = x.shape

        # Self-attention (pre-norm + residual)
        residual = x
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.d_model)
        x = residual + self.o_proj(out)

        # FFN (pre-norm + residual)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x


class LoopStepEmbedding(nn.Module):
    """Learned embedding that encodes which loop iteration we're on.
    Injected additively into the token representations at each step
    so the shared layers can distinguish loop iterations.
    """

    def __init__(self, d_model: int, max_loops: int = 12):
        super().__init__()
        self.embeddings = nn.Embedding(max_loops, d_model)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Add loop-step embedding to input.
        x: (B, N, d_model)
        step: int
        """
        return x + self.embeddings.weight[step].unsqueeze(0).unsqueeze(0)


class LoopedTransformerCore(nn.Module):
    """Weight-tied recurrent transformer with adaptive loop depth.

    The same n_core_layers blocks are applied repeatedly for L iterations.
    At each iteration, the model can optionally halt early if the
    classification head is confident enough (entropy < tau).

    Parameters: only n_core_layers * (transformer block params)
    Effective depth: n_core_layers * max_loops
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_core_layers: int = 4, max_loops: int = 12,
                 min_loops: int = 3, ffn_expansion: int = 2,
                 dropout: float = 0.1, tau: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.n_core_layers = n_core_layers
        self.max_loops = max_loops
        self.min_loops = min_loops
        self.tau = tau  # confidence halting threshold

        # Shared transformer blocks (weight-tied across loops)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_expansion, dropout)
            for _ in range(n_core_layers)
        ])

        # Loop-step embedding
        self.loop_embedding = LoopStepEmbedding(d_model, max_loops)

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head (for confidence-based halting)
        # This is a lightweight head used only for halting decisions
        self.halt_head = nn.Linear(d_model, 3)  # max classes (AD: 3)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                loop_depth: Optional[int] = None,
                return_all_states: bool = False) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        x:          (B, N, d_model) — input token sequence (fused)
        mask:       (B, N) — True for valid tokens
        loop_depth: Optional[int] — if provided, run exactly this many loops
        Returns:
            output:     (B, N, d_model) — final representation
            loops_used: int — number of loops actually executed
            all_states: List of (B, N, d_model) — per-loop states (if return_all_states)
        """
        all_states = []
        L = loop_depth if loop_depth is not None else self.max_loops

        for t in range(L):
            # Add loop-step embedding
            x = self.loop_embedding(x, t)

            # Apply shared transformer blocks
            for block in self.blocks:
                x = block(x, mask)

            # Check confidence-based halting (only in adaptive mode)
            if loop_depth is None and t >= self.min_loops - 1:
                # Compute halting score from pooled representation
                if mask is not None:
                    pool = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
                else:
                    pool = x.mean(dim=1)
                logits = self.halt_head(pool)
                entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1))
                entropy = entropy.sum(dim=-1).mean()
                if entropy < self.tau:
                    x = self.final_norm(x)
                    all_states.append(x)
                    return x, t + 1, all_states

            if return_all_states:
                all_states.append(x)

        x = self.final_norm(x)
        if return_all_states:
            all_states.append(x)
        return x, L, all_states
