"""
NeuroLoop-Q EEG Encoder
========================
Convolutional Patch Encoder for rs-EEG spectrograms.

Pipeline:
  Raw STFT spectrogram (n_channels, n_freq_bins, n_time_windows)
    -> Conv1D patch encoding per channel -> d_model embedding
    -> RoPE positional encoding (channel, frequency, time)
    -> Multi-head self-attention with EEG connectivity graph bias
    -> Output: (batch, n_eeg_tokens, d_model)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for multi-dimensional positions.
    Extends 1D RoPE (Su et al., 2021) to handle (channel, frequency, time)
    position triplets by applying rotation in interleaved sub-dimensions.
    """

    def __init__(self, d_model: int, max_channels: int = 128,
                 max_freq_bins: int = 64, max_time_windows: int = 64):
        super().__init__()
        self.d_model = d_model
        assert d_model % 6 == 0, \
            f"d_model ({d_model}) must be divisible by 6 for 3D RoPE"

        d_per_axis = d_model // 3
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_per_axis, 2).float() / d_per_axis))

        def _build_table(max_pos):
            pos = torch.arange(max_pos).float()
            freq = torch.einsum("i,j->ij", pos, inv_freq)
            return freq.cos(), freq.sin()

        cos_ch, sin_ch = _build_table(max_channels)
        cos_f, sin_f = _build_table(max_freq_bins)
        cos_t, sin_t = _build_table(max_time_windows)

        self.register_buffer("cos_ch", cos_ch, persistent=False)
        self.register_buffer("sin_ch", sin_ch, persistent=False)
        self.register_buffer("cos_f", cos_f, persistent=False)
        self.register_buffer("sin_f", sin_f, persistent=False)
        self.register_buffer("cos_t", cos_t, persistent=False)
        self.register_buffer("sin_t", sin_t, persistent=False)

    def _rotate(self, x_chunk, cos, sin, idx):
        cos_val = cos[idx]
        sin_val = sin[idx]
        x_even = x_chunk[..., 0::2]
        x_odd = x_chunk[..., 1::2]
        return torch.stack([
            x_even * cos_val - x_odd * sin_val,
            x_even * sin_val + x_odd * cos_val,
        ], dim=-1).flatten(-2)

    def forward(self, x, ch_idx, f_idx, t_idx):
        B, N, D = x.shape
        d_per_axis = D // 3
        x1 = x[..., :d_per_axis]
        x2 = x[..., d_per_axis:2 * d_per_axis]
        x3 = x[..., 2 * d_per_axis:]
        x1 = self._rotate(x1, self.cos_ch, self.sin_ch, ch_idx)
        x2 = self._rotate(x2, self.cos_f, self.sin_f, f_idx)
        x3 = self._rotate(x3, self.cos_t, self.sin_t, t_idx)
        return torch.cat([x1, x2, x3], dim=-1)


class ConnectivityAwareAttention(nn.Module):
    """Multi-head self-attention with PLV connectivity matrix bias.
    softmax((QK^T / sqrt(d)) + alpha * connectivity) V
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
        self.scale = math.sqrt(self.d_head)
        self.connectivity_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, connectivity=None, mask=None):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if connectivity is not None:
            scores = scores + self.connectivity_alpha * connectivity.unsqueeze(1)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.o_proj(out)


class EEGEncoder(nn.Module):
    """EEG Spectrogram -> Token Sequence Encoder.

    Input:
        tokens:        (B, n_channels, n_freq_bins * n_time_windows)
        connectivity:  (B, n_channels, n_channels) PLV matrix
        bandpower:     (B, n_channels, n_bands)
    Output:
        encoded:       (B, n_channels, d_model)
        global_token:  (B, 1, d_model)
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_bands: int = 5, n_channels: int = 128,
                 n_freq_bins: int = 23, n_time_windows: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels

        self.conv_patch = nn.Conv1d(1, d_model // 2, kernel_size=7, stride=3, padding=3)
        self.conv_norm = nn.LayerNorm(d_model // 2)
        self.proj = nn.Linear(d_model // 2, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

        self.bandpower_proj = nn.Sequential(
            nn.Linear(n_channels * n_bands, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.rope = RotaryPositionalEmbedding(d_model, n_channels, n_freq_bins, n_time_windows)
        self.self_attn = ConnectivityAwareAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.register_buffer("ch_indices", torch.arange(n_channels), persistent=False)
        self.register_buffer("f_indices", torch.zeros(n_channels, dtype=torch.long), persistent=False)
        self.register_buffer("t_indices", torch.zeros(n_channels, dtype=torch.long), persistent=False)

    def forward(self, tokens, connectivity=None, bandpower=None, mask=None):
        B, N, _ = tokens.shape

        # Per-channel conv encoding
        x = tokens.view(B * N, 1, -1)
        x = self.conv_patch(x).transpose(1, 2)
        x = self.conv_norm(x)
        x = F.gelu(x).mean(dim=1)
        x = self.proj(x)
        x = self.proj_norm(x)
        x = F.gelu(x).view(B, N, self.d_model)

        # RoPE
        x = self.rope(x, self.ch_indices, self.f_indices, self.t_indices)

        # Self-attention + residual
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, connectivity, mask)
        x = self.attn_dropout(x)
        x = residual + x

        # FFN + residual
        x = x + self.ffn(self.ffn_norm(x))

        # Global bandpower token
        global_token = None
        if bandpower is not None:
            global_token = self.bandpower_proj(
                bandpower.view(B, -1)
            ).unsqueeze(1)

        return x, global_token
