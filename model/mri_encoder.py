"""
NeuroLoop-Q MRI Encoder
=======================
Graph Convolutional Network (GCN) encoder for brain ROI graphs.

Two input streams:
  1. sMRI ROI volumes  -> (n_rois,) volume vector -> broadcast as node features
  2. fMRI FC matrix    -> (n_rois, n_rois) -> graph adjacency / edge weights

The GCN propagates information across the brain connectivity graph,
producing d_model-dimensional node (ROI) embeddings.

Architecture:
  Node features: [GM volume, degree centrality, clustering coefficient]
  GCN layers: 2 layers (hidden=256) with residual connections
  Output: (batch, n_rois, d_model)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvolutionLayer(nn.Module):
    """Graph Convolution layer (Kipf & Welling, 2017) with residual.

    H' = sigma(D^{-1/2} (A + I) D^{-1/2} H W + b)
    where A is adjacency, D is degree matrix.
    """

    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if residual and in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = nn.Identity()

    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """Compute D^{-1/2} (A + I) D^{-1/2}."""
        B, N, _ = adj.shape
        # Add self-loops
        adj = adj + torch.eye(N, device=adj.device).unsqueeze(0)
        # Degree matrix
        deg = adj.sum(dim=-1)  # (B, N)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        # D^{-1/2} A D^{-1/2}
        norm_adj = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
        return norm_adj

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, N, in_features)
        adj: (B, N, N) — adjacency / FC matrix
        """
        norm_adj = self.normalize_adjacency(adj)
        out = torch.bmm(norm_adj, x)          # (B, N, in_features)
        out = self.weight(out) + self.bias     # (B, N, out_features)
        out = F.gelu(out)
        out = self.dropout(out)
        if self.residual:
            out = out + self.res_proj(x)
        return out


class BrainGraphFeatureExtractor(nn.Module):
    """Extract node-level features from FC matrix + ROI volumes.

    For each ROI node, computes:
      - GM volume (from sMRI)
      - Degree centrality (from FC)
      - Clustering coefficient (from FC, approximate)
      - Eigenvalue-based centrality (from FC, approximate via power iteration)
    """

    def __init__(self, n_rois: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(4, d_model)  # 4 node features -> d_model
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _degree_centrality(fc: torch.Tensor) -> torch.Tensor:
        """Sum of absolute FC weights per node."""
        return fc.abs().sum(dim=-1)  # (B, N)

    @staticmethod
    def _clustering_coefficient(fc: torch.Tensor) -> torch.Tensor:
        """Approximate clustering: mean of |FC_ij * FC_jk * FC_ik| per node."""
        B, N, _ = fc.shape
        # Threshold to binary adjacency
        adj = (fc.abs() > fc.abs().median(dim=-1, keepdim=True)[0]).float()
        # Clustering: C_i = (1/deg_i) * sum_j sum_k A_ij A_jk A_ik
        triangles = torch.bmm(adj, adj) * adj  # element-wise (i,k) triangles
        clustering = triangles.sum(dim=-1)  # (B, N)
        degree = adj.sum(dim=-1) + 1e-8
        return clustering / degree

    @staticmethod
    def _eigen_centrality(fc: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
        """Approximate eigenvector centrality via power iteration."""
        B, N, _ = fc.shape
        v = torch.ones(B, N, 1, device=fc.device) / math.sqrt(N)
        for _ in range(n_iters):
            v = torch.bmm(fc.abs(), v)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
        return v.squeeze(-1)  # (B, N)

    def forward(self, roi_volumes: Optional[torch.Tensor],
                fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        roi_volumes: (B, N) or None
        fc_matrix:   (B, N, N)
        Returns: (B, N, d_model)
        """
        B, N, _ = fc_matrix.shape

        # Node features
        degree = self._degree_centrality(fc_matrix).unsqueeze(-1)  # (B, N, 1)
        clustering = self._clustering_coefficient(fc_matrix).unsqueeze(-1)
        eigen = self._eigen_centrality(fc_matrix).unsqueeze(-1)

        if roi_volumes is not None:
            volumes = roi_volumes.unsqueeze(-1)  # (B, N, 1)
        else:
            volumes = torch.zeros(B, N, 1, device=fc_matrix.device)

        # Stack: (B, N, 4)
        node_features = torch.cat([volumes, degree, clustering, eigen], dim=-1)
        node_features = self.proj(node_features)
        return self.norm(F.gelu(node_features))


class MRIEncoder(nn.Module):
    """MRI ROI Graph -> Token Sequence Encoder.

    Input:
        roi_volumes: (B, n_rois) — normalized GM volumes
        fc_matrix:   (B, n_rois, n_rois) — functional connectivity
    Output:
        encoded:     (B, n_rois, d_model)
    """

    def __init__(self, d_model: int = 256, n_rois: int = 116,
                 n_gc_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_rois = n_rois

        # Node feature extraction
        self.feature_extractor = BrainGraphFeatureExtractor(n_rois, d_model)

        # GCN layers
        self.gc_layers = nn.ModuleList([
            GraphConvolutionLayer(d_model, d_model, dropout, residual=True)
            for _ in range(n_gc_layers)
        ])

        # Output projection + normalization
        self.output_norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, roi_volumes: Optional[torch.Tensor],
                fc_matrix: torch.Tensor) -> torch.Tensor:
        """
        roi_volumes: (B, n_rois) or None
        fc_matrix:   (B, n_rois, n_rois)
        Returns:     (B, n_rois, d_model)
        """
        # Extract initial node features
        x = self.feature_extractor(roi_volumes, fc_matrix)  # (B, N, d_model)

        # Graph convolution layers
        for gc_layer in self.gc_layers:
            x = gc_layer(x, fc_matrix)

        x = self.output_norm(x)
        x = self.output_dropout(x)
        return x
