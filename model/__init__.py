"""
NeuroLoop-Q Model Package
Cognivance Labs — Multi-Modal Neural Fusion for AD & ASD

Modules:
  eeg_encoder        — Conv patch encoder + connectivity-aware attention
  mri_encoder        — GCN over ROI brain graph
  cmssf              — Cross-Modal Spectral-Spatial Fusion (gated)
  looped_transformer  — Weight-tied recurrent transformer core
  neuroloop_q        — Full model assembly
  quantum_rlo        — VQC-based RL optimizer (PennyLane)
"""
from .eeg_encoder import EEGEncoder, ConnectivityAwareAttention, RotaryPositionalEmbedding
from .mri_encoder import MRIEncoder, GraphConvolutionLayer, BrainGraphFeatureExtractor
from .cmssf import CMSSFBlock, CrossModalAttention, GatedFusion
from .looped_transformer import LoopedTransformerCore, TransformerBlock
from .neuroloop_q import NeuroLoopQ, ClassificationHead
from .quantum_rlo import QRLOptimizer, VariationalQuantumCircuit

__all__ = [
    "EEGEncoder", "ConnectivityAwareAttention", "RotaryPositionalEmbedding",
    "MRIEncoder", "GraphConvolutionLayer", "BrainGraphFeatureExtractor",
    "CMSSFBlock", "CrossModalAttention", "GatedFusion",
    "LoopedTransformerCore", "TransformerBlock",
    "NeuroLoopQ", "ClassificationHead",
    "QRLOptimizer", "VariationalQuantumCircuit",
]
