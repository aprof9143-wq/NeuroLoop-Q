"""
NeuroLoop-Q Central Configuration
Cognivance Labs — Multi-Modal Neural Fusion for AD & ASD
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

─── Paths ───
PROJECT_ROOT = Path("/data/neuroloop_q")
RAW_DATA_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

Dataset-specific raw paths (adjust to your local layout)
ADNI_DIR = RAW_DATA_DIR / "ADNI"
ABIDE_DIR = RAW_DATA_DIR / "ABIDE-II"

─── EEG Configuration ───
@dataclass
class EEGConfig:
    sampling_rate: int = 500          # Hz (downsample if higher)
    n_channels: int = 128            # 128-channel montage (64 also supported)
    n_bands: int = 5                  # delta, theta, alpha, beta, gamma
    # Frequency bands (Hz): (name, low, high)
    freq_bands: list = field(default_factory=lambda: [
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta",  13.0, 30.0),
        ("gamma", 30.0, 45.0),
    ])
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    notch_freq: float = 50.0          # power-line (60 for US)
    ica_n_components: int = 25
    epoch_duration: float = 2.0       # seconds per epoch
    epoch_overlap: float = 0.5        # 50% overlap
    reference: str = "average"        # common average reference
    stft_window: int = 256            # samples
    stft_overlap: int = 128            # 50% overlap
    connectivity_metric: str = "plv"  # phase-locking value

─── MRI Configuration ───
@dataclass
class MRIConfig:
    # Structural MRI
    atlas: str = "aal"                 # AAL3 atlas (116 ROIs)
    n_rois: int = 116
    voxel_size: tuple = (2.0, 2.0, 2.0)  # mm, for resampling
    # Functional MRI
    tr: float = 2.0                   # repetition time (s)
    low_pass: float = 0.08            # Hz, for bandpass
    high_pass: float = 0.01           # Hz
    smoothing_fwhm: float = 6.0       # mm
    n_confounds: int = 24             # motion params + derivatives + WM/CSF
    connectivity_metric: str = "pearson"  # ROI-level FC
    # Graph
    sparsity_threshold: float = 0.2   # keep top 20% of connections

─── Model Configuration ───
@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_core_layers: int = 4             # weight-tied layers
    max_loops: int = 12
    min_loops: int = 3
    ffn_expansion: int = 2             # reduced from standard 4
    dropout: float = 0.1
    tau: float = 0.1                  # confidence halting threshold
    n_classes_ad: int = 3              # CN, MCI, AD
    n_classes_asd: int = 2             # TD, ASD
    n_qubits: int = 8
    vqc_layers: int = 3

─── Training Configuration ───
@dataclass
class TrainConfig:
    batch_size: int = 16
    base_lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    max_epochs: int = 200
    patience: int = 20
    qrlo_update_freq: int = 50        # Q-RLO policy update every N steps
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    distill_temperature: float = 4.0
    # Loss weights
    lambda_asd: float = 0.8
    lambda_age: float = 0.1
    lambda_sex: float = 0.1
    lambda_contrastive: float = 0.3
