"""
NeuroLoop-Q Preprocessing Package
Cognivance Labs — Multi-Modal Neural Fusion for AD & ASD

Modules:
  eeg_preprocessing   — MNE-Python rs-EEG pipeline (filter, ICA, epoch, STFT, PLV)
  mri_preprocessing   — Nilearn/ANTs sMRI + fMRI pipeline (skull-strip, segment, FC)
  dataset_adni        — ADNI loader (CN/MCI/AD labels, MRI + EEG pairing)
  dataset_abide       — ABIDE-II loader (TD/ASD labels, fMRI)
  feature_extraction  — Cross-modal tensor assembly for model input
  config              — Central configuration dataclasses
"""
from .config import (
    EEGConfig, MRIConfig, ModelConfig, TrainConfig,
    PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DIR,
    ADNI_DIR, ABIDE_DIR,
)
from .eeg_preprocessing import EEGPreprocessor
from .mri_preprocessing import MRIPreprocessor
from .dataset_adni import ADNIDatasetBuilder, ADNIDataset, get_adni_dataloaders
from .dataset_abide import ABIDEDatasetBuilder, ABIDEDataset, get_abide_dataloaders
from .feature_extraction import FeatureAssembler, run_feature_extraction

__all__ = [
    "EEGConfig", "MRIConfig", "ModelConfig", "TrainConfig",
    "PROJECT_ROOT", "RAW_DATA_DIR", "PROCESSED_DIR",
    "ADNI_DIR", "ABIDE_DIR",
    "EEGPreprocessor", "MRIPreprocessor",
    "ADNIDatasetBuilder", "ADNIDataset", "get_adni_dataloaders",
    "ABIDEDatasetBuilder", "ABIDEDataset", "get_abide_dataloaders",
    "FeatureAssembler", "run_feature_extraction",
]
