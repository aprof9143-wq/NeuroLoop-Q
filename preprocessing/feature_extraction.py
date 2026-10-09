"""
NeuroLoop-Q Feature Extraction & Tensor Assembly
Converts preprocessed EEG + MRI data into model-ready token sequences.

EEG → (n_eeg_tokens, d_model) via conv patch encoder
MRI → (n_mri_tokens, d_model) via GCN + linear projection
Connectivity → auxiliary graph features for the GCN
"""
import numpy as np
import torch
from typing import Optional, Dict
from pathlib import Path

from .config import EEGConfig, MRIConfig, ModelConfig

class FeatureAssembler:
    """Assembles preprocessed EEG + MRI features into model-ready tensors."""

    def __init__(self, eeg_config=None, mri_config=None, model_config=None):
        self.eeg_cfg = eeg_config or EEGConfig()
        self.mri_cfg = mri_config or MRIConfig()
        self.model_cfg = model_config or ModelConfig()
        self.d_model = self.model_cfg.d_model

    # ─── EEG Feature Assembly ───

    def assemble_eeg_tokens(self, stft, bandpower, connectivity):
        """Convert EEG features into token sequence + graph features.
        stft: (n_channels, n_freq_bins, n_time_windows)
        bandpower: (n_channels, n_bands)
        connectivity: (n_bands, n_channels, n_channels)
        """
        n_channels, n_freq_bins, n_time_windows = stft.shape
        log_stft = np.log1p(stft)
        patches = log_stft.reshape(n_channels, -1)
        patches = torch.FloatTensor(patches)
        patches = (patches - patches.mean(dim=-1, keepdim=True)) / \
                   (patches.std(dim=-1, keepdim=True) + 1e-8)

        avg_connectivity = connectivity.mean(axis=0)
        avg_connectivity = torch.FloatTensor(avg_connectivity)

        bp = torch.FloatTensor(bandpower)
        bp = (bp - bp.mean(dim=0, keepdim=True)) / (bp.std(dim=0, keepdim=True) + 1e-8)

        return {"tokens": patches, "connectivity": avg_connectivity, "bandpower": bp}

    # ─── MRI Feature Assembly ───

    def assemble_mri_tokens(self, roi_volumes=None, fc_matrix=None):
        """Convert MRI features into token sequence + graph features."""
        result = {}
        if roi_volumes is not None:
            volumes = torch.FloatTensor(roi_volumes)
            volumes = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            result["volumes"] = volumes

        if fc_matrix is not None:
            fc = torch.FloatTensor(fc_matrix)
            result["fc_matrix"] = fc
            n_rois = fc.shape[0]
            triu_indices = torch.triu_indices(n_rois, n_rois, offset=1)
            edge_features = fc[triu_indices[0], triu_indices[1]]
            result["edge_features"] = edge_features

        return result

    # ─── Combined Assembly ───

    def assemble(self, eeg_data=None, mri_data=None):
        assembled = {}
        if eeg_data is not None:
            assembled["eeg"] = self.assemble_eeg_tokens(
                stft=eeg_data["stft"], bandpower=eeg_data["bandpower"],
                connectivity=eeg_data["connectivity"])
        if mri_data is not None:
            assembled["mri"] = self.assemble_mri_tokens(
                roi_volumes=mri_data.get("roi_volumes"),
                fc_matrix=mri_data.get("fc_matrix"))
        return assembled

    # ─── Batch Assembly (for DataLoader) ───

    def assemble_batch(self, batch):
        """Assemble a batch from the DataLoader into model-ready tensors."""
        model_input = {}
        batch_size = len(batch.get("label_ad", batch.get("label_asd", [1])))

        # ── EEG ──
        if batch.get("eeg_stft") is not None:
            eeg_stft = batch["eeg_stft"]
            eeg_bp = batch["eeg_bandpower"]
            eeg_conn = batch["eeg_connectivity"]
            B, C, F, T = eeg_stft.shape
            eeg_tokens = eeg_stft.reshape(B, C, -1)
            eeg_tokens = (eeg_tokens - eeg_tokens.mean(dim=-1, keepdim=True)) / \
                         (eeg_tokens.std(dim=-1, keepdim=True) + 1e-8)
            eeg_conn_avg = eeg_conn.mean(dim=1)
            model_input["eeg_tokens"] = eeg_tokens
            model_input["eeg_connectivity"] = eeg_conn_avg
            model_input["eeg_bandpower"] = eeg_bp
        else:
            n_channels = self.eeg_cfg.n_channels
            n_freq_bins = 23
            n_time_windows = 15
            n_bands = self.eeg_cfg.n_bands
            model_input["eeg_tokens"] = torch.zeros(batch_size, n_channels, n_freq_bins * n_time_windows)
            model_input["eeg_connectivity"] = torch.eye(n_channels).unsqueeze(0).expand(batch_size, -1, -1)
            model_input["eeg_bandpower"] = torch.zeros(batch_size, n_channels, n_bands)
            model_input["eeg_mask"] = torch.zeros(batch_size, 1)

        # ── MRI ──
        if batch.get("mri_volumes") is not None or batch.get("mri_fc") is not None:
            if batch.get("mri_volumes") is not None:
                volumes = batch["mri_volumes"]
                volumes = (volumes - volumes.mean(dim=-1, keepdim=True)) / \
                           (volumes.std(dim=-1, keepdim=True) + 1e-8)
                model_input["mri_volumes"] = volumes
            else:
                model_input["mri_volumes"] = torch.zeros(batch_size, self.mri_cfg.n_rois)
            if batch.get("mri_fc") is not None:
                model_input["mri_fc"] = batch["mri_fc"]
            else:
                model_input["mri_fc"] = torch.eye(self.mri_cfg.n_rois).unsqueeze(0).expand(batch_size, -1, -1)
            model_input["mri_mask"] = torch.ones(batch_size, 1)
        else:
            n_rois = self.mri_cfg.n_rois
            model_input["mri_volumes"] = torch.zeros(batch_size, n_rois)
            model_input["mri_fc"] = torch.eye(n_rois).unsqueeze(0).expand(batch_size, -1, -1)
            model_input["mri_mask"] = torch.zeros(batch_size, 1)

        # ── Labels ──
        if "label_ad" in batch:
            model_input["label_ad"] = batch["label_ad"]
        if "label_asd" in batch:
            model_input["label_asd"] = batch["label_asd"]
        if "age" in batch:
            model_input["age"] = batch["age"]
        if "sex" in batch:
            model_input["sex"] = batch["sex"]

        return model_input

def run_feature_extraction(eeg_processed_dir, mri_processed_dir, output_dir,
                           eeg_config=None, mri_config=None, model_config=None):
    """Run feature extraction on all preprocessed subjects."""
    assembler = FeatureAssembler(eeg_config, mri_config, model_config)
    eeg_dir = Path(eeg_processed_dir)
    mri_dir = Path(mri_processed_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_subjects = {f.stem.replace("_eeg", "") for f in eeg_dir.glob("*_eeg.npz")}
    mri_subjects = {f.stem.replace("_mri", "") for f in mri_dir.glob("*_mri.npz")}
    all_subjects = eeg_subjects | mri_subjects

    print(f"Feature extraction: {len(all_subjects)} subjects "
          f"(EEG: {len(eeg_subjects)}, MRI: {len(mri_subjects)})")

    for sid in sorted(all_subjects):
        out_file = out_dir / f"{sid}_features.npz"
        if out_file.exists():
            continue

        eeg_data = None
        mri_data = None

        eeg_file = eeg_dir / f"{sid}_eeg.npz"
        if eeg_file.exists():
            eeg_npz = np.load(eeg_file)
            eeg_data = {
                "stft": eeg_npz["stft"][0],
                "bandpower": eeg_npz["bandpower"][0],
                "connectivity": eeg_npz["connectivity"][0],
            }

        mri_file = mri_dir / f"{sid}_mri.npz"
        if mri_file.exists():
            mri_npz = np.load(mri_file)
            mri_data = {}
            if "roi_volumes" in mri_npz:
                mri_data["roi_volumes"] = mri_npz["roi_volumes"]
            if "fc_matrix" in mri_npz:
                mri_data["fc_matrix"] = mri_npz["fc_matrix"]

        features = assembler.assemble(eeg_data, mri_data)

        save_dict = {}
        if "eeg" in features:
            save_dict["eeg_tokens"] = features["eeg"]["tokens"].numpy()
            save_dict["eeg_connectivity"] = features["eeg"]["connectivity"].numpy()
            save_dict["eeg_bandpower"] = features["eeg"]["bandpower"].numpy()
        if "mri" in features:
            if "volumes" in features["mri"]:
                save_dict["mri_volumes"] = features["mri"]["volumes"].numpy()
            if "fc_matrix" in features["mri"]:
                save_dict["mri_fc"] = features["mri"]["fc_matrix"].numpy()
            if "edge_features" in features["mri"]:
                save_dict["mri_edge_features"] = features["mri"]["edge_features"].numpy()

        np.savez_compressed(out_file, save_dict)

    print(f"✓ Feature extraction complete: {out_dir}")

if __name__ == "__main__":
    run_feature_extraction(
        eeg_processed_dir="./data/processed/ADNI/eeg",
        mri_processed_dir="./data/processed/ADNI/mri",
        output_dir="./data/processed/ADNI/features",
    )
