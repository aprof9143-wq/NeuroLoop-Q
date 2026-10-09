"""
ADNI Dataset Loader for NeuroLoop-Q
Handles: subject listing, label mapping (CN/MCI/AD), file discovery,
         and pairing of EEG (if available) with MRI data.

ADNI data access: http://adni.loni.usc.edu (requires registration)
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

from .config import ADNI_DIR, PROCESSED_DIR

class ADNIDatasetBuilder:
    """Builds subject-level file lists and labels from ADNI directory."""

    LABEL_MAP = {
        "CN": 0, "SMC": 0, "EMCI": 1, "LMCI": 1, "MCI": 1, "AD": 2, "Dementia": 2,
    }

    def __init__(self, adni_dir: str = None):
        self.adni_dir = Path(adni_dir or ADNI_DIR)
        self.processed_dir = PROCESSED_DIR / "ADNI"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_clinical_labels(self) -> pd.DataFrame:
        dx_file = self.adni_dir / "clinical" / "DXSUM_PDXCON_ADNIALL.csv"
        if not dx_file.exists():
            raise FileNotFoundError(
                f"Clinical file not found: {dx_file}\n"
                "Download from ADNI: Study Files → Clinical → Diagnostic Summary"
            )

        df = pd.read_csv(dx_file)
        df = df.sort_values(["PTID", "EXAMDATE"])
        df = df.groupby("PTID").last().reset_index()

        def map_label(row):
            for col in ["DXGROUP", "DXCURREN", "DIAGNOSIS"]:
                if col in row.index and pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if val in self.LABEL_MAP:
                        return self.LABEL_MAP[val]
            return -1

        df["label"] = df.apply(map_label, axis=1)
        df = df[df["label"] >= 0]
        df = df[["PTID", "label", "AGE", "PTGENDER"]]
        df.columns = ["subject_id", "label", "age", "sex"]
        df["sex"] = (df["sex"].str.lower() == "male").astype(int)

        print(f"  ADNI labels: {len(df)} subjects — "
              f"CN: {(df.label==0).sum()}, MCI: {(df.label==1).sum()}, "
              f"AD: {(df.label==2).sum()}")
        return df

    def find_mri_files(self) -> dict:
        mri_dir = self.adni_dir / "MRI"
        subject_files = {}
        for root, dirs, files in os.walk(mri_dir):
            for f in files:
                if f.endswith((".nii", ".nii.gz")):
                    parts = Path(root).parts
                    for p in parts:
                        if p.startswith("0") and len(p) >= 4:
                            sid = p
                            if sid not in subject_files:
                                subject_files[sid] = {}
                            if "T1" in f or "t1" in f:
                                subject_files[sid]["t1"] = os.path.join(root, f)
                            elif "bold" in f.lower() or "rest" in f.lower():
                                subject_files[sid]["bold"] = os.path.join(root, f)
        print(f"  Found MRI files for {len(subject_files)} subjects")
        return subject_files

    def find_eeg_files(self) -> dict:
        eeg_dir = self.adni_dir / "EEG"
        subject_files = {}
        if eeg_dir.exists():
            for root, dirs, files in os.walk(eeg_dir):
                for f in files:
                    if f.endswith((".set", ".edf", ".fif", ".vhdr")):
                        parts = Path(root).parts
                        for p in parts:
                            if p.startswith("0") and len(p) >= 4:
                                sid = p
                                subject_files[sid] = os.path.join(root, f)
        print(f"  Found EEG files for {len(subject_files)} subjects")
        return subject_files

    def build_subject_list(self) -> pd.DataFrame:
        labels = self.load_clinical_labels()
        mri_files = self.find_mri_files()
        eeg_files = self.find_eeg_files()

        rows = []
        for _, row in labels.iterrows():
            sid = row["subject_id"]
            entry = {
                "subject_id": sid, "label": row["label"],
                "age": row["age"], "sex": row["sex"],
                "t1_path": mri_files.get(sid, {}).get("t1"),
                "bold_path": mri_files.get(sid, {}).get("bold"),
                "eeg_path": eeg_files.get(sid),
            }
            rows.append(entry)

        df = pd.DataFrame(rows)
        df = df[df["t1_path"].notna() | df["eeg_path"].notna()]
        print(f"  Final: {len(df)} subjects with at least one modality")
        return df

class ADNIDataset(Dataset):
    """PyTorch Dataset for preprocessed ADNI data."""

    def __init__(self, subject_list: pd.DataFrame, processed_dir: str = None,
                 max_epochs_per_subject: int = 50):
        self.subjects = subject_list.reset_index(drop=True)
        self.processed_dir = Path(processed_dir or PROCESSED_DIR / "ADNI")
        self.max_epochs = max_epochs_per_subject

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        row = self.subjects.iloc[idx]
        sid = row["subject_id"]
        data = {"label_ad": int(row["label"])}

        eeg_file = self.processed_dir / f"{sid}_eeg.npz"
        if eeg_file.exists():
            eeg = np.load(eeg_file)
            n_epochs = min(eeg["stft"].shape[0], self.max_epochs)
            ep_idx = np.random.randint(n_epochs)
            data["eeg_stft"] = torch.FloatTensor(eeg["stft"][ep_idx])
            data["eeg_bandpower"] = torch.FloatTensor(eeg["bandpower"][ep_idx])
            data["eeg_connectivity"] = torch.FloatTensor(eeg["connectivity"][ep_idx])
        else:
            data["eeg_stft"] = None
            data["eeg_bandpower"] = None
            data["eeg_connectivity"] = None

        mri_file = self.processed_dir / f"{sid}_mri.npz"
        if mri_file.exists():
            mri = np.load(mri_file)
            data["mri_volumes"] = torch.FloatTensor(mri["roi_volumes"]) if "roi_volumes" in mri else None
            data["mri_fc"] = torch.FloatTensor(mri["fc_matrix"]) if "fc_matrix" in mri else None
        else:
            data["mri_volumes"] = None
            data["mri_fc"] = None

        data["age"] = torch.FloatTensor([row.get("age", 0)])
        data["sex"] = torch.LongTensor([int(row.get("sex", 0))])
        return data

def collate_adni(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        vals = [b[key] for b in batch]
        if all(v is None for v in vals):
            out[key] = None
        elif isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out

def get_adni_dataloaders(batch_size=16, train_ratio=0.7, val_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    builder = ADNIDatasetBuilder()
    subjects = builder.build_subject_list()

    train_df, temp_df = train_test_split(
        subjects, test_size=(1 - train_ratio), stratify=subjects["label"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed)

    train_ds = ADNIDataset(train_df)
    val_ds = ADNIDataset(val_df)
    test_ds = ADNIDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_adni)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_adni)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_adni)

    print(f"  ADNI Dataloaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
