"""
ABIDE-II Dataset Loader for NeuroLoop-Q
Handles: subject listing, label mapping (TD/ASD), file discovery,
         and pairing of fMRI with sMRI data.

ABIDE data access: http://fcon_1000.projects.nitrc.org/indi/abide/
Download preprocessed data from:
  http://preprocessed-connectomes-project.org/abide/
  (CPAC pipeline, bandpass filtering, no global signal regression recommended)
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

from .config import ABIDE_DIR, PROCESSED_DIR

class ABIDEDatasetBuilder:
    """Builds subject-level file lists and labels from ABIDE-II."""

    LABEL_MAP = {1: 1, 2: 0}  # 1=ASD, 2=TD

    def __init__(self, abide_dir: str = None):
        self.abide_dir = Path(abide_dir or ABIDE_DIR)
        self.processed_dir = PROCESSED_DIR / "ABIDE"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_phenotypic(self) -> pd.DataFrame:
        pheno_file = self.abide_dir / "ABIDEII_Composite_Pheno.csv"
        if not pheno_file.exists():
            pheno_file = self.abide_dir / "ABIDEI_Pheno.csv"
        if not pheno_file.exists():
            raise FileNotFoundError(
                f"Phenotypic file not found in {self.abide_dir}\n"
                "Download from: http://fcon_1000.projects.nitrc.org/indi/abide/"
            )

        df = pd.read_csv(pheno_file)
        if "DX_GROUP" in df.columns:
            df["label"] = df["DX_GROUP"].map(self.LABEL_MAP)
        else:
            raise ValueError("DX_GROUP column not found in phenotypic file")

        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "age" in cl:
                col_map[col] = "age"
            elif "sex" in cl:
                col_map[col] = "sex"
            elif "subject" in cl or "sub" in cl:
                col_map[col] = "subject_id"
        df = df.rename(columns=col_map)

        if "subject_id" not in df.columns:
            for c in ["SUB_ID", "Subject", "participant_id"]:
                if c in df.columns:
                    df["subject_id"] = df[c].astype(str)
                    break

        df["sex"] = (df["sex"] == 1).astype(int)
        df = df[["subject_id", "label", "age", "sex"]]
        df = df[df["label"].notna()]

        print(f"  ABIDE labels: {len(df)} subjects — "
              f"TD: {(df.label==0).sum()}, ASD: {(df.label==1).sum()}")
        return df

    def find_mri_files(self) -> dict:
        cpac_dir = self.abide_dir / "CPAC" / "filt_noglobal"
        subject_files = {}
        if cpac_dir.exists():
            for subj_dir in cpac_dir.iterdir():
                if subj_dir.is_dir():
                    sid = subj_dir.name
                    bold_files = list(subj_dir.glob("*rest*.nii.gz")) + \
                                 list(subj_dir.glob("*functional*.nii.gz")) + \
                                 list(subj_dir.glob("*bold*.nii.gz"))
                    if bold_files:
                        subject_files[sid] = {"bold": str(bold_files[0])}
                    t1_files = list(subj_dir.glob("*anat*.nii.gz")) + \
                               list(subj_dir.glob("*T1*.nii.gz"))
                    if t1_files:
                        subject_files[sid]["t1"] = str(t1_files[0])
        print(f"  Found fMRI files for {len(subject_files)} subjects")
        return subject_files

    def build_subject_list(self) -> pd.DataFrame:
        pheno = self.load_phenotypic()
        mri_files = self.find_mri_files()

        rows = []
        for _, row in pheno.iterrows():
            sid = str(row["subject_id"])
            entry = {
                "subject_id": sid, "label": int(row["label"]),
                "age": row.get("age", np.nan), "sex": int(row.get("sex", 0)),
                "t1_path": mri_files.get(sid, {}).get("t1"),
                "bold_path": mri_files.get(sid, {}).get("bold"),
                "eeg_path": None,
            }
            rows.append(entry)

        df = pd.DataFrame(rows)
        df = df[df["bold_path"].notna()]
        print(f"  Final: {len(df)} subjects with fMRI")
        return df

class ABIDEDataset(Dataset):
    """PyTorch Dataset for preprocessed ABIDE-II data."""

    def __init__(self, subject_list: pd.DataFrame, processed_dir: str = None):
        self.subjects = subject_list.reset_index(drop=True)
        self.processed_dir = Path(processed_dir or PROCESSED_DIR / "ABIDE")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        row = self.subjects.iloc[idx]
        sid = row["subject_id"]
        data = {"label_asd": int(row["label"])}

        mri_file = self.processed_dir / f"{sid}_mri.npz"
        if mri_file.exists():
            mri = np.load(mri_file)
            data["mri_volumes"] = torch.FloatTensor(mri["roi_volumes"]) if "roi_volumes" in mri else None
            data["mri_fc"] = torch.FloatTensor(mri["fc_matrix"]) if "fc_matrix" in mri else None
        else:
            data["mri_volumes"] = None
            data["mri_fc"] = None

        # ABIDE has no EEG — structured zeros (gated fusion down-weights)
        data["eeg_stft"] = None
        data["eeg_bandpower"] = None
        data["eeg_connectivity"] = None

        data["age"] = torch.FloatTensor([float(row.get("age", 0) or 0)])
        data["sex"] = torch.LongTensor([int(row.get("sex", 0))])
        return data

def collate_abide(batch):
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

def get_abide_dataloaders(batch_size=16, train_ratio=0.7, val_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    builder = ABIDEDatasetBuilder()
    subjects = builder.build_subject_list()

    train_df, temp_df = train_test_split(
        subjects, test_size=(1 - train_ratio), stratify=subjects["label"], random_state=seed)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed)

    train_ds = ABIDEDataset(train_df)
    val_ds = ABIDEDataset(val_df)
    test_ds = ABIDEDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_abide)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_abide)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_abide)

    print(f"  ABIDE Dataloaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader
