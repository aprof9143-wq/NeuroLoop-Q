"""
NeuroLoop-Q Preprocessing Master Pipeline
Run this script to execute the full preprocessing pipeline:

  1. EEG preprocessing (ADNI subjects with EEG)
  2. MRI preprocessing (ADNI + ABIDE-II)
  3. Feature extraction & tensor assembly
  4. Dataset statistics & quality report

Usage:
  python -m preprocessing.run_pipeline --dataset ADNI
  python -m preprocessing.run_pipeline --dataset ABIDE
  python -m preprocessing.run_pipeline --dataset ALL
"""
import argparse
import time
from pathlib import Path

from .config import (
    EEGConfig, MRIConfig, ModelConfig,
    ADNI_DIR, ABIDE_DIR, PROCESSED_DIR,
)
from .eeg_preprocessing import EEGPreprocessor
from .mri_preprocessing import MRIPreprocessor
from .dataset_adni import ADNIDatasetBuilder
from .dataset_abide import ABIDEDatasetBuilder
from .feature_extraction import run_feature_extraction

def preprocess_adni(skip_ants=True):
    """Full ADNI preprocessing pipeline."""
    print("=" * 70)
    print("  NeuroLoop-Q — ADNI Preprocessing Pipeline")
    print("=" * 70)

    builder = ADNIDatasetBuilder()
    subject_list = builder.build_subject_list()

    # ── EEG Preprocessing ──
    eeg_subjects = subject_list[subject_list["eeg_path"].notna()]
    if len(eeg_subjects) > 0:
        print(f"\n── EEG Preprocessing ({len(eeg_subjects)} subjects) ──")
        eeg_cfg = EEGConfig()
        preprocessor = EEGPreprocessor(eeg_cfg)
        eeg_output = PROCESSED_DIR / "ADNI" / "eeg"
        eeg_output.mkdir(parents=True, exist_ok=True)

        for _, row in eeg_subjects.iterrows():
            sid = row["subject_id"]
            out_file = eeg_output / f"{sid}_eeg.npz"
            if out_file.exists():
                continue
            try:
                result = preprocessor.process(row["eeg_path"])
                np.savez_compressed(
                    out_file,
                    stft=result["stft"],
                    bandpower=result["bandpower"],
                    connectivity=result["connectivity"],
                    n_epochs=result["n_epochs"],
                    n_channels=result["n_channels"],
                    sfreq=result["sfreq"],
                )
            except Exception as e:
                print(f"  ERROR {sid}: {e}")

    # ── MRI Preprocessing ──
    mri_subjects = subject_list[
        (subject_list["t1_path"].notna()) |
        (subject_list["bold_path"].notna())
    ]
    if len(mri_subjects) > 0:
        print(f"\n── MRI Preprocessing ({len(mri_subjects)} subjects) ──")
        mri_cfg = MRIConfig()
        preprocessor = MRIPreprocessor(mri_cfg)
        mri_output = PROCESSED_DIR / "ADNI" / "mri"
        mri_output.mkdir(parents=True, exist_ok=True)

        for _, row in mri_subjects.iterrows():
            sid = row["subject_id"]
            out_file = mri_output / f"{sid}_mri.npz"
            if out_file.exists():
                continue
            try:
                result = preprocessor.process(
                    t1_path=row.get("t1_path"),
                    bold_path=row.get("bold_path"),
                    confounds_path=row.get("confounds_path"),
                    skip_ants=skip_ants,
                )
                save_dict = {"n_rois": result["n_rois"]}
                if "roi_volumes" in result:
                    save_dict["roi_volumes"] = result["roi_volumes"]
                if "fc_matrix" in result:
                    save_dict["fc_matrix"] = result["fc_matrix"]
                np.savez_compressed(out_file, save_dict)
            except Exception as e:
                print(f"  ERROR {sid}: {e}")

    # ── Feature Extraction ──
    print("\n── Feature Extraction (ADNI) ──")
    run_feature_extraction(
        eeg_processed_dir=str(PROCESSED_DIR / "ADNI" / "eeg"),
        mri_processed_dir=str(PROCESSED_DIR / "ADNI" / "mri"),
        output_dir=str(PROCESSED_DIR / "ADNI" / "features"),
    )

    print("\n✓ ADNI preprocessing complete.")

def preprocess_abide():
    """Full ABIDE-II preprocessing pipeline."""
    print("=" * 70)
    print("  NeuroLoop-Q — ABIDE-II Preprocessing Pipeline")
    print("=" * 70)

    builder = ABIDEDatasetBuilder()
    subject_list = builder.build_subject_list()

    print(f"\n── MRI Preprocessing ({len(subject_list)} subjects) ──")
    mri_cfg = MRIConfig()
    preprocessor = MRIPreprocessor(mri_cfg)
    mri_output = PROCESSED_DIR / "ABIDE" / "mri"
    mri_output.mkdir(parents=True, exist_ok=True)

    for _, row in subject_list.iterrows():
        sid = row["subject_id"]
        out_file = mri_output / f"{sid}_mri.npz"
        if out_file.exists():
            continue
        try:
            result = preprocessor.process(
                t1_path=row.get("t1_path"),
                bold_path=row.get("bold_path"),
                confounds_path=row.get("confounds_path"),
                skip_ants=True,
            )
            save_dict = {"n_rois": result["n_rois"]}
            if "roi_volumes" in result:
                save_dict["roi_volumes"] = result["roi_volumes"]
            if "fc_matrix" in result:
                save_dict["fc_matrix"] = result["fc_matrix"]
            np.savez_compressed(out_file, save_dict)
        except Exception as e:
            print(f"  ERROR {sid}: {e}")

    print("\n── Feature Extraction (ABIDE) ──")
    run_feature_extraction(
        eeg_processed_dir=str(PROCESSED_DIR / "ABIDE" / "eeg"),
        mri_processed_dir=str(PROCESSED_DIR / "ABIDE" / "mri"),
        output_dir=str(PROCESSED_DIR / "ABIDE" / "features"),
    )

    print("\n✓ ABIDE-II preprocessing complete.")

def print_quality_report():
    """Print a quality report on the processed data."""
    print("\n" + "=" * 70)
    print("  NeuroLoop-Q — Preprocessing Quality Report")
    print("=" * 70)

    import numpy as np

    for dataset_name in ["ADNI", "ABIDE"]:
        feat_dir = PROCESSED_DIR / dataset_name / "features"
        if not feat_dir.exists():
            continue

        files = list(feat_dir.glob("*_features.npz"))
        print(f"\n  {dataset_name}: {len(files)} subjects with features")

        if len(files) == 0:
            continue

        sample = np.load(files[0])
        print(f"  Sample shapes ({files[0].stem}):")
        for key in sample.files:
            print(f"    {key}: {sample[key].shape}")

        nan_count = 0
        inf_count = 0
        for f in files:
            data = np.load(f)
            for key in data.files:
                arr = data[key]
                if np.any(np.isnan(arr)):
                    nan_count += 1
                if np.any(np.isinf(arr)):
                    inf_count += 1

        if nan_count == 0 and inf_count == 0:
            print(f"  ✓ No NaNs or Infs detected")
        else:
            print(f"  ⚠ NaNs: {nan_count}, Infs: {inf_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroLoop-Q Preprocessing")
    parser.add_argument("--dataset", choices=["ADNI", "ABIDE", "ALL"], default="ALL")
    parser.add_argument("--skip-ants", action="store_true", default=True)
    parser.add_argument("--quality-report", action="store_true", default=True)
    args = parser.parse_args()

    start_time = time.time()

    if args.dataset in ("ADNI", "ALL"):
        preprocess_adni(skip_ants=args.skip_ants)
    if args.dataset in ("ABIDE", "ALL"):
        preprocess_abide()
    if args.quality_report:
        print_quality_report()

    elapsed = time.time() - start_time
    print(f"\nTotal preprocessing time: {elapsed/60:.1f} minutes")
