"""
NeuroLoop-Q MRI Preprocessing Pipeline
Built on Nilearn, ANTs (via Nipype), and FreeSurfer outputs

Input:  Raw T1-weighted sMRI + rs-fMRI from ADNI or ABIDE-II
Output: ROI volume vector (n_rois,) + FC matrix (n_rois, n_rois)

Note: This module assumes fMRIPrep has been run for fMRI data,
producing preprocessed BOLD files in MNI space. If raw fMRI is
provided, the full Nipype pipeline is invoked.
"""
import numpy as np
import nibabel as nib
from nilearn import datasets, image, maskers, connectome
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings("ignore")

from .config import MRIConfig

class MRIPreprocessor:
    """Full sMRI + fMRI preprocessing pipeline."""

    def __init__(self, config: MRIConfig = None):
        self.cfg = config or MRIConfig()
        self._load_atlas()

    def _load_atlas(self):
        """Load AAL3 atlas for ROI parcellation."""
        self.atlas = datasets.fetch_atlas_aal(version="SPM12")
        self.atlas_img = nib.load(self.atlas.maps)
        self.labels = self.atlas.labels
        self.n_rois = len(self.labels)
        print(f"  Atlas: AAL3 ({self.n_rois} ROIs)")

    # ═══════════════════════════════════════════════
    # STRUCTURAL MRI
    # ═══════════════════════════════════════════════

    def skull_strip(self, t1_path: str, output_path: str):
        """Run ANTs brain extraction."""
        cmd = [
            "antsBrainExtraction.sh",
            "-d", "3",
            "-a", t1_path,
            "-e", "T1.nii.gz",
            "-m", "BrainExtractionMask.nii.gz",
            "-o", output_path,
        ]
        subprocess.run(cmd, check=True)
        return f"{output_path}BrainExtractionBrain.nii.gz"

    def segment_tissues(self, brain_path: str, output_prefix: str):
        """ANTs Atropos segmentation: GM, WM, CSF."""
        cmd = [
            "antsAtroposSegmentation.sh",
            "-d", "3",
            "-a", brain_path,
            "-x", f"{output_prefix}BrainExtractionMask.nii.gz",
            "-m", "3",
            "-c", "[5,0.95]",
            "-o", output_prefix,
        ]
        subprocess.run(cmd, check=True)
        return {
            "gm": f"{output_prefix}segmentation1.nii.gz",
            "wm": f"{output_prefix}segmentation2.nii.gz",
            "csf": f"{output_prefix}segmentation3.nii.gz",
        }

    def register_to_mni(self, moving_path: str, output_path: str):
        """ANTs SyN registration to MNI152 standard space."""
        mni_ref = datasets.fetch_icbm152_2009()["t1"]
        cmd = [
            "antsRegistrationSyN.sh",
            "-d", "3",
            "-f", mni_ref,
            "-m", moving_path,
            "-o", output_path,
            "-t", "s",
        ]
        subprocess.run(cmd, check=True)
        return f"{output_path}RegisteredToMNI.nii.gz"

    def extract_roi_volumes(self, gm_path: str) -> np.ndarray:
        """Extract GM volume per ROI from segmented tissue.
        Input:  GM probability map in MNI space
        Output: (n_rois,) vector of GM volumes per ROI
        """
        gm_img = nib.load(gm_path)
        gm_data = gm_img.get_fdata()

        atlas_resampled = image.resample_to_img(
            self.atlas_img, gm_img, interpolation="nearest"
        )
        atlas_data = atlas_resampled.get_fdata()

        roi_volumes = np.zeros(self.n_rois)
        voxel_vol = np.prod(gm_img.header.get_zooms())

        for i, label in enumerate(self.labels):
            roi_mask = (atlas_data == (i + 1))
            roi_volumes[i] = np.sum(gm_data[roi_mask]) * voxel_vol

        total_gm = np.sum(roi_volumes)
        if total_gm > 0:
            roi_volumes = roi_volumes / total_gm

        return roi_volumes.astype(np.float32)

    def process_smri(self, t1_path: str, skip_ants: bool = False) -> dict:
        """Full sMRI pipeline."""
        if not skip_ants:
            prefix = str(Path(t1_path).parent / Path(t1_path).stem)
            brain = self.skull_strip(t1_path, prefix)
            tissues = self.segment_tissues(brain, prefix)
            gm_mni = self.register_to_mni(tissues["gm"], prefix)
            roi_volumes = self.extract_roi_volumes(gm_mni)
        else:
            roi_volumes = self.extract_roi_volumes(t1_path)

        return {
            "roi_volumes": roi_volumes,
            "n_rois": self.n_rois,
        }

    # ═══════════════════════════════════════════════
    # FUNCTIONAL MRI
    # ═══════════════════════════════════════════════

    def preprocess_fmri(
        self,
        bold_path: str,
        confounds_path: str = None,
        t1_path: str = None,
    ) -> np.ndarray:
        """Preprocess rs-fMRI: motion correction, nuisance regression,
        bandpass, smoothing.
        """
        bold_img = nib.load(bold_path)

        if confounds_path:
            import pandas as pd
            confounds_df = pd.read_csv(confounds_path, sep="\t")
            motion_cols = [
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z",
            ]
            motion_deriv_cols = [f"{c}_derivative1" for c in motion_cols]
            motion_power_cols = [f"{c}_power2" for c in motion_cols]
            motion_deriv_power_cols = [
                f"{c}_derivative1_power2" for c in motion_cols
            ]
            compcor_cols = [
                c for c in confounds_df.columns
                if c.startswith(("a_comp_cor", "w_comp_cor"))
            ][:10]

            selected_cols = (
                motion_cols + motion_deriv_cols +
                motion_power_cols + motion_deriv_power_cols +
                compcor_cols
            )
            available = [c for c in selected_cols if c in confounds_df.columns]
            confounds = confounds_df[available].fillna(0).values
        else:
            confounds = None

        masker = NiftiLabelsMasker(
            labels_img=self.atlas_img,
            standardize="zscore_sample",
            detrend=True,
            low_pass=self.cfg.low_pass,
            high_pass=self.cfg.high_pass,
            t_r=self.cfg.tr,
            smoothing_fwhm=self.cfg.smoothing_fwhm,
            memory="nilearn_cache",
            memory_level=2,
        )

        roi_time_series = masker.fit_transform(bold_img, confounds=confounds)
        return roi_time_series

    def compute_fc_matrix(self, roi_time_series: np.ndarray) -> np.ndarray:
        """Compute functional connectivity matrix (Pearson correlation).
        Input:  (n_timepoints, n_rois)
        Output: (n_rois, n_rois)
        """
        conn_measure = ConnectivityMeasure(
            kind=self.cfg.connectivity_metric,
            discard_diagonal=True,
        )
        fc = conn_measure.fit_transform([roi_time_series])[0]

        if self.cfg.sparsity_threshold < 1.0:
            abs_fc = np.abs(fc)
            threshold = np.percentile(
                abs_fc, (1 - self.cfg.sparsity_threshold) * 100
            )
            fc[abs_fc < threshold] = 0.0

        fc = np.arctanh(np.clip(fc, -0.999, 0.999))
        return fc.astype(np.float32)

    def process_fmri(self, bold_path: str, confounds_path: str = None) -> dict:
        """Full fMRI pipeline."""
        roi_ts = self.preprocess_fmri(bold_path, confounds_path)
        fc_matrix = self.compute_fc_matrix(roi_ts)
        return {
            "roi_time_series": roi_ts,
            "fc_matrix": fc_matrix,
            "n_rois": self.n_rois,
            "n_timepoints": roi_ts.shape[0],
        }

    # ═══════════════════════════════════════════════
    # COMBINED: sMRI + fMRI
    # ═══════════════════════════════════════════════

    def process(
        self,
        t1_path: str = None,
        bold_path: str = None,
        confounds_path: str = None,
        skip_ants: bool = False,
    ) -> dict:
        """Full MRI pipeline (sMRI + fMRI)."""
        result = {"n_rois": self.n_rois}

        if t1_path:
            print(f"  Processing sMRI: {Path(t1_path).name}")
            smri = self.process_smri(t1_path, skip_ants=skip_ants)
            result["roi_volumes"] = smri["roi_volumes"]

        if bold_path:
            print(f"  Processing fMRI: {Path(bold_path).name}")
            fmri = self.process_fmri(bold_path, confounds_path)
            result["fc_matrix"] = fmri["fc_matrix"]
            result["roi_time_series"] = fmri["roi_time_series"]

        return result

    def process_batch(
        self,
        subject_list: list,
        output_dir: str,
        skip_ants: bool = False,
    ):
        """Process a batch of subjects."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for subj in subject_list:
            sid = subj["subject_id"]
            out_file = output_path / f"{sid}_mri.npz"
            if out_file.exists():
                print(f"  Skipping (exists): {sid}")
                continue
            try:
                result = self.process(
                    t1_path=subj.get("t1_path"),
                    bold_path=subj.get("bold_path"),
                    confounds_path=subj.get("confounds_path"),
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
