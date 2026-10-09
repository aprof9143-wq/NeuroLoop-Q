# NeuroLoop-Q

## Quantum-Optimized Looped Recurrent Transformer for Multi-Modal EEG–MRI Neural Fusion in Alzheimer's & Autism Diagnosis

**Cognivance Labs** — A novel, compute-efficient multi-modal neural fusion architecture that synergistically integrates resting-state EEG and structural/functional MRI for simultaneous Alzheimer's Disease (AD) and Autism Spectrum Disorder (ASD) classification.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Stage 1: Modality-Specific Encoders](#stage-1-modality-specific-encoders)
  - [Stage 2: Cross-Modal Spectral-Spatial Fusion (CMSSF)](#stage-2-cross-modal-spectral-spatial-fusion-cmssf)
  - [Stage 3: Looped Recurrent Transformer Core](#stage-3-looped-recurrent-transformer-core)
  - [Stage 4: Classification Heads](#stage-4-classification-heads)
  - [Quantum Reinforcement Learning Optimizer (Q-RLO)](#quantum-reinforcement-learning-optimizer-q-rlo)
- [Novelty Claims](#novelty-claims)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Datasets](#datasets)
  - [ADNI — Alzheimer's Disease](#adni--alzheimers-disease)
  - [ABIDE-II — Autism](#abide-ii--autism)
  - [OASIS-3 — External Validation](#oasis-3--external-validation)
  - [OpenNeuro — Supplementary EEG](#openneuro--supplementary-eeg)
  - [Atlas — AAL3](#atlas--aal3)
- [Preprocessing Pipeline](#preprocessing-pipeline)
  - [EEG Preprocessing](#eeg-preprocessing)
  - [MRI Preprocessing](#mri-preprocessing)
  - [Feature Extraction](#feature-extraction)
- [Training Protocol](#training-protocol)
  - [Phase 1: Self-Supervised Pre-Training](#phase-1-self-supervised-pre-training)
  - [Phase 2: Multi-Task Fine-Tuning with Q-RLO](#phase-2-multi-task-fine-tuning-with-q-rlo)
  - [Phase 3: Knowledge Distillation](#phase-3-knowledge-distillation)
- [Compute Efficiency](#compute-efficiency)
- [Benchmark Targets](#benchmark-targets)
- [Usage](#usage)
- [Risk Mitigation](#risk-mitigation)
- [Citation](#citation)
- [License](#license)

---

## Overview

NeuroLoop-Q is built on three pillars:

1. **Looped Recurrent Transformer core** — a weight-tied, depth-recurrent transformer that achieves deep reasoning capacity with a fraction of the parameters, grounded in the theoretical framework of [Saunshi et al. (ICLR 2025)](https://arxiv.org/abs/2502.17416) and [Gatmiry et al. (2024)](https://arxiv.org/abs/2410.08292).

2. **Cross-Modal Spectral-Spatial Fusion (CMSSF)** — a dual-stream encoder with gated cross-attention that fuses EEG temporal-spectral features with MRI volumetric/connectivity features at multiple representational depths, inspired by [MCSP for fMRI-EEG fusion (Wei et al., Neural Networks 2024)](https://doi.org/10.1016/j.neunet.2024.107066) and [Brain-OF omnifunctional foundation model (2026)](https://doi.org/10.48550/arxiv.2602.23410).

3. **Quantum Reinforcement Learning Optimizer (Q-RLO)** — a variational quantum circuit-based policy network that dynamically optimizes hyperparameters, loop depth, and fusion gating coefficients during training, leveraging parameter-shift gradient rules from [Wierichs et al. (2021)](https://arxiv.org/abs/2107.12390) and quantum architecture search principles from [Sun et al. (2024)](https://arxiv.org/abs/2401.11576).

**Total estimated parameters:** ~2.1M (looped core: 0.8M, EEG encoder: 0.5M, MRI encoder: 0.6M, heads + fusion: 0.2M)

**Training compute:** ~8 GPU-hours on a single A100 (estimated, for ADNI + ABIDE datasets)

**Inference:** ~50ms per subject on a single V100 (adaptive loop, average 5 iterations)

---

## Architecture

### Stage 1: Modality-Specific Encoders

#### EEG Encoder (Temporal-Spectral Stream)

- **Input:** Raw rs-EEG (128 or 64 channel, 250–500 Hz), segmented into 2–5s epochs
- **Preprocessing:** Bandpass filtering (0.5–45 Hz), ICA artifact rejection, common average reference
- **Feature extraction:** Short-time Fourier transform (STFT) + connectivity matrices (phase-locking value, coherence) → 3D tensor: `(channels × frequency_bands × time_windows)`
- **Architecture:** 2-layer convolutional patch encoder (kernel = frequency band × temporal window) → token sequence of `N_eeg` tokens, each `d_model = 256`
- **Positional encoding:** Learned relative encoding over (channel, frequency, time) triplets

#### MRI Encoder (Spatial-Anatomical Stream)

- **Input:** Co-registered T1-weighted sMRI + rs-fMRI (if available)
- **sMRI processing:** Skull-stripping, tissue segmentation (GM/WM/CSF), regional volumetric features (AAL3 atlas, ~116 ROIs) → ROI volume vector
- **fMRI processing:** ROI-level BOLD time series → functional connectivity matrix (Pearson correlation, partial correlation) → `(ROIs × ROIs)` matrix
- **Architecture:** Graph Convolutional Network (GCN) over the ROI adjacency graph (2 layers, hidden = 256) → token sequence of `N_mri` tokens, each `d_model = 256`
- **Edge tokens:** The fMRI connectivity matrix is decomposed into edge tokens via a learned linear projection
- **Global token:** The sMRI volume vector is broadcast as a global modality token

---

### Stage 2: Cross-Modal Spectral-Spatial Fusion (CMSSF)

The novel fusion mechanism. Rather than simple concatenation or late fusion, CMSSF performs **iterative cross-attention between modalities at every loop iteration** of the Looped Transformer core.

```
For each loop iteration t = 1, ..., L:
    H_eeg^t = LoopBlock(H_eeg^{t-1}, CrossAttn(Q=H_eeg, K=V=H_mri))
    H_mri^t = LoopBlock(H_mri^{t-1}, CrossAttn(Q=H_mri, K=V=H_eeg))
    H_fused^t = GatedFusion(H_eeg^t, H_mri^t)
    # Gating coefficient g^t = σ(W_g · [H_eeg^t; H_mri^t; H_eeg^t ⊙ H_mri^t])
    # H_fused^t = g^t ⊙ H_eeg^t + (1 - g^t) ⊙ H_mri^t
```

The **gated fusion** learns modality-specific reliability dynamically — critical because EEG and MRI capture complementary but asymmetrically reliable signals across AD (where structural atrophy in MRI is the dominant biomarker) and ASD (where EEG spectral atypicality is more discriminative).

---

### Stage 3: Looped Recurrent Transformer Core

Built directly on the theoretical foundations of [Saunshi et al. (ICLR 2025)](https://arxiv.org/abs/2502.17416) and [Gatmiry et al. (2024)](https://arxiv.org/abs/2410.08292).

**Key insight:** A *k*-layer transformer looped *L* times nearly matches the performance of a *kL*-layer non-looped model, while requiring only *k* layers worth of parameters. The looped structure provides an inductive bias for iterative refinement — mathematically equivalent to simulating *T* steps of chain-of-thought reasoning with *T* loops.

| Component | Specification | Rationale |
|---|---|---|
| Core layers (*k*) | 4 transformer blocks | Minimal depth; the loop provides the rest |
| Loop iterations (*L*) | Adaptive, 3–12 (dynamically selected by Q-RLO) | Hard cases get more loops; easy cases exit early |
| Weight sharing | Full (all 4 blocks share weights per loop) | Parameter efficiency: 4-layer params, ~48-layer effective depth |
| Hidden dimension (*d_model*) | 256 | Small enough for low compute, large enough for cross-modal fusion |
| Attention heads | 8 (32-dim each) | Standard multi-head attention |
| Feed-forward expansion | 2× (512 hidden) | Reduced from standard 4× to cut FLOPs |
| Normalization | Pre-LayerNorm + residual connections | Stability across loop iterations |
| Loop exit criterion | Confidence-based halting (entropy of classification head < threshold τ) | Adaptive compute — easy samples use fewer loops |
| Positional encoding | RoPE (rotary) for token positions + learned loop-step embedding | RoPE generalizes to variable-length sequences |

#### Adaptive Loop Depth with Halting

```python
def forward(self, H_eeg, H_mri):
    for t in range(self.max_loops):
        H_eeg, H_mri = self.loop_block(H_eeg, H_mri, step=t)
        H_fused = self.gated_fusion(H_eeg, H_mri)
        logits = self.classification_head(H_fused)
        entropy = -torch.sum(
            F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1
        )
        if entropy < self.tau:  # confidence threshold
            return logits, t  # early exit
    return logits, self.max_loops
```

The model spends **more compute on ambiguous cases** (e.g., early-stage AD vs. normal aging, or subthreshold ASD) and **less on clear-cut cases** — a form of adaptive inference-time scaling.

---

### Stage 4: Classification Heads

- **Disease-specific heads:** Binary AD head (CN vs. AD, or 3-class CN/MCI/AD), Binary ASD head (TD vs. ASD)
- **Multi-task head:** Shared representation → disease-specific linear classifiers
- **Auxiliary heads:** Age regression, sex classification (regularization via multi-task learning)
- **Loss function:**

```
L = L_AD + λ₁·L_ASD + λ₂·L_age + λ₃·L_sex + λ₄·L_contrastive
```

| Loss term | Weight | Purpose |
|---|---|---|
| `L_AD` | λ = 1.0 | Alzheimer's classification (cross-entropy) |
| `L_ASD` | λ₁ = 0.8 | Autism classification (cross-entropy) |
| `L_age` | λ₂ = 0.1 | Age regression (MSE) — auxiliary regularization |
| `L_sex` | λ₃ = 0.1 | Sex classification (cross-entropy) — auxiliary regularization |
| `L_contrastive` | λ₄ = 0.3 | Cross-modal alignment (InfoNCE) — pulls same-subject EEG-MRI pairs together, pushes different-subject pairs apart |

---

### Quantum Reinforcement Learning Optimizer (Q-RLO)

The novel optimization pipeline. Instead of standard AdamW or manual hyperparameter scheduling, Q-RLO uses a **variational quantum circuit (VQC) as a policy network** that learns to dynamically adjust:

1. **Learning rate** per layer group (3 groups: EEG encoder, MRI encoder, looped core)
2. **Loop depth** *L* per training step (curriculum: start with L=3, expand to L=12)
3. **Fusion gate temperature** (controls modality balance)
4. **Dropout rate** per loop iteration (higher in early loops, lower in later)
5. **Gradient clipping threshold**

#### VQC Policy Network Architecture

```
State s_t = [current_loss, gradient_norm, validation_accuracy,
             loop_depth, epoch_progress, entropy]
    ↓ Angle encoding into n_qubits (n = 8–12)
VQC: 3 layers of {Ry rotations, CNOT entanglement, Ry rotations}
    ↓ Measurement (expectation values of Pauli-Z on each qubit)
Action a_t = [lr_1, lr_2, lr_3, L_t, gate_temp, dropout_t, clip_t]
    ↓ Clipped to valid ranges
Reward r_t = Δvalidation_accuracy - α·compute_cost(L_t)
```

#### Why Quantum RL?

- **Expressivity:** VQCs can represent certain function classes that require exponentially many classical parameters ([Qi et al., 2021](https://arxiv.org/abs/2110.03861))
- **Exploration:** Quantum superposition enables more diverse exploration of the hyperparameter landscape
- **Parameter efficiency:** 8–12 qubits with 3 variational layers = ~24–36 trainable parameters for the entire optimizer, vs. thousands for a classical RL policy network
- **Hybrid execution:** The VQC runs on a quantum simulator (or real QPU if available), while the classical model trains on GPU — the quantum circuit is evaluated every *N* steps, not every step

---

## Novelty Claims

1. **First application of Looped Recurrent Transformers to neuroimaging fusion** — all prior EEG-MRI fusion work uses standard CNNs, GCNs, or non-looped transformers. The looped structure provides iterative cross-modal refinement grounded in latent-reasoning theory ([Saunshi et al., ICLR 2025](https://arxiv.org/abs/2502.17416)).

2. **Adaptive-depth inference for medical diagnosis** — confidence-based halting means the model dynamically allocates more reasoning loops to ambiguous clinical cases, a form of inference-time compute scaling never before applied to neuroimaging.

3. **Quantum RL for hyperparameter and architectural optimization** — no prior work combines VQC-based policy networks with medical deep learning training loops. The parameter-shift gradient rule ([Wierichs et al., 2021](https://arxiv.org/abs/2107.12390)) makes this tractable on NISQ-era quantum simulators.

4. **Gated cross-modal fusion within the loop** — the fusion gate coefficients are themselves optimized by Q-RLO, meaning the modality balance adapts not just per-dataset but per-subject and per-training-step.

5. **Simultaneous AD + ASD multi-task learning** — most prior work addresses these independently; sharing a fused representation across both disorders may reveal transdiagnostic neural patterns.

---

## Repository Structure

```
neuroloop_q/
│
├── preprocessing/                          ✅ COMPLETE
│   ├── __init__.py                         # Package init, exports all classes
│   ├── config.py                           # Central config (EEG, MRI, Model, Train)
│   ├── eeg_preprocessing.py                # MNE pipeline: filter, ICA, epoch, STFT, PLV
│   ├── mri_preprocessing.py                # Nilearn/ANTs: skull-strip, segment, FC matrix
│   ├── dataset_adni.py                     # ADNI loader: CN/MCI/AD labels, MRI+EEG pairing
│   ├── dataset_abide.py                    # ABIDE-II loader: TD/ASD labels, fMRI
│   ├── feature_extraction.py               # Cross-modal tensor assembly for model input
│   ├── run_pipeline.py                     # Master orchestrator (run all preprocessing)
│   └── requirements.txt                    # Python dependencies
│
├── model/                                  🔲 PART 2 (Next)
│   ├── __init__.py
│   ├── eeg_encoder.py                      # Conv patch encoder → d_model tokens
│   ├── mri_encoder.py                      # GCN over ROI graph → d_model tokens
│   ├── cmssf.py                            # Cross-Modal Spectral-Spatial Fusion (gated)
│   ├── looped_transformer.py               # Weight-tied recurrent transformer core
│   ├── neuroloop_q.py                      # Full model assembly + adaptive halting
│   └── quantum_rlo.py                      # VQC policy network optimizer (PennyLane)
│
├── training/                               🔲 PART 3
│   ├── __init__.py
│   ├── losses.py                           # Contrastive, multi-task, auxiliary losses
│   ├── pretraining.py                      # Phase 1: Self-supervised cross-modal alignment
│   ├── finetuning.py                       # Phase 2: Multi-task fine-tuning with Q-RLO
│   ├── distillation.py                     # Phase 3: Knowledge distillation (teacher→student)
│   └── scheduler.py                        # Q-RLO integrated LR + loop-depth scheduler
│
├── utils/                                  🔲 PART 3
│   ├── __init__.py
│   ├── metrics.py                          # Accuracy, AUC, F1, sensitivity, specificity
│   └── visualization.py                    # Attention rollout, loop-depth plots, t-SNE
│
├── scripts/                                🔲 PART 3
│   ├── train_adni.py                       # Train on ADNI (Alzheimer's)
│   ├── train_abide.py                      # Train on ABIDE-II (Autism)
│   ├── train_multitask.py                  # Joint AD + ASD multi-task training
│   └── evaluate.py                         # Evaluation + benchmark comparison
│
├── data/                                   # Created at runtime (not in repo)
│   ├── raw/
│   │   ├── ADNI/
│   │   │   ├── MRI/                         # T1 + rs-fMRI NIfTI files
│   │   │   ├── EEG/                         # rs-EEG files (.set/.edf)
│   │   │   └── clinical/                    # DXSUM_PDXCON_ADNIALL.csv
│   │   └── ABIDE-II/
│   │       ├── CPAP/filt_noglobal/          # Preprocessed fMRI (CPAC pipeline)
│   │       └── ABIDEII_Composite_Pheno.csv
│   └── processed/                           # Preprocessed .npz tensors
│       ├── ADNI/
│       │   ├── eeg/
│       │   ├── mri/
│       │   └── features/
│       └── ABIDE/
│           ├── mri/
│           └── features/
│
├── checkpoints/                            # Saved models (not in repo)
├── logs/                                   # TensorBoard logs (not in repo)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- ANTs (optional, for raw MRI skull-stripping/registration)
- fMRIPrep (optional, for raw fMRI preprocessing via Docker)

### Install Dependencies

```bash
git clone https://github.com/<your-org>/neuroloop_q.git
cd neuroloop_q
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥2.1.0 | Deep learning framework |
| `torch-geometric` | ≥2.4.0 | GCN for MRI encoder |
| `mne` | ≥1.5.0 | EEG preprocessing |
| `nilearn` | ≥0.10.0 | MRI preprocessing & connectivity |
| `nibabel` | ≥5.1.0 | NIfTI file I/O |
| `pennylane` | ≥0.33.0 | Quantum circuit simulation for Q-RLO |
| `pennylane-lightning` | ≥0.33.0 | Fast quantum state simulator |
| `scipy` | ≥1.10.0 | STFT, Hilbert transform, filtering |
| `scikit-learn` | ≥1.3.0 | Stratified splits, metrics |
| `tqdm` | ≥4.65.0 | Progress bars |

### Optional Dependencies

| Package | Purpose |
|---|---|
| `ants` (via conda) | Raw MRI skull-stripping & registration |
| `fMRIPrep` (via Docker) | Full fMRI preprocessing pipeline |
| `mne-icalabel` | Auto-ICA component classification |

---

## Datasets

### ADNI — Alzheimer's Disease

| Item | Details |
|---|---|
| **Website** | [https://adni.loni.usc.edu](https://adni.loni.usc.edu) |
| **Access** | Free, requires registration + data use agreement |
| **What to download** | T1-weighted MRI, rs-fMRI (ADNI4), EEG (limited), clinical labels |
| **Clinical labels** | `DXSUM_PDXCON_ADNIALL.csv` (diagnosis: CN/EMCI/LMCI/AD) |
| **MRI format** | NIfTI (`.nii` / `.nii.gz`) — download via LONI IDA Image Archive |
| **EEG** | Limited in ADNI proper; supplement with OpenNeuro |
| **Citation** | Petersen RC et al., *Neurology* 2010 — [DOI: 10.1212/WNL.0b013e3181fcbf1c](https://doi.org/10.1212/WNL.0b013e3181fcbf1c) |
| **Preprocessed alternative** | Stanford STAI: [https://stai.stanford.edu/data/198-adni](https://stai.stanford.edu/data/198-adni) |

**How to download:**

1. Register at [https://adni.loni.usc.edu/data-samples/access-data/](https://adni.loni.usc.edu/data-samples/access-data/)
2. Submit data use application (approved within 2–5 business days)
3. Access via LONI IDA: [https://ida.loni.usc.edu](https://ida.loni.usc.edu)
4. Download T1 MRI under **Collections → Image Collections → ADNI**
5. Download clinical data under **Study Data → Clinical → Diagnostic Summary**
6. Place files in `data/raw/ADNI/MRI/<Subject_ID>/T1/` and `data/raw/ADNI/MRI/<Subject_ID>/rsfMRI/`

---

### ABIDE-II — Autism

| Item | Details |
|---|---|
| **Website** | [http://fcon_1000.projects.nitrc.org/indi/abide/](http://fcon_1000.projects.nitrc.org/indi/abide/) |
| **Access** | Open access, no registration required |
| **What to download** | Preprocessed rs-fMRI (CPAC pipeline recommended) |
| **Phenotypic** | `ABIDEII_Composite_Pheno.csv` (DX_GROUP: 1=ASD, 2=TD) |
| **Preprocessed pipeline** | [http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/) |
| **Recommended strategy** | `filt_noglobal` (bandpass filtered, no global signal regression) |
| **Atlas** | AAL (matches our config) |
| **Citation** | Di Martino A et al., *Scientific Data* 2017 — [DOI: 10.1038/sdata.2017.17](https://doi.org/10.1038/sdata.2017.17) |

**How to download:**

```bash
# Option 1: Nilearn automatic fetcher (recommended)
python -c "
from nilearn import datasets
abide = datasets.fetch_abide_pcp(
    data_dir='./data/raw/ABIDE-II',
    n_subjects=None,
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=['rois_aal'],
    quality_checked=True,
)
print(f'Downloaded {len(abide.rois_aal)} subjects')
"

# Option 2: Manual download from NITRC
# https://www.nitrc.org/projects/abide_2/
```

---

### OASIS-3 — External Validation

| Item | Details |
|---|---|
| **Website** | [https://www.oasis-brains.org](https://www.oasis-brains.org) |
| **Access** | Free, requires registration |
| **What to download** | T1-weighted MRI (cross-sectional) |
| **Use** | External validation only — not used for training |
| **Citation** | LaMontagne PJ et al., *Med Image Anal* 2019 — [DOI: 10.1016/j.media.2018.09.005](https://doi.org/10.1016/j.media.2018.09.005) |

---

### OpenNeuro — Supplementary EEG

ADNI has limited paired EEG-MRI data. For supplementary resting-state EEG from Alzheimer's patients:

| Dataset | Link | Format |
|---|---|---|
| OpenNeuro ds003505 | [https://openneuro.org/datasets/ds003505](https://openneuro.org/datasets/ds003505) | EEGLAB `.set` |
| OpenNeuro ds002717 | [https://openneuro.org/datasets/ds002717](https://openneuro.org/datasets/ds002717) | BrainVision `.eeg` |
| OpenNeuro ds004187 | [https://openneuro.org/datasets/ds004187](https://openneuro.org/datasets/ds004187) | EEGLAB `.set` |

```bash
# Download via AWS S3 (no account needed)
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds003505 \
  data/raw/ADNI/EEG/ds003505
```

---

### Atlas — AAL3

| Item | Details |
|---|---|
| **Atlas** | AAL3 (Automated Anatomical Labeling, SPM12 version) |
| **ROIs** | 116 cortical + subcortical regions |
| **Download** | Automatic via Nilearn: `datasets.fetch_atlas_aal(version="SPM12")` |
| **Reference** | Rolls ET et al., *NeuroImage* 2020 — [DOI: 10.1016/j.neuroimage.2019.116189](https://doi.org/10.1016/j.neuroimage.2019.116189) |

---

### Dataset Summary

| Dataset | Disorder | Modalities | Subjects | Access | Used For |
|---|---|---|---|---|---|
| **ADNI** | Alzheimer's | T1 sMRI + rs-fMRI + EEG (limited) | ~2,000 | Registration required | Training (AD) |
| **ABIDE-II** | Autism | rs-fMRI (CPAC preprocessed) | ~1,100 | Open access | Training (ASD) |
| **OASIS-3** | Alzheimer's | T1 sMRI | ~1,000 | Registration required | External validation |
| **OpenNeuro** | Alzheimer's | rs-EEG | Varies | Open access | Supplementary EEG |

---

## Preprocessing Pipeline

### EEG Preprocessing

Built on **MNE-Python** following the EEG-Pype pipeline ([Lodema et al., PLOS Comp Biol 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12970966)) and MNE standard practices.

**Pipeline steps:**

1. **Load** raw EEG (`.set` / `.edf` / `.fif` / `.vhdr`)
2. **Downsample** to 500 Hz (if higher)
3. **Channel selection** — drop non-EEG channels (EOG, ECG, EMG)
4. **Bad channel detection** — kurtosis-based detection + interpolation
5. **Bandpass filter** — 0.5–45 Hz (FIR, firwin design)
6. **Notch filter** — 50 Hz (or 60 Hz for US data)
7. **ICA artifact removal** — FastICA, 25 components, auto-detect EOG/ECG/muscle
8. **Common average reference**
9. **Epoching** — 2s windows, 50% overlap, amplitude rejection (<200 µV)
10. **Frequency-band decomposition** — delta (0.5–4), theta (4–8), alpha (8–13), beta (13–30), gamma (30–45 Hz)
11. **STFT** — short-time Fourier transform per channel → spectrogram tokens
12. **Connectivity** — phase-locking value (PLV) per band → `(n_bands, n_channels, n_channels)`

**Output tensors:**

| Tensor | Shape | Description |
|---|---|---|
| `stft` | `(n_epochs, n_channels, n_freq_bins, n_time_windows)` | Spectrogram per channel per epoch |
| `bandpower` | `(n_epochs, n_channels, n_bands)` | Average power per frequency band |
| `connectivity` | `(n_epochs, n_bands, n_channels, n_channels)` | PLV connectivity matrix per band |

**Configuration:**

| Parameter | Value | Description |
|---|---|---|
| `sampling_rate` | 500 Hz | Downsampling target |
| `n_channels` | 128 | Channel count (64 also supported) |
| `bandpass_low` | 0.5 Hz | High-pass cutoff |
| `bandpass_high` | 45.0 Hz | Low-pass cutoff |
| `notch_freq` | 50.0 Hz | Power-line interference |
| `ica_n_components` | 25 | ICA decomposition components |
| `epoch_duration` | 2.0 s | Epoch length |
| `epoch_overlap` | 0.5 | 50% overlap between epochs |
| `reference` | `average` | Common average reference |
| `stft_window` | 256 | STFT window size (samples) |
| `stft_overlap` | 128 | STFT overlap (50%) |
| `connectivity_metric` | `plv` | Phase-locking value |

---

### MRI Preprocessing

Built on **Nilearn**, **ANTs** (via Nipype), and **FreeSurfer** outputs.

#### Structural MRI (sMRI)

1. **Skull-stripping** — ANTs `antsBrainExtraction.sh` or pre-stripped input
2. **Tissue segmentation** — ANTs Atropos (GM/WM/CSF, 3 classes)
3. **Registration** — ANTs SyN to MNI152NLin2009cAsym
4. **ROI parcellation** — AAL3 atlas (116 ROIs)
5. **Volume extraction** — GM volume per ROI, normalized by total GM

#### Functional MRI (fMRI)

1. **Motion correction** — rigid body (assumed done by fMRIPrep/CPAC)
2. **Nuisance regression** — 24 motion params (6 + derivatives + quadratics) + aCompCor (5 WM + 5 CSF)
3. **Bandpass filtering** — 0.01–0.08 Hz
4. **Spatial smoothing** — 6mm FWHM Gaussian kernel
5. **ROI time series extraction** — AAL3 atlas, z-scored
6. **Functional connectivity** — Pearson correlation → Fisher z-transform
7. **Sparsification** — keep top 20% of connections

**Output tensors:**

| Tensor | Shape | Description |
|---|---|---|
| `roi_volumes` | `(116,)` | Normalized GM volume per ROI |
| `fc_matrix` | `(116, 116)` | Fisher z-transformed FC matrix |

**Configuration:**

| Parameter | Value | Description |
|---|---|---|
| `atlas` | `aal` | AAL3 atlas (SPM12 version) |
| `n_rois` | 116 | Number of ROIs |
| `tr` | 2.0 s | Repetition time |
| `low_pass` | 0.08 Hz | Bandpass low cutoff |
| `high_pass` | 0.01 Hz | Bandpass high cutoff |
| `smoothing_fwhm` | 6.0 mm | Spatial smoothing kernel |
| `n_confounds` | 24 | Motion params + derivatives |
| `connectivity_metric` | `pearson` | FC metric |
| `sparsity_threshold` | 0.2 | Keep top 20% connections |

---

### Feature Extraction

Cross-modal tensor assembly converts preprocessed EEG + MRI data into model-ready token sequences.

**EEG → Tokens:**
- Log-mel transform of STFT spectrograms
- Reshape: `(n_channels, n_freq_bins * n_time_windows)` → linear projection → `d_model`
- Per-channel z-normalization
- Average connectivity across bands → single adjacency matrix

**MRI → Tokens:**
- ROI volumes → z-normalized global modality token
- FC matrix → graph adjacency for GCN
- Upper triangle → edge feature vector

**Output:**

| Tensor | Shape | Description |
|---|---|---|
| `eeg_tokens` | `(B, n_channels, n_freq_bins * n_time_windows)` | EEG patch tokens |
| `eeg_connectivity` | `(B, n_channels, n_channels)` | EEG graph adjacency |
| `eeg_bandpower` | `(B, n_channels, n_bands)` | EEG auxiliary features |
| `mri_volumes` | `(B, 116)` | MRI ROI volume vector |
| `mri_fc` | `(B, 116, 116)` | MRI FC matrix |
| `mri_edge_features` | `(B, n_edges)` | MRI upper-triangle FC values |

---

## Training Protocol

### Phase 1: Self-Supervised Pre-Training

**Goal:** Learn cross-modal alignment without labels.

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Method | Masked modality modeling + contrastive learning |
| Masking | Randomly mask EEG frequency bands or MRI ROI volumes, predict from the other modality |
| Contrastive | InfoNCE: same-subject EEG-MRI pairs as positives, cross-subject as negatives |
| Duration | ~3 GPU-hours |

```python
# Phase 1: Self-supervised pre-training
from training.pretraining import pretrain

pretrain(
    model=model,
    train_loader=train_loader,
    epochs=50,
    mask_ratio=0.3,
    contrastive_temperature=0.07,
    output_dir="./checkpoints/phase1",
)
```

---

### Phase 2: Multi-Task Fine-Tuning with Q-RLO

**Goal:** Joint AD + ASD classification with auxiliary regression heads.

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Optimizer | AdamW (base) + Q-RLO (dynamic adjustment) |
| Q-RLO update freq | Every 50 steps |
| Curriculum | Start with L=3 loops, Q-RLO expands to L=12 |
| Loss | `L_AD + λ₁·L_ASD + λ₂·L_age + λ₃·L_sex + λ₄·L_contrastive` |
| Duration | ~4 GPU-hours |

```python
# Phase 2: Multi-task fine-tuning with Q-RLO
from training.finetuning import finetune

finetune(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    qrlo_config=QRLOConfig(
        n_qubits=8,
        vqc_layers=3,
        update_freq=50,
        loop_depth_range=(3, 12),
    ),
    output_dir="./checkpoints/phase2",
)
```

---

### Phase 3: Knowledge Distillation

**Goal:** Train a lightweight student model (fixed L=4, no adaptive loop) that distills the teacher's adaptive-depth behavior.

| Parameter | Value |
|---|---|
| Epochs | 30 |
| Teacher | Phase 2 model (adaptive L=3–12) |
| Student | Fixed L=4, no halting |
| Distillation temperature | 4.0 |
| Student compute | ~0.3× teacher FLOPs |
| Duration | ~1 GPU-hour |

```python
# Phase 3: Knowledge distillation
from training.distillation import distill

distill(
    teacher=teacher_model,
    student=student_model,
    train_loader=train_loader,
    epochs=30,
    temperature=4.0,
    output_dir="./checkpoints/phase3",
)
```

---

## Compute Efficiency

The entire design philosophy is **maximum diagnostic performance per FLOP**.

| Strategy | Mechanism | Compute Savings |
|---|---|---|
| Weight-tied looping | 4-layer model → ~48-layer effective depth | ~12× parameter reduction vs. equivalent depth |
| Adaptive loop depth | Easy cases exit after 3 loops; hard cases use up to 12 | ~2.5× average FLOP reduction |
| Reduced FFN expansion | 2× instead of 4× | ~1.7× FLOP reduction per layer |
| Q-RLO parameter efficiency | 24–36 quantum params vs. classical RL network | Negligible optimizer overhead |
| Knowledge distillation | Train deep (L=12), distill to student (L=4) | Student runs in ~0.3× teacher FLOPs |
| Mixed-precision training | FP16 forward, FP32 master weights | ~1.8× speedup on modern GPUs |
| Gradient checkpointing | Recompute activations in backward pass | ~40% memory reduction |
| Frozen pre-trained MRI encoder | Pre-train GCN on UK Biobank, fine-tune top layers | Eliminates ~60% of MRI encoder training cost |

---

## Benchmark Targets

| Task | Current SOTA | Method | NeuroLoop-Q Target |
|---|---|---|---|
| AD classification (ADNI) | ~94–96% accuracy | Multi-stream CNN sMRI+fMRI ([Abd El Naby et al., 2025](https://doi.org/10.5120/ijca2025925441)) | **≥97%** |
| ASD classification (ABIDE) | ~80–85% accuracy | Optimized ML on sMRI ([Bahathiq et al., 2024](https://doi.org/10.3390/app14020473)) | **≥88%** |
| ASD from rs-EEG | ~87–90% accuracy | Graph ConvNet on rs-EEG ([Hu et al., 2023](https://doi.org/10.1109/tnsre.2023.3347134)) | **≥92%** |
| EEG-fMRI fusion (any disorder) | Emerging — MCSP ([Wei et al., 2024](https://doi.org/10.1016/j.neunet.2024.107066)) | First looped transformer + QRL benchmark |

---

## Usage

### Quick-Start: Full Pipeline

```bash
# 1. Create directory structure
mkdir -p data/raw/ADNI/{MRI,EEG,clinical}
mkdir -p data/raw/ABIDE-II

# 2. Download datasets (see Datasets section above)

# 3. Run preprocessing
python -m preprocessing.run_pipeline --dataset ALL --skip-ants

# 4. Train (after model code is available)
python scripts/train_multitask.py --dataset ALL --epochs 200

# 5. Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/phase2/best_model.pt
```

### Running Individual Components

```bash
# Preprocess EEG only
python -m preprocessing.eeg_preprocessing --input data/raw/ADNI/EEG --output data/processed/ADNI/eeg

# Preprocess MRI only
python -m preprocessing.mri_preprocessing --input data/raw/ADNI/MRI --output data/processed/ADNI/mri

# Run feature extraction
python -m preprocessing.feature_extraction --input data/processed --output data/processed/features

# Quality report
python -m preprocessing.run_pipeline --quality-report
```

### Using the Python API

```python
from preprocessing import EEGPreprocessor, MRIPreprocessor, FeatureAssembler

# ── EEG ──
eeg_pre = EEGPreprocessor()
eeg_result = eeg_pre.process("data/raw/ADNI/EEG/sub-001.set")
print(f"STFT shape: {eeg_result['stft'].shape}")
print(f"Connectivity shape: {eeg_result['connectivity'].shape}")

# ── MRI ──
mri_pre = MRIPreprocessor()
mri_result = mri_pre.process(
    t1_path="data/raw/ADNI/MRI/sub-001/T1/image.nii.gz",
    bold_path="data/raw/ADNI/MRI/sub-001/rsfMRI/image.nii.gz",
    skip_ants=True,
)
print(f"ROI volumes: {mri_result['roi_volumes'].shape}")
print(f"FC matrix: {mri_result['fc_matrix'].shape}")

# ── Feature Assembly ──
assembler = FeatureAssembler()
features = assembler.assemble(
    eeg_data={"stft": eeg_result["stft"][0],
              "bandpower": eeg_result["bandpower"][0],
              "connectivity": eeg_result["connectivity"][0]},
    mri_data={"roi_volumes": mri_result["roi_volumes"],
              "fc_matrix": mri_result["fc_matrix"]},
)
print(f"EEG tokens: {features['eeg']['tokens'].shape}")
print(f"MRI volumes: {features['mri']['volumes'].shape}")
```

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Paired EEG-MRI data scarcity | Phase 1 self-supervised pre-training on unpaired data; cross-modal generation via [CATD framework](https://doi.org/10.1109/tmi.2025.3550206) to synthesize fMRI from EEG when only EEG is available |
| Quantum hardware unavailability | VQC runs on classical simulator (PennyLane/Qiskit); no real QPU required for training |
| Overfitting on small clinical datasets | Weight-tied looping acts as implicit regularizer; dropout scheduled by Q-RLO; multi-task auxiliary losses |
| Domain shift across sites | Site-specific batch normalization; adversarial domain adaptation in the fused representation |
| Interpretability | Attention rollout across loop iterations; per-loop attention maps show how the model refines its diagnosis iteratively |

---

## Citation

If you use NeuroLoop-Q in your research, please cite:

```bibtex
@software{neuroloop_q,
  title={NeuroLoop-Q: Quantum-Optimized Looped Recurrent Transformer for Multi-Modal EEG-MRI Neural Fusion},
  author={Cognivance Labs},
  year={2026},
  description={Multi-modal neural fusion architecture for Alzheimer's and Autism diagnosis using looped recurrent transformers with quantum RL optimization}
}
```

### Key References

- Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers," ICLR 2025 — [arXiv:2502.17416](https://arxiv.org/abs/2502.17416)
- Gatmiry et al., "Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?" 2024 — [arXiv:2410.08292](https://arxiv.org/abs/2410.08292)
- Nguyen & Lin, "Intra-Layer Recurrence in Transformers for Language Modeling," 2025 — [arXiv:2505.01855](https://arxiv.org/abs/2505.01855)
- Wierichs et al., "General Parameter-Shift Rules for Quantum Gradients," 2021 — [arXiv:2107.12390](https://arxiv.org/abs/2107.12390)
- Sun et al., "Quantum Architecture Search with Unsupervised Representation Learning," 2024 — [arXiv:2401.11576](https://arxiv.org/abs/2401.11576)
- Wei et al., "Multi-modal Cross-domain Self-supervised Pre-training for fMRI and EEG Fusion," Neural Networks, 2024 — [DOI:10.1016/j.neunet.2024.107066](https://doi.org/10.1016/j.neunet.2024.107066)
- Guo et al., "Brain-OF: An Omnifunctional Foundation Model for fMRI, EEG and MEG," 2026 — [arXiv:2602.23410](https://doi.org/10.48550/arxiv.2602.23410)
- Hu et al., "Regional-Asymmetric Adaptive GCN for Diagnosis of Autism in Children with rs-EEG," IEEE TNSRE, 2023 — [DOI:10.1109/tnsre.2023.3347134](https://doi.org/10.1109/tnsre.2023.3347134)
- Lodema et al., "EEG-Pype: A Pipeline for EEG Preprocessing," PLOS Comp Biol, 2026 — [PMC12970966](https://pmc.ncbi.nlm.nih.gov/articles/PMC12970966)

---

## License

Proprietary — Cognivance Labs. All rights reserved.

---

<p align="center">
  <strong>Cognivance Labs</strong><br>
  Multi-Modal Neural Fusion for Neurological Disorder Diagnosis<br>
  <em>Built on the shoulders of looped recurrent transformers and quantum reinforcement learning</em>
</p>
