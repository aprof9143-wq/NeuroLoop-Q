"""
NeuroLoop-Q EEG Preprocessing Pipeline
Built on MNE-Python — follows EEG-Pype best practices (Lodema et al., 2026)

Input:  Raw rs-EEG files (.set/.edf/.fif) from ADNI or ABIDE-II
Output: Preprocessed tensor: (n_epochs, n_channels, n_bands, n_time_windows)
        Connectivity tensor: (n_epochs, n_bands, n_channels, n_channels)
"""
import mne
import numpy as np
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .config import EEGConfig

class EEGPreprocessor:
    """Full resting-state EEG preprocessing pipeline."""

    def __init__(self, config: EEGConfig = None):
        self.cfg = config or EEGConfig()
        self.sfreq = self.cfg.sampling_rate
        self.bands = self.cfg.freq_bands

    # ─── Step 1: Load & Re-reference ───
    def load_raw(self, filepath: str) -> mne.io.Raw:
        """Load raw EEG and apply common average reference."""
        ext = Path(filepath).suffix.lower()
        if ext in (".set",):
            raw = mne.io.read_raw_eeglab(filepath, preload=True)
        elif ext in (".edf", ".bdf"):
            raw = mne.io.read_raw_edf(filepath, preload=True)
        elif ext in (".fif"):
            raw = mne.io.read_raw_fif(filepath, preload=True)
        elif ext in (".vhdr",):
            raw = mne.io.read_raw_brainvision(filepath, preload=True)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Downsample if needed
        if raw.info["sfreq"] > self.sfreq:
            raw = raw.resample(self.sfreq, npad="auto")

        # Set common average reference
        raw.set_eeg_reference(self.cfg.reference, projection=False)

        # Drop non-EEG channels (EOG, ECG, EMG) if present
        eeg_channels = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw.pick(eeg_channels)

        print(f"  Loaded: {raw.info['nchan']} channels, "
              f"{raw.info['sfreq']} Hz, {raw.times[-1]:.1f}s")
        return raw

    # ─── Step 2: Filtering ───
    def filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Bandpass + notch filter."""
        raw = raw.filter(
            l_freq=self.cfg.bandpass_low,
            h_freq=self.cfg.bandpass_high,
            method="fir",
            fir_design="firwin",
            phase="zero-double",
        )
        if self.cfg.notch_freq > 0:
            raw = raw.notch_filter(
                freqs=[self.cfg.notch_freq],
                method="fir",
                fir_design="firwin",
            )
        return raw

    # ─── Step 3: ICA Artifact Rejection ───
    def apply_ica(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Fit ICA and remove artifact components (eye, muscle, heartbeat)."""
        ica = mne.preprocessing.ICA(
            n_components=self.cfg.ica_n_components,
            method="fastica",
            max_iter="auto",
            random_state=42,
            fit_params=dict(extended=True),
        )

        raw_for_ica = raw.copy().filter(1.0, 47.0, method="fir")
        ica.fit(raw_for_ica)

        # Auto-detect artifact components
        try:
            eog_comps = ica.find_bads_eog(raw_for_ica, threshold=3.0)[0]
        except Exception:
            eog_comps = []

        try:
            ecg_comps = ica.find_bads_ecg(raw_for_ica, threshold=3.0)[0]
        except Exception:
            ecg_comps = []

        muscle_comps = self._detect_muscle_components(ica, raw_for_ica)

        exclude = list(set(eog_comps + ecg_comps + muscle_comps))
        ica.exclude = exclude
        print(f"  ICA: removing {len(exclude)} artifact components "
              f"(EOG: {len(eog_comps)}, ECG: {len(ecg_comps)}, "
              f"Muscle: {len(muscle_comps)})")

        raw = ica.apply(raw.copy())
        return raw

    def _detect_muscle_components(self, ica, raw):
        """Heuristic: components with high power > 30 Hz are likely muscle."""
        comps = ica.get_sources(raw)
        data = comps.get_data()
        sfreq = raw.info["sfreq"]
        muscle_idx = []
        for i in range(data.shape[0]):
            freqs, psd = signal.welch(data[i], sfreq, nperseg=512)
            high_power = np.mean(psd[freqs > 30])
            total_power = np.mean(psd)
            if high_power / (total_power + 1e-10) > 0.5:
                muscle_idx.append(i)
        return muscle_idx[:3]

    # ─── Step 4: Epoching ───
    def epoch(self, raw: mne.io.Raw) -> np.ndarray:
        """Segment into 2s epochs with 50% overlap."""
        epoch_len = int(self.cfg.epoch_duration * self.sfreq)
        overlap = int(self.cfg.epoch_overlap * epoch_len)
        step = epoch_len - overlap

        data = raw.get_data()
        n_samples = data.shape[1]
        n_epochs = (n_samples - epoch_len) // step + 1

        epochs = []
        for i in range(n_epochs):
            start = i * step
            end = start + epoch_len
            epoch_data = data[:, start:end]
            if np.max(np.abs(epoch_data)) < 200e-6:
                epochs.append(epoch_data)

        epochs = np.array(epochs)
        print(f"  Epoching: {n_epochs} → {len(epochs)} clean epochs "
              f"({self.cfg.epoch_duration}s, {self.cfg.epoch_overlap*100:.0f}% overlap)")
        return epochs

    # ─── Step 5: Frequency-Band Decomposition ───
    def bandpower(self, epoch_data: np.ndarray) -> np.ndarray:
        """Compute bandpower for each frequency band.
        Input:  (n_channels, n_samples)
        Output: (n_channels, n_bands)
        """
        sfreq = self.sfreq
        n_channels = epoch_data.shape[0]
        bandpowers = np.zeros((n_channels, len(self.bands)))

        for ch in range(n_channels):
            for b, (name, low, high) in enumerate(self.bands):
                freqs, psd = signal.welch(
                    epoch_data[ch], sfreq, nperseg=min(256, len(epoch_data[ch]))
                )
                idx = np.logical_and(freqs >= low, freqs < high)
                bandpowers[ch, b] = np.mean(psd[idx])

        return bandpowers

    # ─── Step 6: STFT → Spectrogram Tokens ───
    def compute_stft(self, epoch_data: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform for each channel.
        Input:  (n_channels, n_samples)
        Output: (n_channels, n_freq_bins, n_time_windows)
        """
        n_channels, n_samples = epoch_data.shape
        window = self.cfg.stft_window
        overlap = self.cfg.stft_overlap

        stfts = []
        for ch in range(n_channels):
            f, t, Zxx = signal.stft(
                epoch_data[ch],
                fs=self.sfreq,
                window="hann",
                nperseg=window,
                noverlap=overlap,
            )
            spectrogram = np.abs(Zxx)
            freq_mask = f <= self.cfg.bandpass_high
            spectrogram = spectrogram[freq_mask, :]
            stfts.append(spectrogram)

        return np.array(stfts)

    # ─── Step 7: Connectivity Matrices ───
    def compute_connectivity(self, epoch_data: np.ndarray) -> np.ndarray:
        """Phase-locking value (PLV) connectivity per frequency band.
        Input:  (n_channels, n_samples)
        Output: (n_bands, n_channels, n_channels)
        """
        n_channels = epoch_data.shape[0]
        n_bands = len(self.bands)
        connectivity = np.zeros((n_bands, n_channels, n_channels))

        for b, (name, low, high) in enumerate(self.bands):
            sos = signal.butter(
                4, [low, high], btype="bandpass", fs=self.sfreq, output="sos"
            )
            filtered = signal.sosfiltfilt(sos, epoch_data, axis=1)

            analytic = signal.hilbert(filtered, axis=1)
            phase = np.angle(analytic)

            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    phase_diff = phase[i] - phase[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    connectivity[b, i, j] = plv
                    connectivity[b, j, i] = plv

        return connectivity

    # ─── Full Pipeline ───
    def process(self, filepath: str) -> dict:
        """Run the full preprocessing pipeline on one subject."""
        print(f"Processing EEG: {Path(filepath).name}")

        raw = self.load_raw(filepath)
        raw = self.filter(raw)
        raw = self.apply_ica(raw)
        epochs = self.epoch(raw)

        stft_list = []
        bandpower_list = []
        conn_list = []

        for ep in tqdm(epochs, desc="  Extracting features"):
            stft = self.compute_stft(ep)
            bp = self.bandpower(ep)
            conn = self.compute_connectivity(ep)
            stft_list.append(stft)
            bandpower_list.append(bp)
            conn_list.append(conn)

        return {
            "stft": np.array(stft_list),
            "bandpower": np.array(bandpower_list),
            "connectivity": np.array(conn_list),
            "n_epochs": len(epochs),
            "n_channels": raw.info["nchan"],
            "sfreq": self.sfreq,
        }

    def process_batch(self, filepaths: list, output_dir: str):
        """Process a batch of subjects and save as .npz files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fp in tqdm(filepaths, desc="EEG batch"):
            subject_id = Path(fp).stem
            out_file = output_path / f"{subject_id}_eeg.npz"
            if out_file.exists():
                print(f"  Skipping (exists): {subject_id}")
                continue
            try:
                result = self.process(fp)
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
                print(f"  ERROR {subject_id}: {e}")
