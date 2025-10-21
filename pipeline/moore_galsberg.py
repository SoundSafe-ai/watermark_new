# soundsafe/pipeline/moore_glasberg.py
# Psychoacoustics backend: Moore–Glasberg-style critical-band analysis on linear STFT bins.
# Provides: band mapping (bins -> critical bands) and per-band thresholds per frame.

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def bark_scale(f_hz: np.ndarray) -> np.ndarray:
    # Zwicker-like Bark approximation (close enough for band partitioning)
    return 13.0 * np.arctan(0.00076 * f_hz) + 3.5 * np.arctan((f_hz / 7500.0) ** 2)

@dataclass
class MooreGlasbergAnalyzer:
    sample_rate: int = 44100
    n_fft: int = 882
    hop_length: int = 441
    n_critical_bands: int = 24
    band_stat: str = "median"     # "mean" | "median" | "p95"

    def __post_init__(self):
        # Frequency bins (one-sided STFT)
        self.freqs = np.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        self.bark = bark_scale(self.freqs)
        # Partition bark range into equal-width bands
        b_min, b_max = self.bark[0], self.bark[-1]
        edges = np.linspace(b_min, b_max, self.n_critical_bands + 1)
        # Map each bin -> band index
        band_idx = np.digitize(self.bark, edges) - 1
        band_idx = np.clip(band_idx, 0, self.n_critical_bands - 1)
        self.band_indices = band_idx
        self.band_edges_bark = edges

    def band_thresholds(self, mag_ft: np.ndarray, thr_step: float | None = None) -> np.ndarray:
        """
        Compute per-band threshold proxy per frame.
        mag_ft: [F, T] linear magnitude (>=0)
        Returns: thresholds [BANDS, T]
        """
        F, T = mag_ft.shape
        assert F == (self.n_fft // 2 + 1), (
            f"Expected mag_ft F={self.n_fft // 2 + 1} to match n_fft, got {F}"
        )
        assert F == len(self.band_indices), (
            f"Band map F={len(self.band_indices)} mismatch with mag_ft F={F}"
        )
        B = self.n_critical_bands
        out = np.zeros((B, T), dtype=np.float32)
        valid_mask = np.zeros(B, dtype=bool)
        for b in range(B):
            mask = (self.band_indices == b)
            if not np.any(mask):
                # Mark band invalid and continue
                valid_mask[b] = False
                continue
            band_vals = mag_ft[mask, :]  # [F_b, T]
            if self.band_stat == "median":
                thr = np.median(band_vals, axis=0)
            elif self.band_stat == "p95":
                thr = np.percentile(band_vals, 95, axis=0)
            else:
                thr = np.mean(band_vals, axis=0)
            out[b] = np.asarray(thr, dtype=np.float32)
            valid_mask[b] = True
        # Small floor to avoid zeros
        out = np.maximum(out, 1e-9)
        # Optional quantization to align with allocator
        if thr_step is not None and thr_step > 0:
            out = (np.floor(out / thr_step) * thr_step).astype(np.float32)
        # Attach mask for downstream handling (bands with no bins)
        self._last_valid_bands = valid_mask
        return out

    def bands_for_bins(self, bins: np.ndarray) -> np.ndarray:
        return self.band_indices[bins]

    def band_bin_counts(self) -> np.ndarray:
        """Return per-band bin counts for telemetry and deterministic redistribution."""
        counts = np.zeros(self.n_critical_bands, dtype=np.int32)
        for b in range(self.n_critical_bands):
            counts[b] = int(np.sum(self.band_indices == b))
        return counts

    def valid_bands_mask(self) -> np.ndarray:
        """Mask from the last band_thresholds() call indicating bands with any bins."""
        return getattr(self, "_last_valid_bands", np.ones(self.n_critical_bands, dtype=bool))
