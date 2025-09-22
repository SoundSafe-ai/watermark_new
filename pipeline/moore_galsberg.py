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
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 512
    n_critical_bands: int = 24
    band_stat: str = "mean"     # "mean" | "median" | "p95"

    def __post_init__(self):
        # Frequency bins (one-sided STFT)
        self.freqs = np.linspace(0.0, self.sample_rate / 2.0, self.n_fft // 2 + 1)
        self.bark = bark_scale(self.freqs)
        # Partition bark range into equal-width bands
        b_min, b_max = self.bark[0], self.bark[-1] if self.bark[-1] > 0 else self.bark[-2]
        edges = np.linspace(b_min, b_max, self.n_critical_bands + 1)
        # Map each bin -> band index
        band_idx = np.digitize(self.bark, edges) - 1
        band_idx = np.clip(band_idx, 0, self.n_critical_bands - 1)
        self.band_indices = band_idx
        self.band_edges_bark = edges

    def band_thresholds(self, mag_ft: np.ndarray) -> np.ndarray:
        """
        Compute per-band threshold proxy per frame.
        mag_ft: [F, T] linear magnitude (>=0)
        Returns: thresholds [BANDS, T]
        """
        F, T = mag_ft.shape
        B = self.n_critical_bands
        out = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            mask = (self.band_indices == b)
            if not np.any(mask):
                continue
            band_vals = mag_ft[mask, :]  # [F_b, T]
            if self.band_stat == "median":
                thr = np.median(band_vals, axis=0)
            elif self.band_stat == "p95":
                thr = np.percentile(band_vals, 95, axis=0)
            else:
                thr = np.mean(band_vals, axis=0)
            out[b] = np.asarray(thr, dtype=np.float32)
        # Small floor to avoid zeros
        return np.maximum(out, 1e-9)

    def bands_for_bins(self, bins: np.ndarray) -> np.ndarray:
        return self.band_indices[bins]
