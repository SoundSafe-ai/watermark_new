# soundsafe/pipeline/adaptive_bit_allocation.py
# Perceptual significance + adaptive bit allocation + slot planning (bin, frame).
# Works with MooreGlasbergAnalyzer (or other analyzers providing per-band thresholds).

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class PerceptualSignificanceMetric:
    """Computes a per-band scalar 'significance' from per-band thresholds."""
    eps: float = 1e-9
    method: str = "inverse"   # "inverse" | "log_inverse" | "softmax"

    def compute(self, band_thr_bt: np.ndarray) -> np.ndarray:
        """
        band_thr_bt: [BANDS, T] (linear magnitude thresholds)
        Returns: sig_b: [BANDS] (time-aggregated significance)
        """
        # Aggregate over time (mean). You can switch to median if needed.
        agg = band_thr_bt.mean(axis=1) + self.eps
        if self.method == "log_inverse":
            sig = 1.0 / np.log1p(agg)
        elif self.method == "softmax":
            x = 1.0 / (agg)
            e = np.exp(x - x.max())
            sig = e / (e.sum() + self.eps)
        else:
            sig = 1.0 / agg
        # Normalize
        return sig / (sig.sum() + self.eps)

@dataclass
class AdaptiveBitAllocator:
    total_bits: int
    allocation_strategy: str = "optimal"  # "proportional" | "threshold" | "optimal"
    min_bits_per_band: int = 0
    max_bits_per_band: int = 1_000_000

    def allocate_bits(self, significance_b: np.ndarray, threshold: float = 0.0) -> Dict[str, np.ndarray]:
        """
        significance_b: [BANDS] normalized (sum ~ 1)
        Returns: dict with 'bit_allocation' [BANDS] (integers that sum to <= total_bits)
        """
        B = significance_b.shape[0]
        alloc = np.zeros(B, dtype=np.int32)

        if self.allocation_strategy == "proportional":
            raw = significance_b * self.total_bits
            alloc = np.floor(raw).astype(np.int32)
            # distribute remainder by largest fractional part
            rem = self.total_bits - alloc.sum()
            if rem > 0:
                frac = raw - alloc
                idx = np.argsort(-frac)
                alloc[idx[:rem]] += 1

        elif self.allocation_strategy == "threshold":
            mask = significance_b >= threshold
            if not np.any(mask):
                # Fallback to proportional
                return self.allocate_bits(significance_b, threshold=0.0)
            share = np.zeros_like(significance_b)
            share[mask] = significance_b[mask] / (significance_b[mask].sum() + 1e-9)
            alloc = np.floor(share * self.total_bits).astype(np.int32)
            rem = self.total_bits - alloc.sum()
            if rem > 0:
                frac = (share * self.total_bits) - alloc
                idx = np.argsort(-frac)
                alloc[idx[:rem]] += 1

        else:  # "optimal" greedy knapsack on utility-per-bit (utility = significance)
            # Start with zero, add one bit at a time to the best band until budget exhausted
            utility = significance_b.copy()
            for _ in range(self.total_bits):
                b = int(np.argmax(utility))
                alloc[b] += 1
                # Diminishing returns: reduce utility slightly as band gets bits
                utility[b] *= 0.999

        # Clamp per-band
        np.clip(alloc, self.min_bits_per_band, self.max_bits_per_band, out=alloc)
        # Final sanity: do not exceed total
        overflow = max(0, int(alloc.sum() - self.total_bits))
        if overflow > 0:
            # Remove from least significant bands
            idx = np.argsort(significance_b)
            i = 0
            while overflow > 0 and i < len(idx):
                take = min(overflow, alloc[idx[i]])
                alloc[idx[i]] -= take
                overflow -= take
                i += 1

        return {"bit_allocation": alloc}

# ---------- Slot planner (turn per-band bits into (bin, frame) slots) ----------

def expand_allocation_to_slots(
    mag_ft: np.ndarray,                 # [F, T] linear magnitude
    band_indices_f: np.ndarray,         # [F] -> band id
    bits_per_band: np.ndarray,          # [BANDS] ints
    per_frame_weight_bt: np.ndarray | None = None  # optional [BANDS, T] weights
) -> List[Tuple[int, int]]:
    """
    For each band b, distribute bits_b over frames proportionally to per-frame weights,
    then choose the top-magnitude bins within that band for each chosen frame.

    Returns: list of (f_bin, t) slots (length ~= sum(bits_per_band))
    """
    F, T = mag_ft.shape
    B = bits_per_band.shape[0]
    slots: List[Tuple[int, int]] = []

    # Precompute per-band bin lists
    band_bins: List[np.ndarray] = []
    for b in range(B):
        band_bins.append(np.where(band_indices_f == b)[0])

    # Default per-frame weights: magnitude (higher mag -> more headroom)
    if per_frame_weight_bt is None:
        per_frame_weight_bt = np.zeros((B, T), dtype=np.float32)
        for b in range(B):
            bins_b = band_bins[b]
            if bins_b.size == 0:
                continue
            per_frame_weight_bt[b] = mag_ft[bins_b, :].mean(axis=0) + 1e-12

    # Distribute per band
    for b in range(B):
        bits_b = int(bits_per_band[b])
        if bits_b <= 0:
            continue
        bins_b = band_bins[b]
        if bins_b.size == 0:
            continue

        # Normalize frame weights
        w = per_frame_weight_bt[b]
        w = w / (w.sum() + 1e-12)
        # Multinomial draw for integer per-frame allocation (deterministic via top-k could be used too)
        # Here we use a deterministic rounding approach for reproducibility:
        desired = w * bits_b
        per_t = np.floor(desired).astype(np.int32)
        rem = bits_b - per_t.sum()
        if rem > 0:
            frac = desired - per_t
            idx = np.argsort(-frac)
            per_t[idx[:rem]] += 1

        # For each frame t, pick top per_t[t] bins by magnitude within band
        for t in range(T):
            k = int(per_t[t])
            if k <= 0:
                continue
            col = mag_ft[bins_b, t]
            if k >= col.size:
                top_idx = np.arange(col.size)
            else:
                top_idx = np.argpartition(-col, k-1)[:k]
            for ii in top_idx:
                f_bin = int(bins_b[int(ii)])
                slots.append((f_bin, t))

    return slots
