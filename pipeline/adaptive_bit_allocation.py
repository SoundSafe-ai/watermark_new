# soundsafe/pipeline/adaptive_bit_allocation.py
# Perceptual significance + adaptive bit allocation + slot planning (bin, frame).
# Works with MooreGlasbergAnalyzer (or other analyzers providing per-band thresholds).

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import hashlib

@dataclass
class PerceptualSignificanceMetric:
    """Computes a per-band scalar 'significance' from quantized per-band thresholds."""
    eps: int = 1  # Integer epsilon for quantized operations
    method: str = "inverse"   # "inverse" | "log_inverse" | "softmax"
    use_median: bool = True   # Use median instead of mean for robustness

    def compute(self, quantized_band_thr_bt: np.ndarray) -> np.ndarray:
        """
        quantized_band_thr_bt: [BANDS, T] (quantized integer thresholds)
        Returns: sig_b: [BANDS] (time-aggregated significance as integers)
        """
        # Use median for robustness to outlier frames
        if self.use_median:
            agg = np.median(quantized_band_thr_bt, axis=1).astype(np.int64) + self.eps
        else:
            agg = np.mean(quantized_band_thr_bt, axis=1).astype(np.int64) + self.eps
        
        # Integer-friendly significance computation
        if self.method == "log_inverse":
            # Use log approximation for integers
            sig = np.maximum(1, 1000000 // (np.log(agg + 1) * 1000 + 1))
        elif self.method == "softmax":
            # Integer softmax approximation
            x = 1000000 // (agg + 1)  # Inverse as integers
            x_max = np.max(x)
            e = np.exp((x - x_max) / 1000)  # Scale down for stability
            sig = (e * 1000000 / (np.sum(e) + self.eps)).astype(np.int64)
        else:
            # Simple inverse for integers
            sig = 1000000 // (agg + 1)  # Scale up to avoid precision loss
        
        # Normalize to integers
        sig_sum = np.sum(sig) + self.eps
        return (sig * 1000000 // sig_sum).astype(np.int64)

@dataclass
class AdaptiveBitAllocator:
    total_bits: int
    allocation_strategy: str = "optimal"  # "proportional" | "threshold" | "optimal"
    min_bits_per_band: int = 0
    max_bits_per_band: int = 1_000_000
    seed: Optional[int] = None  # For deterministic tie-breaking (must be set in production)

    def _compute_deterministic_tie_key(self, band_idx: int, significance: int, quantized_thr: int) -> int:
        """Compute deterministic tie-breaking key for a band."""
        # Create deterministic hash from band index, significance, and quantized threshold
        data = f"{band_idx}_{significance}_{quantized_thr}".encode()
        if self.seed is not None:
            data += f"_{self.seed}".encode()
        return int(hashlib.sha256(data).hexdigest()[:8], 16)

    def allocate_bits(self, significance_b: np.ndarray, quantized_thr_b: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        significance_b: [BANDS] integer significance values
        quantized_thr_b: [BANDS] quantized thresholds for tie-breaking (optional)
        Returns: dict with 'bit_allocation' [BANDS] (integers that sum to <= total_bits)
        """
        if self.seed is None:
            # Enforce seed presence to avoid non-anchored determinism in production paths
            raise ValueError("AdaptiveBitAllocator.seed must be set (anchor-derived) for production determinism")
        B = significance_b.shape[0]
        alloc = np.zeros(B, dtype=np.int32)
        
        # Use quantized thresholds for tie-breaking if available
        if quantized_thr_b is None:
            quantized_thr_b = np.ones(B, dtype=np.int64)

        if self.allocation_strategy == "proportional":
            # Integer-based proportional allocation
            sig_sum = np.sum(significance_b) + 1
            raw = (significance_b * self.total_bits) // sig_sum
            alloc = raw.astype(np.int32)
            
            # Distribute remainder deterministically
            rem = self.total_bits - alloc.sum()
            if rem > 0:
                # Compute deterministic tie-breaking keys
                tie_keys = np.array([
                    self._compute_deterministic_tie_key(i, int(significance_b[i]), int(quantized_thr_b[i]))
                    for i in range(B)
                ])
                # Sort by tie key (descending) to get deterministic order
                idx = np.argsort(-tie_keys)
                for i in range(min(rem, len(idx))):
                    alloc[idx[i]] += 1

        elif self.allocation_strategy == "threshold":
            # Integer-based threshold allocation
            threshold = np.median(significance_b)  # Use median as threshold
            mask = significance_b >= threshold
            if not np.any(mask):
                # Fallback to proportional
                return self.allocate_bits(significance_b, quantized_thr_b)
            
            sig_masked = significance_b[mask]
            sig_sum = np.sum(sig_masked) + 1
            share = np.zeros_like(significance_b, dtype=np.int64)
            share[mask] = (sig_masked * self.total_bits) // sig_sum
            alloc = share.astype(np.int32)
            
            # Distribute remainder deterministically
            rem = self.total_bits - alloc.sum()
            if rem > 0:
                tie_keys = np.array([
                    self._compute_deterministic_tie_key(i, int(significance_b[i]), int(quantized_thr_b[i]))
                    for i in range(B) if mask[i]
                ])
                masked_indices = np.where(mask)[0]
                idx = masked_indices[np.argsort(-tie_keys)]
                for i in range(min(rem, len(idx))):
                    alloc[idx[i]] += 1

        else:  # "optimal" deterministic integer-based allocation
            # Compute integer raw allocation
            sig_sum = np.sum(significance_b) + 1
            raw = (significance_b * self.total_bits) // sig_sum
            alloc = raw.astype(np.int32)
            
            # Distribute remainder using deterministic tie-breaking
            rem = self.total_bits - alloc.sum()
            if rem > 0:
                # Compute tie-breaking keys for all bands
                tie_keys = np.array([
                    self._compute_deterministic_tie_key(i, int(significance_b[i]), int(quantized_thr_b[i]))
                    for i in range(B)
                ])
                # Sort by tie key (descending) and allocate remainder
                idx = np.argsort(-tie_keys)
                for i in range(min(rem, len(idx))):
                    alloc[idx[i]] += 1

        # Clamp per-band
        np.clip(alloc, self.min_bits_per_band, self.max_bits_per_band, out=alloc)
        
        # Final sanity: do not exceed total
        overflow = max(0, int(alloc.sum() - self.total_bits))
        if overflow > 0:
            # Remove from least significant bands (deterministic order)
            tie_keys = np.array([
                self._compute_deterministic_tie_key(i, int(significance_b[i]), int(quantized_thr_b[i]))
                for i in range(B)
            ])
            idx = np.argsort(tie_keys)  # Ascending order (least significant first)
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
    per_frame_weight_bt: np.ndarray | None = None,  # optional [BANDS, T] weights
    seed: Optional[int] = None,         # For deterministic tie-breaking
    quantize_magnitude: bool = True     # Use quantized magnitude for deterministic ranking
) -> List[Tuple[int, int]]:
    """
    For each band b, distribute bits_b over frames proportionally to per-frame weights,
    then choose the top-magnitude bins within that band for each chosen frame.
    Uses deterministic tie-breaking and quantized magnitude ranking.

    Returns: list of (f_bin, t) slots (length ~= sum(bits_per_band))
    """
    F, T = mag_ft.shape
    B = bits_per_band.shape[0]
    slots: List[Tuple[int, int]] = []
    
    # Quantize magnitude for deterministic ranking
    if quantize_magnitude:
        mag_quantized = (mag_ft * 1000000).astype(np.int64)  # Scale up and quantize
    else:
        mag_quantized = mag_ft.astype(np.int64)

    # Precompute per-band bin lists
    band_bins: List[np.ndarray] = []
    for b in range(B):
        band_bins.append(np.where(band_indices_f == b)[0])

    # Default per-frame weights: rank-based summary to reduce scale sensitivity
    if per_frame_weight_bt is None:
        per_frame_weight_bt = np.zeros((B, T), dtype=np.int64)
        for b in range(B):
            bins_b = band_bins[b]
            if bins_b.size == 0:
                continue
            # Compute per-bin ranks within this band for each frame
            ranks = np.zeros((bins_b.size, T), dtype=np.int64)
            for t in range(T):
                # Rank 1..K (highest magnitude gets highest rank)
                order = np.argsort(mag_quantized[bins_b, t])  # ascending
                ranks[:, t] = np.argsort(order) + 1           # 1..K
            # Use median rank across bins as frame weight (higher is stronger)
            per_frame_weight_bt[b] = np.median(ranks, axis=0).astype(np.int64) + 1

    def _compute_deterministic_tie_key(bin_idx: int, magnitude: int, frame: int, band: int) -> int:
        """Compute deterministic tie-breaking key for a bin."""
        data = f"{bin_idx}_{magnitude}_{frame}_{band}".encode()
        if seed is not None:
            data += f"_{seed}".encode()
        return int(hashlib.sha256(data).hexdigest()[:8], 16)

        # Distribute per band
    for b in range(B):
        bits_b = int(bits_per_band[b])
        if bits_b <= 0:
            continue
        bins_b = band_bins[b]
        if bins_b.size == 0:
                # No bins available in this band; redistribute deterministically
                # to the next bands according to tie-keys
                # Build tie-keys for all other bands
                other = [bb for bb in range(B) if bb != b and band_bins[bb].size > 0]
                if len(other) == 0:
                    continue
                tie_keys = np.array([
                    int(hashlib.sha256(f"redist_{bb}_{self.seed}".encode()).hexdigest()[:8], 16)
                    for bb in other
                ])
                order = np.argsort(-tie_keys)
                for i in range(bits_b):
                    tgt = other[order[i % len(other)]]
                    # assign one bit to frame with max weight in target band
                    t_star = int(np.argmax(per_frame_weight_bt[tgt]))
                    slots.append((int(np.median(band_bins[tgt]).astype(int) if band_bins[tgt].size>0 else 0), t_star))
                continue

        # Integer-based frame weight distribution
        w = per_frame_weight_bt[b].astype(np.int64)
        w_sum = np.sum(w) + 1
        
        # Distribute bits deterministically
        per_t = (w * bits_b) // w_sum
        rem = bits_b - np.sum(per_t)
        
        if rem > 0:
            # Distribute remainder using deterministic tie-breaking
            tie_keys = np.array([
                _compute_deterministic_tie_key(0, int(w[t]), t, b)  # Use frame as tie-breaker
                for t in range(T)
            ])
            idx = np.argsort(-tie_keys)  # Descending order
            for i in range(min(rem, len(idx))):
                per_t[idx[i]] += 1

        # For each frame t, pick top per_t[t] bins by normalized rank within band
        for t in range(T):
            k = int(per_t[t])
            if k <= 0:
                continue
                
            col = mag_quantized[bins_b, t]
            # Compute ranks within this frame for this band
            order = np.argsort(col)                  # ascending
            rank = np.argsort(order) + 1             # 1..K (higher = stronger)
            # Build deterministic key: (-rank, tie-key)
            bin_data = [(i, int(rank[i]), int(bins_b[i])) for i in range(col.size)]
            bin_data.sort(key=lambda x: (
                -x[1],  # Higher rank first
                _compute_deterministic_tie_key(x[2], x[1], t, b)
            ))
            top_idx = np.array([x[0] for x in bin_data[:min(k, len(bin_data))]])
                
            for ii in top_idx:
                f_bin = int(bins_b[int(ii)])
                slots.append((f_bin, t))

    return slots
