from __future__ import annotations
"""
Acoustic DNA helpers:
- Extract a stable, content-tied anchor signature from a 1 s EUL spectrogram
- Derive a deterministic seed from the signature
- Permute allocator slots with the seed (content-keyed mapping)
- Generate a deterministic PN pilot sequence for channel/sign estimation

Design notes:
- We avoid heavy dependencies and use robust but simple anchors: per-frame top-k
  frequency bins aggregated into a histogram across the EUL. The resulting top
  histogram indices are quantized and hashed to produce a seed. This is stable
  across small EQ/AGC variations and common codecs.
- Pilot bits are derived deterministically from the same seed, ensuring encoder
  and decoder agree without side-channels.
"""

import hashlib
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def _magnitude_from_ri(X_ri: torch.Tensor) -> np.ndarray:
    """
    X_ri: [1,2,F,T] torch tensor, returns mag [F,T] as numpy.
    """
    assert X_ri.ndim == 4 and X_ri.shape[0] == 1 and X_ri.shape[1] == 2
    mag = torch.sqrt(torch.clamp(X_ri[:, 0] ** 2 + X_ri[:, 1] ** 2, min=1e-12))[0]
    return mag.detach().cpu().numpy()


def extract_anchor_signature(X_ri: torch.Tensor, *, top_bins_per_frame: int = 8, max_hist_bins: int = 64) -> bytes:
    """
    Build a compact, repeatable signature from the EUL content by:
      1) For each frame, taking the top-K frequency bins by magnitude
      2) Accumulating a histogram over frequency bins
      3) Taking the top-M histogram indices and their counts
      4) Serializing indices+counts into bytes

    Returns: signature bytes suitable for hashing.
    """
    mag = _magnitude_from_ri(X_ri)  # [F, T]
    F, T = mag.shape
    K = int(max(1, min(top_bins_per_frame, F)))
    hist = np.zeros(F, dtype=np.int32)
    for t in range(T):
        col = mag[:, t]
        if K >= F:
            idx = np.arange(F)
        else:
            # argpartition for efficiency
            idx = np.argpartition(-col, K - 1)[:K]
        hist[idx] += 1
    M = int(max(1, min(max_hist_bins, F)))
    top = np.argpartition(-hist, M - 1)[:M]
    top_sorted = top[np.argsort(-hist[top])]  # sort by count desc
    # Serialize as (index uint16, count uint16) tuples, clipped
    out = bytearray()
    for i in top_sorted:
        idx16 = int(np.clip(i, 0, 65535))
        cnt16 = int(np.clip(hist[i], 0, 65535))
        out.append((idx16 >> 8) & 0xFF); out.append(idx16 & 0xFF)
        out.append((cnt16 >> 8) & 0xFF); out.append(cnt16 & 0xFF)
    return bytes(out)


def seed_from_signature(sig: bytes) -> int:
    """Map signature bytes to a 64-bit deterministic integer seed."""
    h = hashlib.sha256(sig).digest()
    # Take first 8 bytes as unsigned 64-bit
    seed = int.from_bytes(h[:8], byteorder="big", signed=False)
    # Avoid degenerate seeds (0/1) by xoring a constant
    return seed ^ 0x9E3779B97F4A7C15


def seed_from_anchors(X_ri: torch.Tensor) -> int:
    """Convenience: extract signature from X_ri then hash to seed."""
    sig = extract_anchor_signature(X_ri)
    return seed_from_signature(sig)


def permute_slots_with_seed(
    slots: Sequence[Tuple[int, int]],
    seed: int,
    *,
    values: Sequence[float]
    | Sequence[np.ndarray]
    | Sequence[torch.Tensor]
    | np.ndarray
    | torch.Tensor
    | None = None,
) -> List[Tuple[int, int]] | tuple[List[Tuple[int, int]], object]:
    """
    Deterministically shuffle slot order using the provided seed.

    When ``values`` is provided, it is permuted with the exact same shuffling
    pattern to keep auxiliary metadata (e.g. per-slot amplitudes) aligned with
    the slots selected during encoding. The helper returns either just the
    permuted slot list or a ``(slots, values)`` tuple depending on the caller's
    needs.
    """

    indices = list(range(len(slots)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    perm_slots = [slots[i] for i in indices]
    if values is None:
        return perm_slots

    if isinstance(values, torch.Tensor):
        flat = values.flatten()
        take = min(len(flat), len(perm_slots))
        perm_tensor = flat[torch.tensor(indices[:take], device=flat.device)]
        return perm_slots[:take], perm_tensor

    if isinstance(values, np.ndarray):
        flat = values.reshape(-1)
        take = min(len(flat), len(perm_slots))
        perm_arr = flat[np.array(indices[:take], dtype=np.int64)]
        return perm_slots[:take], perm_arr

    values_list = list(values)
    take = min(len(values_list), len(perm_slots))
    perm_values = [values_list[i] for i in indices[:take]]
    return perm_slots[:take], perm_values


def generate_pn_bits(seed: int, length: int) -> List[int]:
    """
    Deterministic pseudo-noise binary sequence in {0,1} using a seed-derived RNG.
    """
    rng = random.Random(seed ^ 0xA5A5A5A5A5A5A5A5)
    return [rng.randrange(2) for _ in range(max(0, int(length)))]


def pilot_correlation(rec_values: np.ndarray, pilot_bits: Iterable[int]) -> float:
    """
    Compute signed correlation between recovered real-valued symbols at pilot slots
    and expected pilot bits. Returns a scalar where sign indicates global sign.
    rec_values: array of real logits/values (not thresholded), shape [P]
    pilot_bits: iterable of 0/1 length P
    """
    pb = np.asarray([1.0 if int(b) else -1.0 for b in pilot_bits], dtype=np.float32)
    rv = np.asarray(rec_values, dtype=np.float32)
    if rv.size == 0 or pb.size == 0:
        return 0.0
    n = min(rv.size, pb.size)
    return float(np.mean(rv[:n] * pb[:n]))


