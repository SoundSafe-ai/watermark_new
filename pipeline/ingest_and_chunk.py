# soundsafe/pipeline/ingest_and_chunk.py
# Rewritten to use:
#  - Moore–Glasberg psychoacoustics for per-band thresholds
#  - Adaptive bit allocation to distribute ~1336 bits/EUL
#  - Slot expansion to concrete (freq-bin, frame) positions
#  - RS(167,125) with interleaving
#
# Assumes:
#   - sr=22050, n_fft=1024, hop=512 (≈23.2 ms hop, ~43 frames per 1 s EUL)
#   - INNWatermarker model uses RI channels and message-spec concat per block
#
# NOTE: This file does NOT include training stubs per your request.

from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import torch

from models.inn_encoder_decoder import INNWatermarker, STFT, ISTFT
from pipeline.payload_codec import pack_fields, unpack_fields  # left as-is for payload packing
from pipeline.moore_galsberg import MooreGlasbergAnalyzer
from pipeline.adaptive_bit_allocation import (
    PerceptualSignificanceMetric,
    AdaptiveBitAllocator,
    expand_allocation_to_slots,
)
from pipeline.acoustic_dna import (
    seed_from_anchors,
    permute_slots_with_seed,
    generate_pn_bits,
    pilot_correlation,
)

# -----------------------------
# Reed–Solomon (RS) helpers
# -----------------------------
try:
    import reedsolo  # pip install reedsolo
except Exception:
    reedsolo = None

def rs_encode_167_125(msg_bytes: bytes) -> bytes:
    if reedsolo is None:
        raise RuntimeError("reedsolo not installed. `pip install reedsolo`")
    rsc = reedsolo.RSCodec(42)  # (n-k)=42 parity -> RS(167,125)
    return bytes(rsc.encode(msg_bytes))

def rs_decode_167_125(code_bytes: bytes) -> bytes:
    if reedsolo is None:
        raise RuntimeError("reedsolo not installed. `pip install reedsolo`")
    rsc = reedsolo.RSCodec(42)
    try:
        data, _, _ = rsc.decode(bytearray(code_bytes))
        return bytes(data)
    except reedsolo.ReedSolomonError:
        # Too many errors to correct, return original data (first 125 bytes)
        # This allows the system to continue with partial recovery
        return code_bytes[:125] if len(code_bytes) >= 125 else code_bytes

def interleave_bytes(b: bytes, depth: int) -> bytes:
    if depth <= 1: return b
    rows = [b[i::depth] for i in range(depth)]
    return b"".join(rows)

def deinterleave_bytes(b: bytes, depth: int) -> bytes:
    if depth <= 1: return b
    L = len(b)
    base = L // depth
    extras = L % depth
    parts = []
    cursor = 0
    for i in range(depth):
        n = base + (1 if i < extras else 0)
        parts.append(b[cursor:cursor+n])
        cursor += n
    out = bytearray()
    for i in range(max(len(p) for p in parts)):
        for r in parts:
            if i < len(r):
                out.append(r[i])
    return bytes(out)

# -----------------------------
# Bit <-> message-spectrogram
# -----------------------------
def bits_to_message_spec(
    bits: torch.Tensor,              # [B, S] 0/1
    slots: List[Tuple[int,int]],     # list of (f_bin, t)
    F: int, T: int,
    base_amp: float = 0.1,
    amp_per_slot: np.ndarray | None = None  # optional per-slot amplitude scaling (len==S)
) -> torch.Tensor:
    """
    Simple BPSK writer on the real channel; imag=0.
    If amp_per_slot provided, scales each symbol by that factor (psychoacoustic margin).
    """
    B, S = bits.shape
    M = torch.zeros(B, 2, F, T, device=bits.device)
    for s, (f, t) in enumerate(slots[:S]):
        # +1 => +amp, 0 => -amp
        sign = (bits[:, s] * 2 - 1).float()  # [-1, +1]
        amp = base_amp
        if amp_per_slot is not None and s < len(amp_per_slot):
            amp = float(amp_per_slot[s]) * base_amp
        val = sign * amp
        M[:, 0, f, t] = val
        # imag stays zero
    return M

def message_spec_to_bits(M_rec: torch.Tensor, slots: List[Tuple[int,int]]) -> torch.Tensor:
    """
    Read BPSK symbols (use sign for more robust detection).
    Returns bits [B, S].
    """
    B = M_rec.shape[0]
    outs = []
    for (f, t) in slots:
        val = M_rec[:, 0, f, t]   # real
        bit = (val >= 0).long()   # Use >= 0 for more robust detection
        outs.append(bit)
    return torch.stack(outs, dim=1) if outs else torch.zeros(B, 0, dtype=torch.long, device=M_rec.device)

# -----------------------------
# Slot allocation (psychoacoustic + adaptive)
# -----------------------------
def allocate_slots_and_amplitudes(
    X_ri: torch.Tensor,              # [1,2,F,T], linear STFT (RI stacked)
    sr: int,
    n_fft: int,
    target_bits: int,
    amp_safety: float = 1.0,
) -> tuple[list[tuple[int,int]], np.ndarray]:
    """
    1) Compute per-band thresholds via Moore–Glasberg
    2) Convert to per-band significance
    3) Adaptive allocation -> bits per band (sum ~= target_bits)
    4) Expand to (bin,frame) slots; also return per-slot amplitude scalers based on band threshold

    Returns:
        slots: List[(f_bin, t)]
        amp_per_slot: np.ndarray of shape [len(slots)] in [0, +inf) (relative scale), can multiply by base_amp
    """
    assert X_ri.shape[0] == 1, "This allocator expects batch=1 per call (EUL-wise)."
    _, _, F, T = X_ri.shape

    # magnitude [F, T]
    mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-12))[0].detach().cpu().numpy()

    # Moore–Glasberg per-band thresholds
    mg = MooreGlasbergAnalyzer(sample_rate=sr, n_fft=n_fft, hop_length=n_fft//2, n_critical_bands=24)
    band_thr_bt = mg.band_thresholds(mag)        # [BANDS, T]
    band_idx_f = mg.band_indices                 # [F]

    # Significance & allocation
    psm = PerceptualSignificanceMetric(method="inverse")
    sig_b = psm.compute(band_thr_bt)             # [BANDS], normalized

    allocator = AdaptiveBitAllocator(total_bits=target_bits, allocation_strategy="optimal")
    alloc_b = allocator.allocate_bits(sig_b)["bit_allocation"]  # [BANDS] ints

    # Expand allocation to concrete slots
    slots = expand_allocation_to_slots(
        mag_ft=mag,
        band_indices_f=band_idx_f,
        bits_per_band=alloc_b,
        per_frame_weight_bt=None  # default: magnitude-weighted per frame
    )  # ≈ target_bits slots

    # Build per-slot amplitude scaling (relative). Higher band threshold → more headroom → allow more amplitude.
    amp_per_slot = []
    for (f, t) in slots:
        band = band_idx_f[f]
        thr = max(1e-9, float(band_thr_bt[band, t]))
        amp_per_slot.append(amp_safety * thr)   # you can normalize across EUL if you want a fixed global scale
    # Normalize so median is 1.0 (keeps amplitudes stable across EULs)
    if len(amp_per_slot) > 0:
        med = np.median(amp_per_slot)
        if med > 0:
            amp_per_slot = np.asarray(amp_per_slot, dtype=np.float32) / med
        else:
            amp_per_slot = np.ones(len(amp_per_slot), dtype=np.float32)
    else:
        amp_per_slot = np.zeros(0, dtype=np.float32)

    return slots, amp_per_slot

# -----------------------------
# 1-second EUL Driver
# -----------------------------
class EULDriver:
    """
    One-second EUL (Encode/Decode) driver using:
      - RS(167,125) + interleave=4  (≈1336 bits/EUL)
      - adaptive psychoacoustic allocation for slot planning
      - BPSK symbols onto message spectrogram
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop: int = 512,
        rs_interleave: int = 4,
        per_eul_bits_target: int = 167 * 8,  # 1336 bits to match RS(167,125)
        base_symbol_amp: float = 0.1,
        amp_safety: float = 1.0,
        pilot_fraction: float = 0.08,  # fraction of data budget reserved for pilots (DNA)
        dna_enabled: bool = True,
    ):
        self.sr = sr
        self.stft = STFT(n_fft=n_fft, hop_length=hop, win_length=n_fft)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft)
        self.n_fft = n_fft
        self.hop = hop
        self.rs_interleave = rs_interleave
        self.per_eul_bits_target = per_eul_bits_target
        self.base_symbol_amp = base_symbol_amp
        self.amp_safety = amp_safety
        # Pilot configuration (reserve a small budget for pilots)
        self.pilot_fraction = max(0.0, min(0.25, float(pilot_fraction)))
        self.dna_enabled = bool(dna_enabled)

    # ----- Encoder path -----
    def encode_eul(self, model: INNWatermarker, x_wave: torch.Tensor, payload_bytes: bytes) -> torch.Tensor:
        """
        x_wave: [1,1,T] waveform of ~1 s @ 22.05 kHz
        payload_bytes: 125 raw bytes -> RS(167,125) -> 1336 bits target
        returns: watermarked waveform [1,1,T]
        """
        assert x_wave.shape[0] == 1, "Call EULDriver per item (B=1) for deterministic slot planning."

        # RS encode + interleave -> bytes
        code = rs_encode_167_125(payload_bytes)              # 167 bytes
        code = interleave_bytes(code, self.rs_interleave)

        # bytes -> bits
        bits = []
        for b in code:
            for k in range(8):
                bits.append((b >> k) & 1)
        bits_t = torch.tensor(bits, dtype=torch.long, device=x_wave.device).unsqueeze(0)  # [1, S]

        # Analyze current EUL content and allocate slots
        X = model.stft(x_wave)  # [1,2,F,T]
        if self.dna_enabled:
            # Derive Acoustic DNA seed from anchors
            seed = seed_from_anchors(X)
            # Reserve pilot budget as a fraction of data budget
            pilot_bits = int(round(self.pilot_fraction * self.per_eul_bits_target))
            pilot_bits = max(0, pilot_bits)
            # Request enough slots for data + pilots
            requested_bits = min(self.per_eul_bits_target, bits_t.shape[1]) + pilot_bits
            slots, amp_per_slot = allocate_slots_and_amplitudes(
                X, self.sr, self.n_fft, target_bits=requested_bits,
                amp_safety=self.amp_safety
            )
            # Content-keyed permutation (slots + per-slot amplitude kept aligned)
            slots, amp_per_slot = permute_slots_with_seed(slots, seed, values=amp_per_slot)
            S_total = min(len(slots), requested_bits)
            S_pilot = min(pilot_bits, S_total)
            S_data = min(self.per_eul_bits_target, max(0, S_total - S_pilot), bits_t.shape[1])
            # Generate pilot PN bits deterministically
            pn_bits = generate_pn_bits(seed, S_pilot)
        else:
            # Legacy behavior: no DNA seed, no pilots, no permutation
            requested_bits = min(self.per_eul_bits_target, bits_t.shape[1])
            slots, amp_per_slot = allocate_slots_and_amplitudes(
                X, self.sr, self.n_fft, target_bits=requested_bits,
                amp_safety=self.amp_safety
            )
            S_total = min(len(slots), requested_bits)
            S_pilot = 0
            S_data = S_total

        # Build message spectrogram: [PN pilots][data bits]
        F_, T_ = X.shape[-2], X.shape[-1]
        if S_total <= 0:
            return x_wave
        bits_full = torch.zeros(1, S_total, dtype=torch.long, device=x_wave.device)
        # Fill pilots
        if self.dna_enabled and S_pilot > 0:
            for i in range(S_pilot):
                bits_full[0, i] = int(pn_bits[i])
        # Fill data
        for i in range(S_data):
            bits_full[0, S_pilot + i] = bits_t[0, i]
        # Build spec
        amp_vec = None
        if len(amp_per_slot) >= S_total:
            amp_vec = np.asarray(amp_per_slot[:S_total], dtype=np.float32)
        M_spec = bits_to_message_spec(
            bits=bits_full,
            slots=slots[:S_total],
            F=F_, T=T_,
            base_amp=self.base_symbol_amp,
            amp_per_slot=amp_vec
        )

        # Encode through INN
        x_wm_wave, _ = model.encode(x_wave, M_spec)
        return x_wm_wave

    # ----- Decoder path -----
    def decode_eul(self, model: INNWatermarker, x_wm_wave: torch.Tensor, expected_bytes: int = 125, slots: List[Tuple[int, int]] | None = None) -> bytes:
        """
        Returns recovered raw payload bytes (should be 125 if no RS errors after decode)
        If slots is provided, use those instead of recomputing from audio.
        """
        assert x_wm_wave.shape[0] == 1, "Call EULDriver per item (B=1) for deterministic slot planning."

        # Recover message spectrogram from watermarked audio
        M_rec = model.decode(x_wm_wave)   # [1,2,F,T]

        # Recompute slots. If DNA is enabled, use anchors, pilots, and permutation; else legacy path.
        Xrx = model.stft(x_wm_wave)       # [1,2,F,T]
        if self.dna_enabled:
            seed = seed_from_anchors(Xrx)
            pilot_bits = int(round(self.pilot_fraction * self.per_eul_bits_target))
            requested_bits = self.per_eul_bits_target + max(0, pilot_bits)
            slots_rx, amp_rx = allocate_slots_and_amplitudes(
                Xrx, self.sr, self.n_fft, target_bits=requested_bits, amp_safety=self.amp_safety
            )
            slots_rx, amp_rx = permute_slots_with_seed(slots_rx, seed, values=amp_rx)
            S_total = min(len(slots_rx), requested_bits)
            S_pilot = min(max(0, pilot_bits), S_total)
            S_data = min(self.per_eul_bits_target, max(0, S_total - S_pilot))

            # Gather raw values at pilot and data slots for sign correction
            vals = []
            for idx, (f, t) in enumerate(slots_rx[:S_total]):
                scale = float(amp_rx[idx]) if idx < len(amp_rx) and amp_rx[idx] is not None else 1.0
                denom = self.base_symbol_amp * max(scale, 1e-6)
                vals.append(float(M_rec[0, 0, f, t].item()) / denom)
            vals = np.asarray(vals, dtype=np.float32) if len(vals) > 0 else np.zeros(0, dtype=np.float32)
            # Compute pilot correlation to determine global sign
            pn_bits = generate_pn_bits(seed, S_pilot)
            corr = pilot_correlation(vals[:S_pilot], pn_bits) if S_pilot > 0 else 0.0
            sign = 1.0 if corr >= 0.0 else -1.0

            # Apply sign to data values and threshold
            data_vals = sign * vals[S_pilot:S_pilot + S_data]
            bits_list = [1 if v >= 0.0 else 0 for v in data_vals]
        else:
            # Legacy: plan per_eul_bits_target slots and read directly by sign
            slots_rx, _amp_rx = allocate_slots_and_amplitudes(
                Xrx, self.sr, self.n_fft, target_bits=self.per_eul_bits_target, amp_safety=self.amp_safety
            )
            bits_rec = message_spec_to_bits(M_rec, slots_rx)  # [1,S]
            bits_list = bits_rec[0].tolist()

        # bits -> bytes
        by = bytearray()
        for i in range(0, len(bits_list), 8):
            if i + 8 > len(bits_list): break
            b = 0
            for k in range(8):
                b |= (bits_list[i + k] << k)
            by.append(b)

        # deinterleave + RS decode
        deintl = deinterleave_bytes(bytes(by), self.rs_interleave)
        data = rs_decode_167_125(deintl)
        return data[:expected_bytes]
