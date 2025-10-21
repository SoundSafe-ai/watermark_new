# soundsafe/pipeline/ingest_and_chunk.py
# Rewritten to use:
#  - Moore–Glasberg psychoacoustics for per-band thresholds
#  - Adaptive bit allocation to distribute ~2040 bits/EUL (255 bytes RS-coded)
#  - Slot expansion to concrete (freq-bin, frame) positions
#  - RS(255,191) with interleaving
#  - Quantized thresholds for robust allocation
#  - Deterministic anchor-seeded allocation
#
# Assumes:
#   - sr=44100, n_fft=882, hop=441 (≈10 ms hop, ~100 frames per 1 s EUL)
#   - INNWatermarker model uses RI channels and message-spec concat per block
#
# NOTE: This file does NOT include training stubs per your request.

from __future__ import annotations
import math
import hashlib
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchaudio

from models.inn_encoder_decoder import INNWatermarker, STFT, ISTFT
from pipeline.payload_codec import pack_fields, unpack_fields  # left as-is for payload packing
from pipeline.moore_galsberg import MooreGlasbergAnalyzer
from pipeline.adaptive_bit_allocation import (
    PerceptualSignificanceMetric,
    AdaptiveBitAllocator,
    expand_allocation_to_slots,
)

# -----------------------------
# Reed–Solomon (RS) helpers
# -----------------------------
try:
    import reedsolo  # pip install reedsolo
except Exception:
    reedsolo = None

def rs_encode_255_191(msg_bytes: bytes) -> bytes:
    if reedsolo is None:
        raise RuntimeError("reedsolo not installed. `pip install reedsolo`")
    rsc = reedsolo.RSCodec(64)  # (n-k)=64 parity -> RS(255,191)
    return bytes(rsc.encode(msg_bytes))

def rs_decode_255_191(code_bytes: bytes) -> bytes:
    if reedsolo is None:
        raise RuntimeError("reedsolo not installed. `pip install reedsolo`")
    rsc = reedsolo.RSCodec(64)
    try:
        data, _, _ = rsc.decode(bytearray(code_bytes))
        return bytes(data)
    except reedsolo.ReedSolomonError:
        # Too many errors to correct, return original data (first 191 bytes)
        # This allows the system to continue with partial recovery
        return code_bytes[:191] if len(code_bytes) >= 191 else code_bytes

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
    thr_step: float = 1e-6,          # Quantization step for thresholds
    use_anchor_seeding: bool = True,  # Use deterministic anchor-based seeding
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
    
    # Quantize thresholds for robustness
    quantized_thr_bt = np.floor(band_thr_bt / thr_step) * thr_step
    
    # Compute anchor seed if requested
    seed = None
    if use_anchor_seeding:
        # Find 4 highest-energy critical bands
        band_energies = np.mean(quantized_thr_bt, axis=1)  # [BANDS]
        top_4_bands = np.argsort(band_energies)[-4:]
        
        # Create deterministic hash from top bands
        anchor_data = b''.join([
            int(quantized_thr_bt[band, 0] * 1e6).to_bytes(4, 'big') 
            for band in top_4_bands
        ])
        seed_hash = hashlib.sha256(anchor_data).digest()
        seed = int.from_bytes(seed_hash[:4], 'big')  # 32-bit seed for numpy compatibility
        
        # Set random seed for deterministic allocation
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Significance & allocation using quantized thresholds
    psm = PerceptualSignificanceMetric(method="inverse", use_median=True)
    sig_b = psm.compute(quantized_thr_bt)             # [BANDS], normalized

    # Compute quantized band thresholds for tie-breaking
    quantized_thr_b = np.median(quantized_thr_bt, axis=1).astype(np.int64)
    
    allocator = AdaptiveBitAllocator(
        total_bits=target_bits, 
        allocation_strategy="optimal",
        seed=seed  # Use anchor seed for deterministic allocation
    )
    alloc_b = allocator.allocate_bits(sig_b, quantized_thr_b)["bit_allocation"]  # [BANDS] ints

    # Expand allocation to concrete slots
    slots = expand_allocation_to_slots(
        mag_ft=mag,
        band_indices_f=band_idx_f,
        bits_per_band=alloc_b,
        per_frame_weight_bt=None,  # default: magnitude-weighted per frame
        seed=seed,  # Use anchor seed for deterministic slot expansion
        quantize_magnitude=True  # Use quantized magnitude for deterministic ranking
    )  # ≈ target_bits slots

    # Build per-slot amplitude scaling (relative). Higher band threshold → more headroom → allow more amplitude.
    # Use quantized thresholds for consistency
    amp_per_slot = []
    for (f, t) in slots:
        band = band_idx_f[f]
        thr = max(1e-9, float(quantized_thr_bt[band, t]))
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
        sr: int = 44100,
        n_fft: int = 882,
        hop: int = 441,
        rs_interleave: int = 4,
        per_eul_bits_target: int = 255 * 8,  # 2040 bits to match RS-coded payload (255 bytes)
        base_symbol_amp: float = 0.1,
        amp_safety: float = 1.0,
        thr_step: float = 1e-6,  # Quantization step for thresholds
        use_anchor_seeding: bool = True,  # Use deterministic anchor-based seeding
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
        self.thr_step = thr_step
        self.use_anchor_seeding = use_anchor_seeding

    def _resample_to_target_sr(self, x_wave: torch.Tensor, orig_sr: Optional[int] = None) -> torch.Tensor:
        """Resample audio to target sample rate if needed."""
        # If original sample rate is provided and it's already 44100, return as-is
        if orig_sr is not None and orig_sr == self.sr:
            return x_wave
            
        # If no original sample rate provided, try to infer from length
        # This is a fallback - ideally the caller should provide orig_sr
        if orig_sr is None:
            current_length = x_wave.shape[-1]
            # Heuristic: if length is close to 44100, assume it's already at 44100 Hz
            if abs(current_length - self.sr) <= 100:  # Allow some tolerance
                return x_wave
            # Otherwise, we can't reliably determine the sample rate
            # In this case, we'll assume it needs resampling
            orig_sr = current_length  # This is a rough estimate
        
        # Only resample if the original sample rate is different from target
        if orig_sr != self.sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.sr
            )
            x_wave = resampler(x_wave)
        return x_wave

    # ----- Encoder path -----
    def encode_eul(self, model: INNWatermarker, x_wave: torch.Tensor, payload_bytes: bytes, orig_sr: Optional[int] = None) -> torch.Tensor:
        """
        x_wave: [1,1,T] waveform of ~1 s @ any sample rate (will be resampled to 44100)
        payload_bytes: 191 raw bytes -> RS(255,191) -> 2040 bits target
        returns: watermarked waveform [1,1,T]
        """
        assert x_wave.shape[0] == 1, "Call EULDriver per item (B=1) for deterministic slot planning."
        
        # Resample to target sample rate if needed
        x_wave = self._resample_to_target_sr(x_wave, orig_sr)

        # RS encode + interleave -> bytes
        code = rs_encode_255_191(payload_bytes)              # 255 bytes
        code = interleave_bytes(code, self.rs_interleave)

        # bytes -> bits
        bits = []
        for b in code:
            for k in range(8):
                bits.append((b >> k) & 1)
        bits_t = torch.tensor(bits, dtype=torch.long, device=x_wave.device).unsqueeze(0)  # [1, S]

        # Analyze current EUL content and allocate slots
        X = model.stft(x_wave)  # [1,2,F,T]
        slots, amp_per_slot = allocate_slots_and_amplitudes(
            X, self.sr, self.n_fft, target_bits=min(self.per_eul_bits_target, bits_t.shape[1]),
            amp_safety=self.amp_safety,
            thr_step=self.thr_step,
            use_anchor_seeding=self.use_anchor_seeding
        )
        S = min(len(slots), bits_t.shape[1])

        # Build message spectrogram
        F_, T_ = X.shape[-2], X.shape[-1]
        M_spec = bits_to_message_spec(
            bits=bits_t[:, :S],
            slots=slots[:S],
            F=F_, T=T_,
            base_amp=self.base_symbol_amp,
            amp_per_slot=amp_per_slot[:S] if len(amp_per_slot) >= S else None
        )

        # Encode through INN
        x_wm_wave, _ = model.encode(x_wave, M_spec)
        return x_wm_wave

    # ----- Decoder path -----
    def decode_eul(self, model: INNWatermarker, x_wm_wave: torch.Tensor, expected_bytes: int = 191, slots: List[Tuple[int, int]] | None = None, orig_sr: Optional[int] = None) -> bytes:
        """
        Returns recovered raw payload bytes (should be 191 if no RS errors after decode)
        If slots is provided, use those instead of recomputing from audio.
        """
        assert x_wm_wave.shape[0] == 1, "Call EULDriver per item (B=1) for deterministic slot planning."
        
        # Resample to target sample rate if needed
        x_wm_wave = self._resample_to_target_sr(x_wm_wave, orig_sr)

        # Recover message spectrogram from watermarked audio
        M_rec = model.decode(x_wm_wave)   # [1,2,F,T]

        # Use provided slots or recompute from received audio
        if slots is None:
            Xrx = model.stft(x_wm_wave)       # [1,2,F,T]
            slots, _ = allocate_slots_and_amplitudes(
                Xrx, self.sr, self.n_fft, target_bits=self.per_eul_bits_target, 
                amp_safety=self.amp_safety,
                thr_step=self.thr_step,
                use_anchor_seeding=self.use_anchor_seeding
            )

        # Read bits at those slots
        bits_rec = message_spec_to_bits(M_rec, slots)  # [1,S]
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
        data = rs_decode_255_191(deintl)
        return data[:expected_bytes]
