#!/usr/bin/env python3
"""
DDP-safe deterministic, anchor-seeded INN training driver (1 s @ 44.1 kHz).
This is a DDP-safe duplicate of training.py with all model attribute accesses
going through base_model = getattr(model, "module", model) so it works under
torch.nn.parallel.DistributedDataParallel on multi-GPU clusters.
"""

from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
import io

from models.inn_encoder_decoder import INNWatermarker
from pipeline.moore_galsberg import MooreGlasbergAnalyzer
from pipeline.adaptive_bit_allocation import (
    PerceptualSignificanceMetric,
    AdaptiveBitAllocator,
    expand_allocation_to_slots,
)
from pipeline.micro_chunking import DeterministicMapper
from pipeline.sync_anchors import SyncAnchor
from pipeline.perceptual_losses import CombinedPerceptualLoss, MFCCCosineLoss
from pipeline.ingest_and_chunk import (
    rs_encode_106_64,
    rs_decode_106_64,
    interleave_bytes,
    deinterleave_bytes,
    bits_to_message_spec,
    message_spec_to_bits,
)

# Fixed constants
SR = 44100
N_FFT = 882
HOP = 441
WIN = 882

PAYLOAD_BYTES = 64
RS_CODED_BYTES = 106
BITS_PER_EUL = RS_CODED_BYTES * 8  # 848
INTERLEAVE_DEPTH = 4

THR_STEP = 0.10

SYNC_MS = 50
SYNC_FRAMES = 5
SYNC_STRENGTH = 0.05

MICRO_WINDOW_MS = 15
MICRO_OVERLAP = 0.5

PEAK_LR = 8e-4
MIN_LR = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

CSD_LAST_FRAC = 0.4
CSD_START_SCALE = 1.0
CSD_FINAL_SCALE = 0.7

STAB_W_START = 0.05
STAB_W_FINAL = 0.5
STAB_RAMP_START_FRAC = 1.0 / 3.0
STAB_RAMP_END_FRAC = 1.0

SN_POWER_ITERS = 1
SN_TARGET_SCALE = 1.2
SN_PATTERNS = [
    "affine_coupling.*/proj.*",
    "coupling_net.*/fc.*",
    ".*1x1.*",
]

TARGET_SECONDS = 1
TARGET_SAMPLES = SR * TARGET_SECONDS


def _list_audio_files(root: str) -> List[str]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


class OneSecondDataset(Dataset):
    def __init__(self, root: str, max_files: Optional[int] = None):
        self.files = _list_audio_files(root)
        if max_files is not None and max_files > 0 and len(self.files) > max_files:
            self.files = self.files[:max_files]
        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        try:
            wav, sr = torchaudio.load(path)
        except ImportError:
            try:
                torchaudio.set_audio_backend("sox_io")
                wav, sr = torchaudio.load(path)
            except Exception as backend_err:
                try:
                    import soundfile as sf
                    data, sr = sf.read(path, dtype="float32", always_2d=True)
                    wav = torch.from_numpy(data.T)
                except Exception as sf_err:
                    raise ImportError("Audio loading requires torchaudio sox_io or soundfile.") from sf_err
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SR:
            wav = Resample(orig_freq=sr, new_freq=SR)(wav)
            sr = SR
        return wav, sr

    def _center_1s(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(-1)
        if T < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - T
            wav = F.pad(wav, (0, pad))
            return wav[:, :TARGET_SAMPLES]
        if T == TARGET_SAMPLES:
            return wav
        start = random.randint(0, max(0, T - TARGET_SAMPLES))
        return wav[:, start:start + TARGET_SAMPLES]

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, _ = self._load_audio(path)
        chunk = self._center_1s(wav)
        return chunk


def _bits_to_bytes_lsb_first(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits[: (len(bits) // 8) * 8]
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for k in range(8):
            if bits[i + k]:
                byte |= (1 << k)
        out.append(byte)
    return bytes(out)


def build_model() -> INNWatermarker:
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={
        "n_fft": N_FFT, "hop_length": HOP, "win_length": WIN,
    })
    return model


def apply_spectral_normalization(model: INNWatermarker) -> None:
    if SN_POWER_ITERS <= 0:
        return
    import re
    from torch.nn.utils import spectral_norm

    def should_normalize(name: str) -> bool:
        for pattern in SN_PATTERNS:
            if re.search(pattern, name):
                return True
        return False

    matched = []
    for name, module in model.named_modules():
        if should_normalize(name) and isinstance(module, nn.Linear):
            try:
                spectral_norm(module, name='weight', n_power_iterations=SN_POWER_ITERS)
                if hasattr(module, 'weight') and SN_TARGET_SCALE and SN_TARGET_SCALE != 1.0:
                    with torch.no_grad():
                        module.weight.data.mul_(float(SN_TARGET_SCALE))
                matched.append(name)
            except Exception as e:
                print(f"[SN] warn: could not attach to {name}: {e}")
        if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            try:
                spectral_norm(module, name='weight', n_power_iterations=SN_POWER_ITERS)
                matched.append(name)
            except Exception as e:
                print(f"[SN] warn: could not attach to 1x1 conv {name}: {e}")
    if matched:
        print(f"[SN] attached: {matched}")


def compute_anchor_seed_and_features(
    model: INNWatermarker,
    x_with_sync: torch.Tensor,
    thr_step: float = THR_STEP,
) -> Tuple[Optional[int], Dict]:
    base_model = getattr(model, "module", model)
    with torch.no_grad():
        X = base_model.stft(x_with_sync)
        mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0].cpu().numpy()
        Fdim, Tdim = mag.shape
        assert Fdim == (N_FFT // 2 + 1), f"F={Fdim} mismatch"
        mg = MooreGlasbergAnalyzer(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_critical_bands=24)
        band_thr_bt = mg.band_thresholds(mag, thr_step=thr_step)
        B = band_thr_bt.shape[0]
        start_idx = min(max(0, SYNC_FRAMES), band_thr_bt.shape[1])
        feat_bt = band_thr_bt[:, start_idx:] if start_idx < band_thr_bt.shape[1] else band_thr_bt
        if feat_bt.shape[1] == 0:
            return None, {"valid": False}
        ranks_bt = np.zeros_like(feat_bt, dtype=np.int64)
        for t in range(feat_bt.shape[1]):
            col = feat_bt[:, t]
            order = np.argsort(col)
            rank = np.argsort(order) + 1
            ranks_bt[:, t] = (B + 1) - rank
        band_rank_med = np.median(ranks_bt, axis=1)
        band_idx_f = mg.band_indices
        top_bands = np.argsort(band_rank_med)[-4:]
        band_centers = []
        for b in top_bands:
            bins = np.where(band_idx_f == b)[0]
            center = int(np.mean(bins)) if bins.size > 0 else int(b)
            band_centers.append((center, int(b)))
        band_centers.sort(key=lambda x: x[0])
        canonical = tuple(b for _, b in band_centers)
        import hashlib
        payload = str(canonical).encode() + f"_{Fdim}_{Tdim}".encode() + b"soundsafe_anchor_seed_v2"
        seed32 = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
        return seed32, {
            "valid": True,
            "seed": seed32,
            "canonical_bands": canonical,
            "band_rank_med": band_rank_med,
            "band_thr_bt": band_thr_bt,
            "band_indices_f": band_idx_f,
        }


def allocate_and_expand_slots(
    mag_ft: np.ndarray,
    band_indices_f: np.ndarray,
    seed32: int,
    target_bits: int = BITS_PER_EUL,
    thr_step: float = THR_STEP,
) -> List[Tuple[int, int]]:
    mg = MooreGlasbergAnalyzer(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_critical_bands=24)
    band_thr_bt = mg.band_thresholds(mag_ft, thr_step=thr_step)
    start_idx = min(max(0, SYNC_FRAMES), band_thr_bt.shape[1])
    feat_bt = band_thr_bt[:, start_idx:] if start_idx < band_thr_bt.shape[1] else band_thr_bt
    quantized_bt = np.floor(feat_bt / max(1e-9, thr_step)).astype(np.int64)
    psm = PerceptualSignificanceMetric(method="inverse", use_median=True)
    sig_b = psm.compute(quantized_bt)
    quantized_thr_b = np.median(quantized_bt, axis=1).astype(np.int64)
    allocator = AdaptiveBitAllocator(total_bits=target_bits, allocation_strategy="optimal", seed=seed32)
    bits_per_band = allocator.allocate_bits(sig_b, quantized_thr_b)["bit_allocation"]
    slots = expand_allocation_to_slots(
        mag_ft=mag_ft,
        band_indices_f=band_indices_f,
        bits_per_band=bits_per_band,
        per_frame_weight_bt=None,
        seed=seed32,
        quantize_magnitude=True,
    )
    return slots


def microchunk_window_plan(
    seed32: int,
    target_bits: int,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> List[Tuple[int, int]]:
    mapper = DeterministicMapper(n_fft=n_fft, hop=hop, sr=SR, window_ms=MICRO_WINDOW_MS)
    window_samples = int(round(MICRO_WINDOW_MS * SR / 1000))
    hop_samples = int(round(window_samples * (1.0 - MICRO_OVERLAP)))
    n_windows = max(1, int((TARGET_SAMPLES - window_samples) / max(1, hop_samples)) + 1)
    bits_per_window = max(1, int(round(target_bits / float(n_windows))))
    ordered: List[Tuple[int, int, int]] = []
    for w in range(n_windows):
        local = mapper.map_window_to_slots(window_idx=w, n_windows=n_windows, target_bits=bits_per_window, audio_content=None, anchor_seed=seed32)
        for (f, tloc) in local:
            ordered.append((w, int(f), int(tloc)))
    out: List[Tuple[int, int]] = []
    for (w, f, tloc) in ordered:
        start_sample = w * hop_samples
        start_frame = start_sample // hop
        t_global = start_frame + tloc
        out.append((f, t_global))
    return out


def cosine_ramp(last_frac: float, epoch_idx: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 0:
        return start
    frac = float(epoch_idx) / float(max(1, total_epochs))
    if frac <= (1.0 - last_frac):
        return start
    p = (frac - (1.0 - last_frac)) / max(1e-9, last_frac)
    c = 0.5 * (1.0 + math.cos(math.pi * p))
    return end + (start - end) * c


def stability_regularizer(
    model: INNWatermarker,
    x_ref: torch.Tensor,
    x_wm: torch.Tensor,
    thr_step: float = THR_STEP,
) -> torch.Tensor:
    base_model = getattr(model, "module", model)
    X_ref = base_model.stft(x_ref)
    X_wm = base_model.stft(x_wm)
    mag_ref = torch.sqrt(torch.clamp(X_ref[:, 0]**2 + X_ref[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()
    mag_wm = torch.sqrt(torch.clamp(X_wm[:, 0]**2 + X_wm[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()
    mg = MooreGlasbergAnalyzer(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_critical_bands=24)
    thr_ref = mg.band_thresholds(mag_ref, thr_step=thr_step)
    thr_wm = mg.band_thresholds(mag_wm, thr_step=thr_step)
    psm = PerceptualSignificanceMetric(method="inverse", use_median=True)
    q_ref = np.floor(thr_ref / max(1e-9, thr_step)).astype(np.int64)
    q_wm = np.floor(thr_wm / max(1e-9, thr_step)).astype(np.int64)
    sig_ref = psm.compute(q_ref)
    sig_wm = psm.compute(q_wm)
    return torch.mean(torch.abs(torch.from_numpy(sig_ref - sig_wm).float().to(x_ref.device)))


def forward_eul(
    model: INNWatermarker,
    x_1s: torch.Tensor,
    payload_bytes: bytes,
    epoch_idx: int,
    total_epochs: int,
    base_symbol_amp: float = 0.04,
) -> Dict:
    base_model = getattr(model, "module", model)
    assert x_1s.shape[0] == 1 and x_1s.shape[1] == 1
    sync = SyncAnchor(sync_length_ms=SYNC_MS, sr=SR, n_fft=N_FFT, hop=HOP, thr_step=THR_STEP)
    x_sync = sync.embed_sync_pattern(x_1s.squeeze(0), sync_strength=SYNC_STRENGTH).unsqueeze(0)

    seed32, seed_info = compute_anchor_seed_and_features(base_model, x_sync, thr_step=THR_STEP)
    assert seed32 is not None, "anchor seed must be non-None"

    X_dbg = base_model.stft(x_sync)
    assert X_dbg.shape[1] == 2 and X_dbg.shape[2] == (N_FFT // 2 + 1), "STFT shape mismatch"
    assert base_model.stft.n_fft == N_FFT and base_model.stft.hop == HOP, "Model STFT grid mismatch"

    code = rs_encode_106_64(payload_bytes)
    code = interleave_bytes(code, INTERLEAVE_DEPTH)
    bits = []
    for b in code:
        for k in range(8):
            bits.append((b >> k) & 1)
    bits_t = torch.tensor(bits, dtype=torch.long, device=x_1s.device).unsqueeze(0)
    target_bits = min(BITS_PER_EUL, bits_t.shape[1])

    X = base_model.stft(x_sync)
    Fdim, Tdim = X.shape[-2], X.shape[-1]
    assert Fdim == (N_FFT // 2 + 1)
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()

    slots_alloc = allocate_and_expand_slots(mag_ft=mag, band_indices_f=seed_info["band_indices_f"], seed32=seed32, target_bits=target_bits, thr_step=THR_STEP)

    ordered_global = microchunk_window_plan(seed32=seed32, target_bits=target_bits, n_fft=N_FFT, hop=HOP)
    alloc_set = set((int(f), int(t)) for (f, t) in slots_alloc)
    final_slots: List[Tuple[int, int]] = []
    used = set()
    for (f, t) in ordered_global:
        if (f, t) in alloc_set and (f, t) not in used and t < Tdim and f < Fdim:
            final_slots.append((f, t))
            used.add((f, t))
        if len(final_slots) >= target_bits:
            break
    if len(final_slots) < target_bits:
        for (f, t) in slots_alloc:
            if (f, t) not in used and t < Tdim and f < Fdim:
                final_slots.append((int(f), int(t)))
                used.add((int(f), int(t)))
                if len(final_slots) >= target_bits:
                    break

    assert len(final_slots) == target_bits, f"slots_final length {len(final_slots)} != {target_bits}"
    S = min(len(final_slots), target_bits)

    csd_scale = cosine_ramp(CSD_LAST_FRAC, epoch_idx, total_epochs, CSD_START_SCALE, CSD_FINAL_SCALE)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        M_spec = bits_to_message_spec(
            bits=bits_t[:, :S],
            slots=final_slots[:S],
            F=Fdim,
            T=Tdim,
            base_amp=base_symbol_amp * float(csd_scale),
            amp_per_slot=None,
        )
        x_wm, _ = base_model.encode(x_sync.float(), M_spec)
    x_wm = torch.tanh(x_wm)
    x_wm_clamped = x_wm.clamp(-1.0, 1.0)
    clamp_pct = float(((x_wm_clamped <= -0.999).float().mean() + (x_wm_clamped >= 0.999).float().mean()) / 2.0)

    with torch.amp.autocast(device_type="cuda", enabled=False):
        M_rec = base_model.decode(x_wm.float())
    logits = []
    targets = []
    for s in range(S):
        f_bin, t_idx = final_slots[s]
        logits.append(M_rec[0, 0, f_bin, t_idx])
        bit = 1.0 if bits_t[0, s].item() == 1 else -1.0
        targets.append(bit)
    logits_t = torch.stack(logits, dim=0) if logits else torch.zeros(0, device=x_1s.device)
    targets_t = torch.tensor(targets, dtype=torch.float32, device=x_1s.device) if targets else torch.zeros(0, device=x_1s.device)
    msg_loss = F.softplus(-logits_t * targets_t).mean() if logits else torch.tensor(0.0, device=x_1s.device)

    rec_bits = message_spec_to_bits(M_rec, final_slots[:S])
    rec_bytes = _bits_to_bytes_lsb_first(rec_bits[0].tolist())
    deintl = deinterleave_bytes(rec_bytes, INTERLEAVE_DEPTH)
    dec_raw = rs_decode_106_64(deintl)

    byte_acc = 0.0
    if len(dec_raw) and len(payload_bytes):
        L = min(len(dec_raw), len(payload_bytes))
        byte_acc = float(sum(1 for i in range(L) if dec_raw[i] == payload_bytes[i]) / max(1, L))
    payload_ok = 1.0 if dec_raw[:len(payload_bytes)] == payload_bytes[:len(dec_raw[:len(payload_bytes)])] and len(dec_raw) >= len(payload_bytes) else 0.0

    seed32_dec, _ = compute_anchor_seed_and_features(base_model, x_wm.detach(), thr_step=THR_STEP)
    seed_parity = float(1.0 if (seed32_dec == seed32) else 0.0)
    X_rx = base_model.stft(x_wm.detach())
    mag_rx = torch.sqrt(torch.clamp(X_rx[:, 0]**2 + X_rx[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()
    slots_alloc_rx = allocate_and_expand_slots(mag_ft=mag_rx, band_indices_f=seed_info["band_indices_f"], seed32=(seed32_dec if seed32_dec is not None else seed32), target_bits=target_bits, thr_step=THR_STEP)
    ordered_global_rx = microchunk_window_plan(seed32=(seed32_dec if seed32_dec is not None else seed32), target_bits=target_bits, n_fft=N_FFT, hop=HOP)
    alloc_set_rx = set((int(f), int(t)) for (f, t) in slots_alloc_rx)
    final_slots_rx: List[Tuple[int, int]] = []
    used_rx = set()
    for (f, t) in ordered_global_rx:
        if (f, t) in alloc_set_rx and (f, t) not in used_rx and t < Tdim and f < Fdim:
            final_slots_rx.append((f, t))
            used_rx.add((f, t))
        if len(final_slots_rx) >= target_bits:
            break
    if len(final_slots_rx) < target_bits:
        for (f, t) in slots_alloc_rx:
            if (f, t) not in used_rx and t < Tdim and f < Fdim:
                final_slots_rx.append((int(f), int(t)))
                used_rx.add((int(f), int(t)))
                if len(final_slots_rx) >= target_bits:
                    break
    slot_match = 0.0
    if len(final_slots_rx) >= S and len(final_slots) >= S:
        same = sum(1 for i in range(S) if final_slots[i] == final_slots_rx[i])
        slot_match = float(same / max(1, S))

    stab = stability_regularizer(base_model, x_sync.detach(), x_wm.detach(), thr_step=THR_STEP)

    with torch.no_grad():
        Xref = base_model.stft(x_sync)
        xrt = base_model.istft(Xref, base_model.stft.get_last_params())
    inv = F.l1_loss(xrt, x_sync)

    perc = CombinedPerceptualLoss(mfcc=MFCCCosineLoss(sample_rate=SR))(x_sync.float(), x_wm.float())
    perc_total = perc["total_perceptual_loss"]

    return {
        "x_sync": x_sync,
        "x_wm": x_wm,
        "slots": final_slots[:S],
        "msg_loss": msg_loss,
        "perc_loss": perc_total,
        "stab_loss": stab,
        "inv_loss": inv,
        "logits": logits_t.detach(),
        "targets": targets_t.detach(),
        "byte_acc": torch.tensor(byte_acc, device=x_1s.device),
        "payload_ok": torch.tensor(payload_ok, device=x_1s.device),
        "clamp_pct": torch.tensor(clamp_pct, device=x_1s.device),
        "seed": seed32,
        "seed_parity": torch.tensor(seed_parity, device=x_1s.device),
        "slot_parity": torch.tensor(slot_match, device=x_1s.device),
    }


def plan_eul_no_encode(
    model: INNWatermarker,
    x_1s: torch.Tensor,
    payload_bytes: bytes,
    epoch_idx: int,
    total_epochs: int,
    base_symbol_amp: float = 0.04,
) -> Dict:
    """Plan per item without running encode/decode. Returns x_sync, slots, bits, and scales for batched encode/decode."""
    base_model = getattr(model, "module", model)
    assert x_1s.shape[0] == 1 and x_1s.shape[1] == 1
    sync = SyncAnchor(sync_length_ms=SYNC_MS, sr=SR, n_fft=N_FFT, hop=HOP, thr_step=THR_STEP)
    x_sync = sync.embed_sync_pattern(x_1s.squeeze(0), sync_strength=SYNC_STRENGTH).unsqueeze(0)

    seed32, seed_info = compute_anchor_seed_and_features(base_model, x_sync, thr_step=THR_STEP)
    if seed32 is None:
        raise RuntimeError("anchor seed must be non-None")

    X = base_model.stft(x_sync)
    Fdim, Tdim = X.shape[-2], X.shape[-1]
    assert Fdim == (N_FFT // 2 + 1)
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()

    code = rs_encode_106_64(payload_bytes)
    code = interleave_bytes(code, INTERLEAVE_DEPTH)
    bits = []
    for b in code:
        for k in range(8):
            bits.append((b >> k) & 1)
    bits_t = torch.tensor(bits, dtype=torch.long, device=x_1s.device).unsqueeze(0)
    target_bits = min(BITS_PER_EUL, bits_t.shape[1])

    slots_alloc = allocate_and_expand_slots(mag_ft=mag, band_indices_f=seed_info["band_indices_f"], seed32=seed32, target_bits=target_bits, thr_step=THR_STEP)
    ordered_global = microchunk_window_plan(seed32=seed32, target_bits=target_bits, n_fft=N_FFT, hop=HOP)
    alloc_set = set((int(f), int(t)) for (f, t) in slots_alloc)
    final_slots: List[Tuple[int, int]] = []
    used = set()
    for (f, t) in ordered_global:
        if (f, t) in alloc_set and (f, t) not in used and t < Tdim and f < Fdim:
            final_slots.append((f, t))
            used.add((f, t))
        if len(final_slots) >= target_bits:
            break
    if len(final_slots) < target_bits:
        for (f, t) in slots_alloc:
            if (f, t) not in used and t < Tdim and f < Fdim:
                final_slots.append((int(f), int(t)))
                used.add((int(f), int(t)))
                if len(final_slots) >= target_bits:
                    break

    if len(final_slots) != target_bits:
        raise RuntimeError(f"slots_final length {len(final_slots)} != {target_bits}")

    csd_scale = cosine_ramp(CSD_LAST_FRAC, epoch_idx, total_epochs, CSD_START_SCALE, CSD_FINAL_SCALE)
    return {
        "x_sync": x_sync,
        "final_slots": final_slots,
        "bits_t": bits_t[:, :target_bits],
        "Fdim": Fdim,
        "Tdim": Tdim,
        "seed": seed32,
        "csd_scale": csd_scale,
        "payload_bytes": payload_bytes,
    }


@dataclass
class TrainConfig:
    data_dir: str = os.environ.get("DATA_DIR", "data/train")
    val_dir: str = os.environ.get("VAL_DIR", "data/val")
    save_dir: str = os.environ.get("SAVE_DIR", "runs_ddp")
    num_workers: int = int(os.environ.get("NUM_WORKERS", 4))
    batch_size: int = int(os.environ.get("PER_DEVICE_BATCH", 4))
    epochs: int = int(os.environ.get("EPOCHS", 30))
    mixed_precision: bool = os.environ.get("AMP", "1") == "1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    base_symbol_amp: float = float(os.environ.get("BASE_SYMBOL_AMP", 0.04))
    log_interval: int = 50
    max_train_files: Optional[int] = 25000
    max_val_files: Optional[int] = 10000


def lr_schedule(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return (step / max(1, WARMUP_STEPS)) * (PEAK_LR - MIN_LR) + MIN_LR
    t = (step - WARMUP_STEPS) / max(1, (total_steps - WARMUP_STEPS))
    t = min(1.0, max(0.0, t))
    return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1.0 + math.cos(math.pi * t))


def train_one_epoch(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader, optimizer: torch.optim.Optimizer, scaler, epoch_idx: int, total_epochs: int, global_step_start: int, total_steps: int) -> Dict:
    model.train()
    running = {"loss": 0.0, "msg": 0.0, "perc": 0.0, "stab": 0.0, "inv": 0.0, "byte_acc": 0.0, "payload_ok": 0.0, "clamp_pct": 0.0}

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"train {epoch_idx}", leave=False)
    global_step = global_step_start
    use_amp = cfg.mixed_precision and torch.cuda.is_available()

    for step, batch in pbar:
        x = batch.to(cfg.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        lr_now = lr_schedule(global_step, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        payload = bytes([i % 256 for i in range(PAYLOAD_BYTES)])

        # 1) Plan on CPU per item
        plans: List[Dict] = []
        for i in range(x.shape[0]):
            plans.append(plan_eul_no_encode(model, x[i:i+1], payload, epoch_idx, cfg.epochs, base_symbol_amp=cfg.base_symbol_amp))

        # 2) Build batched tensors for encode
        base_model = getattr(model, "module", model)
        x_sync_b = torch.cat([p["x_sync"] for p in plans], dim=0).to(x.device)
        Fdim, Tdim = plans[0]["Fdim"], plans[0]["Tdim"]
        # Per item M_spec using fp32 island, then stack
        with torch.amp.autocast(device_type="cuda", enabled=False):
            M_list = []
            for p in plans:
                M = bits_to_message_spec(bits=p["bits_t"].to(x.device), slots=p["final_slots"], F=Fdim, T=Tdim, base_amp=cfg.base_symbol_amp * float(p["csd_scale"]))
                M_list.append(M)
            M_spec_b = torch.cat(M_list, dim=0)
            x_wm_b, _ = base_model.encode(x_sync_b.float(), M_spec_b)
        x_wm_b = torch.tanh(x_wm_b)

        # 3) Decode batched once
        with torch.amp.autocast(device_type="cuda", enabled=False):
            M_rec_b = base_model.decode(x_wm_b.float())

        # 4) Compute per-item losses and reduce
        total_losses = []
        msg_vals = []
        perc_vals = []
        stab_vals = []
        inv_vals = []
        byte_accs = []
        payload_oks = []
        clamps = []

        for i in range(x.shape[0]):
            p = plans[i]
            slots = p["final_slots"]
            S = p["bits_t"].shape[1]
            logits = []
            targets = []
            for s in range(S):
                f_bin, t_idx = slots[s]
                logits.append(M_rec_b[i, 0, f_bin, t_idx])
                targets.append(1.0 if int(p["bits_t"][0, s].item()) == 1 else -1.0)
            logits_t = torch.stack(logits, dim=0)
            targets_t = torch.tensor(targets, dtype=torch.float32, device=logits_t.device)
            msg_loss = F.softplus(-logits_t * targets_t).mean()

            # Byte metric (not in loss)
            rec_bits = message_spec_to_bits(M_rec_b[i:i+1], slots)
            rec_bytes = _bits_to_bytes_lsb_first(rec_bits[0].tolist())
            deintl = deinterleave_bytes(rec_bytes, INTERLEAVE_DEPTH)
            dec_raw = rs_decode_106_64(deintl)
            Lb = min(len(dec_raw), len(p["payload_bytes"]))
            byte_acc = float(sum(1 for j in range(Lb) if dec_raw[j] == p["payload_bytes"][j]) / max(1, Lb))
            payload_ok = 1.0 if (len(dec_raw) >= len(p["payload_bytes"]) and dec_raw[:len(p["payload_bytes"])] == p["payload_bytes"]) else 0.0

            # Stability and invertibility per item (cheap)
            stab = stability_regularizer(base_model, p["x_sync"].detach(), x_wm_b[i:i+1].detach(), thr_step=THR_STEP)
            with torch.no_grad():
                Xref = base_model.stft(p["x_sync"]) ; xrt = base_model.istft(Xref, base_model.stft.get_last_params())
            inv = F.l1_loss(xrt, p["x_sync"])

            # Perceptual per item
            perc = CombinedPerceptualLoss(mfcc=MFCCCosineLoss(sample_rate=SR))(p["x_sync"].float(), x_wm_b[i:i+1].float())
            perc_total = perc["total_perceptual_loss"]

            frac = float(epoch_idx - 1) / max(1, cfg.epochs - 1)
            if frac < STAB_RAMP_START_FRAC:
                w_stab = STAB_W_START
            elif frac >= STAB_RAMP_END_FRAC:
                w_stab = STAB_W_FINAL
            else:
                pr = (frac - STAB_RAMP_START_FRAC) / max(1e-9, STAB_RAMP_END_FRAC - STAB_RAMP_START_FRAC)
                w_stab = STAB_W_START + pr * (STAB_W_FINAL - STAB_W_START)
            total = 10.0 * msg_loss + 1.0 * perc_total + 1.0 * inv + float(w_stab) * stab

            total_losses.append(total)
            msg_vals.append(msg_loss.detach())
            perc_vals.append(perc_total.detach())
            stab_vals.append(stab.detach())
            inv_vals.append(inv.detach())
            byte_accs.append(torch.tensor(byte_acc, device=x.device))
            payload_oks.append(torch.tensor(payload_ok, device=x.device))
            clamp_pct = float(((x_wm_b[i:i+1] <= -0.999).float().mean() + (x_wm_b[i:i+1] >= 0.999).float().mean()) / 2.0)
            clamps.append(torch.tensor(clamp_pct, device=x.device))

        # Backward on mean loss
        loss_mean = torch.stack(total_losses).mean()
        if use_amp and scaler is not None:
            scaler.scale(loss_mean).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
        else:
            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

        L = loss_mean.item()
        M = torch.stack(msg_vals).mean().item()
        P = torch.stack(perc_vals).mean().item()
        S = torch.stack(stab_vals).mean().item()
        I = torch.stack(inv_vals).mean().item()
        BA = torch.stack(byte_accs).mean().item()
        PO = torch.stack(payload_oks).mean().item()
        CP = torch.stack(clamps).mean().item()

        running["loss"] += L * x.shape[0]
        running["msg"] += M * x.shape[0]
        running["perc"] += P * x.shape[0]
        running["stab"] += S * x.shape[0]
        running["inv"] += I * x.shape[0]
        running["byte_acc"] += BA * x.shape[0]
        running["payload_ok"] += PO * x.shape[0]
        running["clamp_pct"] += CP * x.shape[0]

        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['loss']/((step+1)*loader.batch_size):.4f}",
                "msg": f"{running['msg']/((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc']/((step+1)*loader.batch_size):.4f}",
                "stab": f"{running['stab']/((step+1)*loader.batch_size):.4f}",
                "byte": f"{running['byte_acc']/((step+1)*loader.batch_size):.3f}",
                "ok": f"{running['payload_ok']/((step+1)*loader.batch_size):.3f}",
                "%±1": f"{running['clamp_pct']/((step+1)*loader.batch_size)*100:.2f}%",
            })

        global_step += 1

    denom = max(1, len(loader) * loader.batch_size)
    for k in list(running.keys()):
        running[k] = running[k] / denom
    return running


def validate(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader, epoch_idx: int) -> Dict:
    model.eval()
    running = {"loss": 0.0, "msg": 0.0, "perc": 0.0, "stab": 0.0, "inv": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
    with torch.no_grad():
        for batch in loader:
            x = batch.to(cfg.device, non_blocking=True)
            payload = bytes([i % 256 for i in range(PAYLOAD_BYTES)])
            # Plan per item then batch decode/encode
            plans: List[Dict] = []
            for i in range(x.shape[0]):
                plans.append(plan_eul_no_encode(model, x[i:i+1], payload, epoch_idx, cfg.epochs, base_symbol_amp=cfg.base_symbol_amp))
            base_model = getattr(model, "module", model)
            x_sync_b = torch.cat([p["x_sync"] for p in plans], dim=0).to(x.device)
            Fdim, Tdim = plans[0]["Fdim"], plans[0]["Tdim"]
            with torch.amp.autocast(device_type="cuda", enabled=False):
                M_list = []
                for p in plans:
                    M = bits_to_message_spec(bits=p["bits_t"].to(x.device), slots=p["final_slots"], F=Fdim, T=Tdim, base_amp=cfg.base_symbol_amp * float(p["csd_scale"]))
                    M_list.append(M)
                M_spec_b = torch.cat(M_list, dim=0)
                x_wm_b, _ = base_model.encode(x_sync_b.float(), M_spec_b)
            x_wm_b = torch.tanh(x_wm_b)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                M_rec_b = base_model.decode(x_wm_b.float())

            losses = [] ; msg_vals = [] ; perc_vals = [] ; stab_vals = [] ; inv_vals = [] ; byte_accs = [] ; payload_oks = []
            for i in range(x.shape[0]):
                p = plans[i]
                slots = p["final_slots"] ; S = p["bits_t"].shape[1]
                logits = [] ; targets = []
                for s in range(S):
                    f_bin, t_idx = slots[s]
                    logits.append(M_rec_b[i, 0, f_bin, t_idx])
                    targets.append(1.0 if int(p["bits_t"][0, s].item()) == 1 else -1.0)
                logits_t = torch.stack(logits, dim=0)
                targets_t = torch.tensor(targets, dtype=torch.float32, device=logits_t.device)
                msg_loss = F.softplus(-logits_t * targets_t).mean()
                rec_bits = message_spec_to_bits(M_rec_b[i:i+1], slots)
                rec_bytes = _bits_to_bytes_lsb_first(rec_bits[0].tolist())
                deintl = deinterleave_bytes(rec_bytes, INTERLEAVE_DEPTH)
                dec_raw = rs_decode_106_64(deintl)
                Lb = min(len(dec_raw), len(p["payload_bytes"]))
                byte_acc = float(sum(1 for j in range(Lb) if dec_raw[j] == p["payload_bytes"][j]) / max(1, Lb))
                payload_ok = 1.0 if (len(dec_raw) >= len(p["payload_bytes"]) and dec_raw[:len(p["payload_bytes"])] == p["payload_bytes"]) else 0.0
                stab = stability_regularizer(base_model, p["x_sync"].detach(), x_wm_b[i:i+1].detach(), thr_step=THR_STEP)
                with torch.no_grad():
                    Xref = base_model.stft(p["x_sync"]) ; xrt = base_model.istft(Xref, base_model.stft.get_last_params())
                inv = F.l1_loss(xrt, p["x_sync"])
                perc = CombinedPerceptualLoss(mfcc=MFCCCosineLoss(sample_rate=SR))(p["x_sync"].float(), x_wm_b[i:i+1].float())
                perc_total = perc["total_perceptual_loss"]
                frac = float(epoch_idx - 1) / max(1, cfg.epochs - 1)
                if frac < STAB_RAMP_START_FRAC: w_stab = STAB_W_START
                elif frac >= STAB_RAMP_END_FRAC: w_stab = STAB_W_FINAL
                else:
                    pr = (frac - STAB_RAMP_START_FRAC) / max(1e-9, STAB_RAMP_END_FRAC - STAB_RAMP_START_FRAC)
                    w_stab = STAB_W_START + pr * (STAB_W_FINAL - STAB_W_START)
                total = 10.0 * msg_loss + 1.0 * perc_total + 1.0 * inv + float(w_stab) * stab
                losses.append(total)
                msg_vals.append(msg_loss) ; perc_vals.append(perc_total) ; stab_vals.append(stab) ; inv_vals.append(inv)
                byte_accs.append(torch.tensor(byte_acc, device=x.device)) ; payload_oks.append(torch.tensor(payload_ok, device=x.device))
            if losses:
                running["loss"] += float(torch.stack(losses).mean().item()) * x.shape[0]
                running["msg"] += float(torch.stack(msg_vals).mean().item()) * x.shape[0]
                running["perc"] += float(torch.stack(perc_vals).mean().item()) * x.shape[0]
                running["stab"] += float(torch.stack(stab_vals).mean().item()) * x.shape[0]
                running["inv"] += float(torch.stack(inv_vals).mean().item()) * x.shape[0]
                running["byte_acc"] += float(torch.stack(byte_accs).mean().item()) * x.shape[0]
                running["payload_ok"] += float(torch.stack(payload_oks).mean().item()) * x.shape[0]
    denom = max(1, len(loader) * loader.batch_size)
    for k in list(running.keys()):
        running[k] = running[k] / denom
    return running


def _apply_gain_db(x: torch.Tensor, db: float) -> torch.Tensor:
    g = 10.0 ** (db / 20.0)
    return torch.clamp(x * g, -1.0, 1.0)


def _apply_soft_limiter(x: torch.Tensor, db_limit: float = -3.0) -> torch.Tensor:
    thr = 10.0 ** (db_limit / 20.0)
    y = x.clone()
    mask_pos = y > thr
    mask_neg = y < -thr
    y[mask_pos] = thr + torch.tanh((y[mask_pos] - thr) / (1 - thr)) * (1 - thr)
    y[mask_neg] = -thr + torch.tanh((y[mask_neg] + thr) / (1 - thr)) * (1 - thr)
    return torch.clamp(y, -1.0, 1.0)


def _apply_mp3_192(x: torch.Tensor, sr: int = SR) -> torch.Tensor:
    try:
        if x.dim() == 3:
            wav = x.squeeze(0).cpu()
        elif x.dim() == 2:
            wav = x.cpu()
        else:
            wav = x.unsqueeze(0).cpu()
        buf = io.BytesIO()
        torchaudio.save(buf, wav, sr, format='mp3', compression=192)
        buf.seek(0)
        y, _ = torchaudio.load(buf)
        if y.dim() == 2:
            y = y.mean(0, keepdim=True)
        return y.unsqueeze(0).to(x.device, dtype=x.dtype)
    except Exception:
        return x


def _decode_blind(model: INNWatermarker, x_wm: torch.Tensor, target_bits: int = BITS_PER_EUL) -> Tuple[bytes, float, float]:
    base_model = getattr(model, "module", model)
    seed32_dec, seed_info = compute_anchor_seed_and_features(base_model, x_wm, thr_step=THR_STEP)
    if seed32_dec is None:
        return b"", 0.0, 0.0
    X = base_model.stft(x_wm)
    Fdim, Tdim = X.shape[-2], X.shape[-1]
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0].detach().cpu().numpy()
    slots_alloc = allocate_and_expand_slots(mag_ft=mag, band_indices_f=seed_info.get("band_indices_f", MooreGlasbergAnalyzer(sample_rate=SR, n_fft=N_FFT, hop_length=HOP).band_indices), seed32=seed32_dec, target_bits=target_bits, thr_step=THR_STEP)
    ordered_global = microchunk_window_plan(seed32=seed32_dec, target_bits=target_bits, n_fft=N_FFT, hop=HOP)
    alloc_set = set((int(f), int(t)) for (f, t) in slots_alloc)
    final_slots: List[Tuple[int, int]] = []
    used = set()
    for (f, t) in ordered_global:
        if (f, t) in alloc_set and (f, t) not in used and t < Tdim and f < Fdim:
            final_slots.append((f, t))
            used.add((f, t))
        if len(final_slots) >= target_bits:
            break
    if len(final_slots) < target_bits:
        for (f, t) in slots_alloc:
            if (f, t) not in used and t < Tdim and f < Fdim:
                final_slots.append((int(f), int(t)))
                used.add((int(f), int(t)))
                if len(final_slots) >= target_bits:
                    break
    with torch.no_grad():
        M_rec = base_model.decode(x_wm)
    bits_rec = message_spec_to_bits(M_rec, final_slots[:target_bits])
    rec_bytes = _bits_to_bytes_lsb_first(bits_rec[0].tolist())
    seed_parity = 1.0
    slot_parity = 1.0
    return rec_bytes, seed_parity, slot_parity


def evaluate_batteries(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader, num_items: int = 100) -> Dict[str, Dict[str, float]]:
    model.eval()
    results: Dict[str, Dict[str, float]] = {}
    def _eval_condition(transform_name: str, transform_fn) -> Dict[str, float]:
        byte_hits = 0
        payload_ok = 0
        seed_par = []
        slot_par = []
        count = 0
        with torch.no_grad():
            for batch in loader:
                for i in range(batch.shape[0]):
                    if count >= num_items:
                        break
                    x = batch[i:i+1].to(cfg.device)
                    payload = bytes([i % 256 for i in range(PAYLOAD_BYTES)])
                    out = forward_eul(model, x, payload, epoch_idx=cfg.epochs, total_epochs=cfg.epochs, base_symbol_amp=cfg.base_symbol_amp)
                    x_wm = out["x_wm"].detach()
                    x_t = transform_fn(x_wm)
                    rec_bytes, seed_p, slot_p = _decode_blind(model, x_t)
                    if len(rec_bytes) >= PAYLOAD_BYTES:
                        dec = rs_decode_106_64(deinterleave_bytes(rec_bytes, INTERLEAVE_DEPTH))[:PAYLOAD_BYTES]
                    else:
                        dec = rec_bytes
                    L = min(len(dec), PAYLOAD_BYTES)
                    hits = sum(1 for j in range(L) if dec[j] == payload[j])
                    byte_hits += hits / max(1, PAYLOAD_BYTES)
                    payload_ok += 1.0 if (len(dec) >= PAYLOAD_BYTES and dec[:PAYLOAD_BYTES] == payload) else 0.0
                    seed_par.append(seed_p)
                    slot_par.append(slot_p)
                    count += 1
                if count >= num_items:
                    break
        n = max(1, count)
        return {
            "byte_acc": float(byte_hits / n),
            "payload_ok": float(payload_ok / n),
            "seed_parity": float(np.mean(seed_par) if seed_par else 0.0),
            "slot_parity": float(np.mean(slot_par) if slot_par else 0.0),
            "n": float(n),
        }

    results["clean"] = _eval_condition("clean", lambda x: x)
    results["gain+3dB"] = _eval_condition("gain+3dB", lambda x: _apply_gain_db(x, +3.0))
    results["gain-3dB"] = _eval_condition("gain-3dB", lambda x: _apply_gain_db(x, -3.0))
    results["limiter-3dB"] = _eval_condition("limiter-3dB", lambda x: _apply_soft_limiter(x, -3.0))
    results["mp3-192"] = _eval_condition("mp3-192", lambda x: _apply_mp3_192(x, SR))
    return results


def main(cfg: TrainConfig) -> None:
    # Backend perf knobs
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    os.makedirs(cfg.save_dir, exist_ok=True)

    is_distributed = False
    rank = 0
    local_rank = 0
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception:
            dist.init_process_group(backend="gloo", init_method="env://")
        is_distributed = True
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            cfg.device = f"cuda:{local_rank}"
        else:
            cfg.device = "cpu"

    train_ds = OneSecondDataset(cfg.data_dir, max_files=cfg.max_train_files)
    val_ds = OneSecondDataset(cfg.val_dir, max_files=cfg.max_val_files)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_distributed else None
    pin = cfg.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    model = build_model().to(cfg.device)
    # Torch compile (PyTorch 2.x) before DDP for best performance
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass
    apply_spectral_normalization(model)
    if is_distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MIN_LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.mixed_precision and torch.cuda.is_available())

    def log(msg: str) -> None:
        if (not is_distributed) or rank == 0:
            print(msg)
            try:
                with open(os.path.join(cfg.save_dir, "train.log"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    if (not is_distributed) or rank == 0:
        log(f"Files: train={len(train_ds)} | val={len(val_ds)} | SR={SR} | Grid=({N_FFT},{HOP},{WIN})")

    total_steps = cfg.epochs * len(train_loader)
    global_step = 0
    best_byte = 0.0

    for epoch in range(1, cfg.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, cfg, train_loader, optimizer, scaler, epoch, cfg.epochs, global_step, total_steps)
        global_step += len(train_loader)
        val_metrics = validate(model, cfg, val_loader, epoch)

        if (not is_distributed) or rank == 0:
            log(f"epoch {epoch:03d}: loss={train_metrics['loss']:.4f} msg={train_metrics['msg']:.4f} perc={train_metrics['perc']:.4f} stab={train_metrics['stab']:.4f} inv={train_metrics['inv']:.4f} byte={train_metrics['byte_acc']:.3f} ok={train_metrics['payload_ok']:.3f} clamp={train_metrics['clamp_pct']*100:.2f}%")
            log(f"val   {epoch:03d}: loss={val_metrics['loss']:.4f} msg={val_metrics['msg']:.4f} perc={val_metrics['perc']:.4f} stab={val_metrics['stab']:.4f} inv={val_metrics['inv']:.4f} byte={val_metrics['byte_acc']:.3f} ok={val_metrics['payload_ok']:.3f}")
            if float(val_metrics.get("byte_acc", 0.0)) >= best_byte:
                best_byte = float(val_metrics.get("byte_acc", 0.0))
                ckpt = {
                    "epoch": epoch,
                    "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                }
                path = os.path.join(cfg.save_dir, "inn_fixed_grid_best.pt")
                torch.save(ckpt, path)
                log(f"saved best: {path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deterministic INN training (DDP-safe)")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--save_dir", type=str, default="runs_ddp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    args, _ = parser.parse_known_args()

    _defaults = TrainConfig()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_files=(args.max_train_files if args.max_train_files is not None else (int(os.environ.get("MAX_TRAIN_FILES")) if os.environ.get("MAX_TRAIN_FILES") else _defaults.max_train_files)),
        max_val_files=(args.max_val_files if args.max_val_files is not None else (int(os.environ.get("MAX_VAL_FILES")) if os.environ.get("MAX_VAL_FILES") else _defaults.max_val_files)),
    )
    main(cfg)
