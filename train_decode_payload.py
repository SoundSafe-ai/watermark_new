#!/usr/bin/env python3
"""
Decode-focused training script: optimize watermark embedding so the payload is
recovered correctly after encode->decode. Tracks BER in addition to losses.

Dataset layout:
  data/train  -> training audio files
  data/val    -> validation audio files

Loss (objective):
  obj = w_bits * L_bits_ce  +  w_mse * L_bits_mse  +  w_perc * L_perc
where
  L_bits_ce  = BCEWithLogits over soft symbols at chosen (f,t) slots
  L_bits_mse = MSE between recovered symbol and target in {-1,+1} (optional)
  L_perc     = Combined perceptual loss (MRSTFT + MFCC + SNR) with small weight

We use the psychoacoustic allocator to choose slots and amplitudes per item.
"""

from __future__ import annotations
import os
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchaudio.functional as AF
from datetime import datetime

from models.inn_encoder_decoder import INNWatermarker, STFT
from pipeline.perceptual_losses import CombinedPerceptualLoss
from pipeline.ingest_and_chunk import (
    allocate_slots_and_amplitudes,
    bits_to_message_spec,
    message_spec_to_bits,
)
from pipeline.psychoacoustic import mel_proxy_threshold
from pipeline.ingest_and_chunk import rs_encode_167_125, rs_decode_167_125, interleave_bytes, deinterleave_bytes
from pipeline.payload_codec import pack_fields

# Quiet known warnings
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)


TARGET_SR = 22050
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)


def list_audio_files(root: str) -> List[str]:
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn.lower())[1] in exts:
                files.append(os.path.join(dirpath, fn))
    return files


class AudioChunkDataset(Dataset):
    def __init__(self, root: str, target_sr: int = TARGET_SR, gpu_resample: bool = False):
        self.files = list_audio_files(root)
        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {root}")
        self.target_sr = target_sr
        self.gpu_resample = gpu_resample

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # If we plan to resample on GPU, skip CPU resample
        if not self.gpu_resample and sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
            sr = self.target_sr
        wav = wav / (wav.abs().max() + 1e-9)
        return wav, sr  # [1,T], sr

    def _random_1s_chunk(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(-1)
        if T < CHUNK_SAMPLES:
            pad = CHUNK_SAMPLES - T
            wav = F.pad(wav, (0, pad))
            return wav[:, :CHUNK_SAMPLES]
        start = random.randint(0, max(0, T - CHUNK_SAMPLES))
        return wav[:, start:start + CHUNK_SAMPLES]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path = self.files[idx]
        wav, sr = self._load_audio(path)
        return self._random_1s_chunk(wav), sr, path


def match_length(y: torch.Tensor, target_T: int) -> torch.Tensor:
    T = y.size(-1)
    if T == target_T:
        return y
    if T > target_T:
        return y[..., :target_T]
    return F.pad(y, (0, target_T - T))


@dataclass
class TrainConfig:
    data_dir: str = "data/train"
    val_dir: str = "data/val"
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    save_dir: str = "decode_payload"
    log_interval: int = 50
    n_fft: int = 1024
    hop: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_resample: bool = True
    # Initialize from a prior imperceptibility checkpoint (stage-1)
    init_from: str | None = "inn_decode_best.pt"  # Try training from scratch for decode
    # Limit number of files used (None uses all)
    train_max_files: int | None = None
    val_max_files: int | None = None
    file_seed: int = 42
    # Loss weights
    w_bits: float = 1.1
    w_mse: float = 0.25
    w_perc: float = 0.009
    # Symbol settings
    base_symbol_amp: float = 0.09  # Will be annealed during training
    target_bits: int = 1344  # Fixed number of bits; no curriculum
    # RS and interleaving settings
    use_rs_interleave: bool = True
    rs_payload_bytes: int = 125  # Raw payload size before RS encoding
    rs_interleave_depth: int = 4  # Interleaving depth
    # RS warmup controls
    rs_warmup_epochs: int = 3
    rs_enable_ber_threshold: float = 0.35
    # Planner selection: "gpu" (fast mel-proxy) or "mg" (Moore–Glasberg)
    planner: str = "gpu"
    # Cache planning per file; re-plan every N epochs
    replan_every: int = 3
    # File logging
    log_file: str = "train_log.txt"
    # Fixed payload controls
    use_fixed_payload: bool = False
    payload_text: str = "Title:You'reasurvivor,Perf:NikeshShah,ISRC:123456789101,ISWC:123456789012,length:04:20,date:01/01/2025,label:warnerbros"


def make_bits(batch_size: int, S: int, device) -> torch.Tensor:
    return torch.randint(low=0, high=2, size=(batch_size, S), device=device, dtype=torch.long)


def make_rs_payload_bytes(batch_size: int, payload_bytes: int, device) -> list[bytes]:
    """Generate random payload bytes for RS encoding."""
    payloads = []
    for _ in range(batch_size):
        payload = bytes([random.randint(0, 255) for _ in range(payload_bytes)])
        payloads.append(payload)
    return payloads


def encode_payload_with_rs(payload_bytes: bytes, interleave_depth: int) -> torch.Tensor:
    """Encode payload with RS(167,125) and interleaving, return as bit tensor."""
    # RS encode: 125 bytes -> 167 bytes
    rs_encoded = rs_encode_167_125(payload_bytes)

    # Interleave
    interleaved = interleave_bytes(rs_encoded, interleave_depth)

    # Convert to bits
    bits = []
    for b in interleaved:
        for k in range(8):
            bits.append((b >> k) & 1)

    return torch.tensor(bits, dtype=torch.long)


def decode_payload_with_rs(bits: torch.Tensor, interleave_depth: int, expected_payload_bytes: int) -> bytes:
    """Decode bit tensor with deinterleaving and RS decoding."""
    # Convert bits to bytes
    by = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        b = 0
        for k in range(8):
            b |= (int(bits[i + k]) & 1) << k
        by.append(b)

    # Deinterleave
    deinterleaved = deinterleave_bytes(bytes(by), interleave_depth)

    # RS decode
    try:
        decoded = rs_decode_167_125(deinterleaved)
        return decoded[:expected_payload_bytes]
    except:
        # Return zeros on decode failure
        return bytes(expected_payload_bytes)


def _parse_kv_csv_to_fields(text: str) -> dict:
    fields: dict = {}
    for seg in text.split(","):
        seg = seg.strip()
        if not seg:
            continue
        # split on first '-' or ':' to allow values like '4:20'
        pos_dash = seg.find("-")
        pos_colon = seg.find(":")
        # choose the earliest non-negative position
        positions = [p for p in [pos_dash, pos_colon] if p >= 0]
        if not positions:
            continue
        p = min(positions)
        key = seg[:p].strip()
        val = seg[p+1:].strip()
        if key:
            fields[key] = val
    return fields


def build_fixed_payload_bytes(cfg: TrainConfig) -> bytes:
    fields = _parse_kv_csv_to_fields(cfg.payload_text)
    raw = pack_fields(fields)
    # Pad/truncate to RS input length
    if len(raw) >= cfg.rs_payload_bytes:
        return raw[:cfg.rs_payload_bytes]
    return raw + bytes(cfg.rs_payload_bytes - len(raw))


def symbols_from_bits(bits: torch.Tensor, amp: float) -> torch.Tensor:
    # 0 -> -amp, 1 -> +amp
    return (bits * 2 - 1).float() * amp


def gather_slots(M: torch.Tensor, slots: List[Tuple[int, int]]) -> torch.Tensor:
    # M: [B,2,F,T]; real channel = M[:,0]
    B = M.size(0)
    vals = []
    for (f, t) in slots:
        vals.append(M[:, 0, f, t])  # [B]
    return torch.stack(vals, dim=1) if len(vals) > 0 else torch.zeros(B, 0, device=M.device)


@torch.no_grad()
def plan_slots_and_amp(model: INNWatermarker, x_wave: torch.Tensor, sr: int, n_fft: int, target_bits: int, base_amp: float):
    model = getattr(model, "module", model)
    # Plan on clean audio to avoid label leakage
    X = model.stft(x_wave)  # [B,2,F,T]
    assert X.size(0) == 1, "Call per item for deterministic plan"
    slots, amp_per_slot = allocate_slots_and_amplitudes(X, sr, n_fft, target_bits, amp_safety=1.0)
    # Normalize per-slot amplitude scaling around 1.0 (handled inside allocator)
    return slots, amp_per_slot


@torch.no_grad()
def fast_gpu_plan_slots(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int) -> tuple[list[tuple[int,int]], torch.Tensor]:
    model = getattr(model, "module", model)
    """
    GPU planner for training: prioritize bins with high mag/headroom using a mel-proxy threshold.
    Returns (slots, amp_per_slot_tensor_on_cpu)
    """
    X = model.stft(x_wave)  # [1,2,F,T] on device
    B, _, F, T = X.shape
    assert B == 1
    mag = torch.sqrt(torch.clamp(X[:,0]**2 + X[:,1]**2, min=1e-6))  # [1,F,T]
    thr = mel_proxy_threshold(X, n_mels=64)  # [1,F,T]
    score = mag / (thr + 1e-6)               # higher score -> more headroom
    k = int(min(target_bits, F*T))
    vals, idx = score[0].flatten().topk(k)
    f = (idx // T).to(torch.int64)
    t = (idx % T).to(torch.int64)
    slots: list[tuple[int,int]] = [(int(f[i]), int(t[i])) for i in range(k)]
    amp = thr[0].flatten()[idx]
    # normalize median to 1.0
    med = torch.median(amp)
    amp = amp / (med + 1e-9)
    return slots, amp.detach().cpu()


@torch.no_grad()
def build_batch_plan(model: INNWatermarker, x: torch.Tensor, cfg: TrainConfig):
    """
    Per-item GPU planning but collated into batch tensors for vectorized loss.
    Returns dict with f_idx [B,S], t_idx [B,S], amp [B,S], mask [B,S], bits [B,S], S.
    """
    model = getattr(model, "module", model)
    B = x.size(0)
    f_lists: list[list[int]] = []
    t_lists: list[list[int]] = []
    amp_lists: list[torch.Tensor] = []
    S_list: list[int] = []
    for i in range(B):
        slots, amp = fast_gpu_plan_slots(model, x[i:i+1], cfg.target_bits)
        S = min(len(slots), cfg.target_bits)
        f = [slots[s][0] for s in range(S)]
        t = [slots[s][1] for s in range(S)]
        f_lists.append(f)
        t_lists.append(t)
        amp_lists.append(amp[:S] if amp.numel() >= S else torch.ones(S, dtype=torch.float32))
        S_list.append(S)
    S_max = int(max(S_list) if len(S_list) > 0 else 0)
    device = x.device
    if S_max == 0:
        return {
            "f_idx": torch.zeros(B, 0, dtype=torch.long, device=device),
            "t_idx": torch.zeros(B, 0, dtype=torch.long, device=device),
            "amp": torch.zeros(B, 0, dtype=torch.float32, device=device),
            "mask": torch.zeros(B, 0, dtype=torch.bool, device=device),
            "bits": torch.zeros(B, 0, dtype=torch.long, device=device),
            "S": 0,
        }
    f_idx = torch.full((B, S_max), -1, dtype=torch.long, device=device)
    t_idx = torch.full((B, S_max), -1, dtype=torch.long, device=device)
    amp = torch.ones(B, S_max, dtype=torch.float32, device=device)
    mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)
    for i in range(B):
        S = S_list[i]
        if S == 0:
            continue
        f_i = torch.tensor(f_lists[i], dtype=torch.long, device=device)
        t_i = torch.tensor(t_lists[i], dtype=torch.long, device=device)
        a_i = amp_lists[i].to(device)
        f_idx[i, :S] = f_i
        t_idx[i, :S] = t_i
        amp[i, :S] = a_i
        mask[i, :S] = True
    # Generate bits: either random (legacy) or RS-encoded payloads
    if cfg.use_rs_interleave:
        bits = torch.zeros(B, S_max, dtype=torch.long, device=device)
        for i in range(B):
            payload_bytes = bytes([random.randint(0, 255) for _ in range(cfg.rs_payload_bytes)])
            encoded_bits = encode_payload_with_rs(payload_bytes, cfg.rs_interleave_depth).to(device)
            # Truncate or pad to match S_max
            if encoded_bits.numel() >= S_max:
                bit_tensor = encoded_bits[:S_max]
            else:
                pad = torch.zeros(S_max - encoded_bits.numel(), dtype=torch.long, device=device)
                bit_tensor = torch.cat([encoded_bits, pad], dim=0)
            bits[i] = bit_tensor
    else:
        bits = torch.randint(0, 2, (B, S_max), device=device, dtype=torch.long)
    return {"f_idx": f_idx, "t_idx": t_idx, "amp": amp, "mask": mask, "bits": bits, "S": S_max}


@torch.no_grad()
def build_batch_plan_with_cache(model: INNWatermarker, x: torch.Tensor, paths: list[str], cfg: TrainConfig, epoch_idx: int, plan_cache: dict):
    base_model = getattr(model, "module", model)
    B = x.size(0)
    f_lists: list[list[int]] = []
    t_lists: list[list[int]] = []
    amp_lists: list[torch.Tensor] = []
    S_list: list[int] = []
    for i in range(B):
        key = (paths[i], cfg.target_bits)
        cached = plan_cache.get(key)
        need_replan = (cached is None) or ((epoch_idx % max(1, cfg.replan_every)) == 0)
        if cfg.planner == "gpu":
            slots, amp = fast_gpu_plan_slots(base_model, x[i:i+1], cfg.target_bits)
            if need_replan:
                plan_cache[key] = {"slots": slots}
        else:
            # fallback to CPU MG if requested (not recommended for speed)
            slots, amp = plan_slots_and_amp(base_model, x[i:i+1], TARGET_SR, cfg.n_fft, cfg.target_bits, cfg.base_symbol_amp)
            if need_replan:
                plan_cache[key] = {"slots": slots}
        S = min(len(slots), cfg.target_bits)
        f_lists.append([slots[s][0] for s in range(S)])
        t_lists.append([slots[s][1] for s in range(S)])
        # Ensure amp is a torch.Tensor
        if isinstance(amp, np.ndarray):
            amp_tensor = torch.from_numpy(amp[:S] if len(amp) >= S else np.ones(S, dtype=np.float32))
        else:
            amp_tensor = amp[:S] if amp.numel() >= S else torch.ones(S, dtype=torch.float32)
        amp_lists.append(amp_tensor)
        S_list.append(S)

    S_max = int(max(S_list) if len(S_list) > 0 else 0)
    device = x.device
    if S_max == 0:
        return {
            "f_idx": torch.zeros(B, 0, dtype=torch.long, device=device),
            "t_idx": torch.zeros(B, 0, dtype=torch.long, device=device),
            "amp": torch.zeros(B, 0, dtype=torch.float32, device=device),
            "mask": torch.zeros(B, 0, dtype=torch.bool, device=device),
            "bits": torch.zeros(B, 0, dtype=torch.long, device=device),
            "S": 0,
        }
    f_idx = torch.full((B, S_max), -1, dtype=torch.long, device=device)
    t_idx = torch.full((B, S_max), -1, dtype=torch.long, device=device)
    amp = torch.ones(B, S_max, dtype=torch.float32, device=device)
    mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)
    for i in range(B):
        S = S_list[i]
        if S == 0:
            continue
        f_idx[i, :S] = torch.tensor(f_lists[i], dtype=torch.long, device=device)
        t_idx[i, :S] = torch.tensor(t_lists[i], dtype=torch.long, device=device)
        if i < len(amp_lists):
            amp[i, :S] = amp_lists[i][:S].to(device)
        mask[i, :S] = True
    # Generate bits from fixed payload (or random fallback)
    if cfg.use_rs_interleave:
        bits = torch.zeros(B, S_max, dtype=torch.long, device=device)
        # Prepare fixed payload once
        fixed_bytes = build_fixed_payload_bytes(cfg) if cfg.use_fixed_payload else None
        for i in range(B):
            payload_bytes = fixed_bytes if fixed_bytes is not None else bytes([random.randint(0, 255) for _ in range(cfg.rs_payload_bytes)])
            encoded_bits = encode_payload_with_rs(payload_bytes, cfg.rs_interleave_depth)
            if isinstance(encoded_bits, torch.Tensor):
                encoded_bits = encoded_bits.to(device)
            else:
                encoded_bits = torch.tensor(encoded_bits, dtype=torch.long, device=device)
            bit_tensor = encoded_bits[:S_max] if encoded_bits.numel() >= S_max else torch.cat([encoded_bits, torch.zeros(S_max - encoded_bits.numel(), dtype=torch.long, device=device)], dim=0)
            bits[i] = bit_tensor
    else:
        bits = torch.randint(0, 2, (B, S_max), device=device, dtype=torch.long)
    return {"f_idx": f_idx, "t_idx": t_idx, "amp": amp, "mask": mask, "bits": bits, "S": S_max}


@torch.no_grad()
def build_message_spec_from_plan(model: INNWatermarker, x: torch.Tensor, plan: dict, base_symbol_amp: float) -> torch.Tensor:
    """
    Build message spectrogram M_spec in batch, identical logic for train and val.
    Uses indices and per-slot amplitudes from plan.
    """
    base_model = getattr(model, "module", model)
    X = base_model.stft(x)
    B, _, F_, T_ = X.shape
    S = int(plan.get("S", 0))
    M_spec = torch.zeros(B, 2, F_, T_, device=x.device)
    if S == 0:
        return M_spec
    f_idx = plan["f_idx"]
    t_idx = plan["t_idx"]
    amp = plan["amp"]
    bits = plan["bits"]
    mask = plan["mask"]
    b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, S)
    signs = (bits * 2 - 1).float() * base_symbol_amp * amp
    valid = mask
    M_spec[b_idx[valid], 0, f_idx[valid], t_idx[valid]] = signs[valid]
    return M_spec


def _maybe_resample_gpu(x: torch.Tensor, sr: torch.Tensor, target_sr: int) -> torch.Tensor:
    # x: [B,1,T], sr: [B]
    out = []
    for i in range(x.size(0)):
        xi = x[i:i+1]
        sri = int(sr[i].item()) if isinstance(sr, torch.Tensor) else int(sr[i])
        if sri != target_sr:
            xi = AF.resample(xi, orig_freq=sri, new_freq=target_sr)
        out.append(xi)
    return torch.cat(out, dim=0) if len(out) > 0 else x


def validate(model: INNWatermarker, stft_cfg: dict, loss_perc: CombinedPerceptualLoss, cfg: TrainConfig, loader: DataLoader) -> dict:
    model.eval()
    base_model = getattr(model, "module", model)
    metrics = {"ber": 0.0, "payload_ber": 0.0, "bits_ce": 0.0, "bits_mse": 0.0, "perc": 0.0, "mfcc": 0.0}
    with torch.no_grad():
        for batch in loader:
            x, sr, _paths = batch
            x = x.to(cfg.device, non_blocking=True)
            sr = sr.to(cfg.device) if isinstance(sr, torch.Tensor) else torch.tensor(sr, device=cfg.device)
            if cfg.gpu_resample:
                x = _maybe_resample_gpu(x, sr, TARGET_SR)
            B = x.size(0)
            # Use the same batch planning + M_spec construction as training
            bers = []
            payload_bers = []
            ce_sum = 0.0
            mse_sum = 0.0
            perc_sum = 0.0
            # Build a batch plan (no cache needed for val, but reuse function)
            paths = ["val_item"] * B
            plan = build_batch_plan_with_cache(model, x, paths, cfg, epoch_idx=0, plan_cache={})
            S = int(plan["S"])
            if S == 0:
                continue
            # Construct M_spec identically to training
            M_spec = build_message_spec_from_plan(model, x, plan, cfg.base_symbol_amp)
            # Encode -> Decode in batch
            x_wm, _ = base_model.encode(x, M_spec)
            x_wm = match_length(x_wm, x.size(-1))
            x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0)
            M_rec = base_model.decode(x_wm)
            M_rec = torch.nan_to_num(M_rec, nan=0.0, posinf=1.0, neginf=-1.0)
            # Compute metrics per item using the same planned indices
            f_idx = plan["f_idx"]; t_idx = plan["t_idx"]; mask = plan["mask"]; bits = plan["bits"]
            Bsz = x.size(0)
            rec_vals = torch.zeros(Bsz, S, device=x.device)
            b_idx = torch.arange(Bsz, device=x.device).unsqueeze(1).expand(Bsz, S)
            valid = mask
            rec_vals[valid] = M_rec[b_idx[valid], 0, f_idx[valid], t_idx[valid]]
            logits = rec_vals.clamp(-6.0, 6.0)
            targets01 = bits.float()
            ce = F.binary_cross_entropy_with_logits(logits[valid], targets01[valid])
            target_sign = (bits * 2 - 1).float() * cfg.base_symbol_amp * plan["amp"]
            mse = F.mse_loss(rec_vals[valid], target_sign[valid])
            perc_out = loss_perc(x, x_wm)
            perc = perc_out["total_perceptual_loss"]
            mfcc = perc_out["mfcc_cos"]
            pred_bits = (rec_vals > 0).long()
            ber = (pred_bits[valid] != bits[valid]).float().mean()
            bers.append(float(ber))
            # Payload BER
            if cfg.use_rs_interleave:
                try:
                    recovered_payload = decode_payload_with_rs(pred_bits[0].detach(), cfg.rs_interleave_depth, cfg.rs_payload_bytes)
                    original_payload = decode_payload_with_rs(bits[0].detach(), cfg.rs_interleave_depth, cfg.rs_payload_bytes)
                    payload_ber = sum(b1 != b2 for b1, b2 in zip(recovered_payload, original_payload)) / len(original_payload) if original_payload else 1.0
                except:
                    payload_ber = 1.0
                payload_bers.append(payload_ber)
            ce_sum += float(ce)
            mse_sum += float(mse)
            perc_sum += float(perc)
            metrics["mfcc"] += float(mfcc)
            if len(bers) > 0:
                metrics["ber"] += sum(bers) / len(bers) * B
                if cfg.use_rs_interleave and len(payload_bers) > 0:
                    metrics["payload_ber"] += sum(payload_bers) / len(payload_bers) * B
                metrics["bits_ce"] += ce_sum * 1.0
                metrics["bits_mse"] += mse_sum * 1.0
                metrics["perc"] += perc_sum * 1.0

    # All-reduce sums across DDP ranks before normalizing
    if dist.is_available() and dist.is_initialized():
        keys = list(metrics.keys())
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        t = torch.tensor([metrics[k] for k in keys], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            metrics[k] = float(t[i].item())
    for k in metrics:
        metrics[k] = metrics[k] / len(loader.dataset)
    return metrics


def train_one_epoch(model: INNWatermarker, stft_cfg: dict, loss_perc: CombinedPerceptualLoss, optimizer: torch.optim.Optimizer, scaler, cfg: TrainConfig, loader: DataLoader, *, epoch_idx: int, plan_cache: dict) -> dict:
    model.train()
    base_model = getattr(model, "module", model)
    running = {"obj": 0.0, "ber": 0.0, "bits_ce": 0.0, "bits_mse": 0.0, "perc": 0.0, "mfcc": 0.0}
    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc="train",
        leave=False,
        disable=(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0),
    )
    for step, batch in pbar:
        x, sr, paths = batch
        x = x.to(cfg.device, non_blocking=True)
        sr = sr.to(cfg.device) if isinstance(sr, torch.Tensor) else torch.tensor(sr, device=cfg.device)
        if cfg.gpu_resample:
            x = _maybe_resample_gpu(x, sr, TARGET_SR)
        B = x.size(0)

        # Batch plan (collated tensors for vector ops) with cache reuse
        plan = build_batch_plan_with_cache(model, x, paths, cfg, epoch_idx, plan_cache)
        S = plan["S"]
        if S == 0:
            continue

        bits = plan["bits"]              # [B,S]
        f_idx = plan["f_idx"]            # [B,S]
        t_idx = plan["t_idx"]            # [B,S]
        amp = plan["amp"]                # [B,S] per-slot amplitude scaling
        mask = plan["mask"]              # [B,S]

        # Build message spectrogram in batch
        X = base_model.stft(x)
        F_, T_ = X.shape[-2], X.shape[-1]
        M_spec = torch.zeros(B, 2, F_, T_, device=x.device)
        b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, S)
        signs = (bits * 2 - 1).float() * cfg.base_symbol_amp * amp
        valid = mask
        M_spec[b_idx[valid], 0, f_idx[valid], t_idx[valid]] = signs[valid]

        optimizer.zero_grad(set_to_none=True)
        device_type = x.device.type
        scaler_enabled = bool(getattr(scaler, "is_enabled", lambda: False)()) if scaler is not None else False
        if device_type == "cuda":
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=scaler_enabled)
            autocast_off_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        elif device_type == "cpu" and scaler_enabled:
            autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=True)
            autocast_off_ctx = nullcontext()
        else:
            autocast_ctx = nullcontext()
            autocast_off_ctx = nullcontext()

        with autocast_ctx:
            x_wm, _ = base_model.encode(x, M_spec)
            x_wm = match_length(x_wm, x.size(-1))
            x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0)
            M_rec = base_model.decode(x_wm)
            M_rec = torch.nan_to_num(M_rec, nan=0.0, posinf=1.0, neginf=-1.0)
        with autocast_off_ctx:
            rec_vals = torch.zeros(B, S, device=x.device)
            rec_vals[valid] = M_rec[b_idx[valid], 0, f_idx[valid], t_idx[valid]]
            # Keep unclamped rec_vals for MSE to reflect scale gaps
            logits = rec_vals.clamp(-6.0, 6.0)  # Clamp only for BCE
            targets01 = bits.float()
            L_bits_ce = F.binary_cross_entropy_with_logits(logits[valid], targets01[valid])
            target_sign = (bits * 2 - 1).float() * cfg.base_symbol_amp * amp
            L_bits_mse = F.mse_loss(rec_vals[valid], target_sign[valid])
            perc_out = loss_perc(x.float(), x_wm.float())
            L_perc = perc_out["total_perceptual_loss"]
            L_mfcc = perc_out["mfcc_cos"]
            L_obj = cfg.w_bits * L_bits_ce + cfg.w_mse * L_bits_mse + cfg.w_perc * L_perc

        if scaler is not None:
            scaler.scale(L_obj).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            L_obj.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            pred_bits = (rec_vals > 0).long()
            ber = (pred_bits[valid] != bits[valid]).float().mean()
            running["obj"] += float(L_obj) * B
            running["ber"] += float(ber) * B
            running["bits_ce"] += float(L_bits_ce) * B
            running["bits_mse"] += float(L_bits_mse) * B
            running["perc"] += float(L_perc) * B
            running["mfcc"] += float(L_mfcc) * B

        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "obj": f"{running['obj'] / ((step+1)*loader.batch_size):.4f}",
                "ber": f"{running['ber'] / ((step+1)*loader.batch_size):.4f}",
                "ce": f"{running['bits_ce'] / ((step+1)*loader.batch_size):.4f}",
                "mse": f"{running['bits_mse'] / ((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc'] / ((step+1)*loader.batch_size):.4f}",
                "mfcc": f"{running['mfcc'] / ((step+1)*loader.batch_size):.4f}",
            })

    # All-reduce sums across DDP ranks before normalizing
    if dist.is_available() and dist.is_initialized():
        keys = list(running.keys())
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        t = torch.tensor([running[k] for k in keys], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            running[k] = float(t[i].item())
    for k in running:
        running[k] = running[k] / len(loader.dataset)
    return running


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    # Enable cuDNN autotune and high-precision matmul
    cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Initialize Distributed
    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Prefer NCCL when available, otherwise fall back to Gloo (e.g., on Windows/CPU)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception:
            dist.init_process_group(backend="gloo", init_method="env://")
        is_distributed = True
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            cfg.device = f"cuda:{local_rank}"
        else:
            cfg.device = "cpu"

    train_ds = AudioChunkDataset(cfg.data_dir, gpu_resample=(cfg.device=="cuda" and cfg.gpu_resample))
    val_ds = AudioChunkDataset(cfg.val_dir, gpu_resample=(cfg.device=="cuda" and cfg.gpu_resample))
    # Optionally limit number of files for faster runs
    if cfg.train_max_files is not None and cfg.train_max_files > 0 and len(train_ds.files) > cfg.train_max_files:
        rnd = random.Random(cfg.file_seed)
        rnd.shuffle(train_ds.files)
        train_ds.files = train_ds.files[:cfg.train_max_files]
    if cfg.val_max_files is not None and cfg.val_max_files > 0 and len(val_ds.files) > cfg.val_max_files:
        rnd = random.Random(cfg.file_seed + 1)
        rnd.shuffle(val_ds.files)
        val_ds.files = val_ds.files[:cfg.val_max_files]
    # Simple logger that prints and appends to a file (rank-0 only)
    log_path = os.path.join(cfg.save_dir, getattr(cfg, "log_file", "train_log.txt"))
    def log(msg: str) -> None:
        if (not is_distributed) or rank == 0:
            print(msg)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
    # Start-of-run header
    if (not is_distributed) or rank == 0:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=== New run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception:
            pass

    if (not is_distributed) or rank == 0:
        log(f"Found {len(train_ds)} training files | {len(val_ds)} validation files")

    pin = (cfg.device == "cuda")
    # Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    if (not is_distributed) or rank == 0:
        log(f"Device: {cfg.device} | Batch: {cfg.batch_size} | Workers: {cfg.num_workers}")

    stft_cfg = {"n_fft": cfg.n_fft, "hop_length": cfg.hop, "win_length": cfg.n_fft}
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg=stft_cfg).to(cfg.device)
    # Warm-start from Stage-1 (imperceptibility) checkpoint if provided
    if cfg.init_from:
        if os.path.isfile(cfg.init_from):
            # Try weights_only=True first (PyTorch 2.6 default). If it fails for this file,
            # fall back to weights_only=False (only if the file is trusted).
            ckpt = None
            try:
                ckpt = torch.load(cfg.init_from, map_location=cfg.device)
            except Exception:
                try:
                    ckpt = torch.load(cfg.init_from, map_location=cfg.device, weights_only=False)
                except Exception as e:
                    print(f"Warning: failed to load init_from '{cfg.init_from}': {e}\nTraining from scratch.")
            if ckpt is not None:
                state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
                try:
                    model.load_state_dict(state, strict=True)
                    ep = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
                    log(f"Loaded init weights from '{cfg.init_from}' (epoch={ep})")
                except Exception as e:
                    log(f"Warning: checkpoint state load failed: {e}\nTraining from scratch.")
        else:
            log(f"Init checkpoint not found: '{cfg.init_from}'. Training from scratch.")

    # Wrap in DDP AFTER loading any checkpoint to avoid key prefix mismatches
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    loss_perc = CombinedPerceptualLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        GradScaler = torch.amp.GradScaler  # type: ignore[attr-defined]
    except Exception:
        from torch.cuda.amp import GradScaler  # type: ignore
    scaler = GradScaler(enabled=(cfg.mixed_precision and torch.cuda.is_available()))
    amp_enabled = cfg.mixed_precision and torch.cuda.is_available()

    best_ber = math.inf
    plan_cache: dict = {}
    # Remember default RS setting to restore after warmup/threshold
    default_use_rs = bool(cfg.use_rs_interleave)
    prev_use_rs = cfg.use_rs_interleave
    for epoch in range(1, cfg.epochs + 1):

        # RS enabled from epoch 1
        cfg.use_rs_interleave = default_use_rs
        if cfg.use_rs_interleave != prev_use_rs:
            log(f"Epoch {epoch}: RS+interleave {'ENABLED' if cfg.use_rs_interleave else 'DISABLED'}")
            prev_use_rs = cfg.use_rs_interleave

        # AMP always enabled
        amp_enabled = cfg.mixed_precision and torch.cuda.is_available()
        scaler = GradScaler(enabled=amp_enabled)


        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if (not is_distributed) or rank == 0:
            log(f"\nEpoch {epoch}/{cfg.epochs}")
        tr = train_one_epoch(model, stft_cfg, loss_perc, optimizer, scaler, cfg, train_loader, epoch_idx=epoch, plan_cache=plan_cache)
        if (not is_distributed) or rank == 0:
            log(f"train: obj={tr['obj']:.4f} ber={tr['ber']:.4f} ce={tr['bits_ce']:.4f} mse={tr['bits_mse']:.4f} perc={tr['perc']:.4f} mfcc={tr['mfcc']:.4f}")
        va = validate(model, stft_cfg, loss_perc, cfg, val_loader)
        if (not is_distributed) or rank == 0:
            payload_ber_str = f" p_ber={va['payload_ber']:.4f}" if cfg.use_rs_interleave else ""
            log(f"val  : ber={va['ber']:.4f}{payload_ber_str} ce={va['bits_ce']:.4f} mse={va['bits_mse']:.4f} perc={va['perc']:.4f} mfcc={va['mfcc']:.4f}")

        # Save by best BER
        if ((not is_distributed) or rank == 0) and (va["ber"] < best_ber):
            best_ber = va["ber"]
            ckpt_path = os.path.join(cfg.save_dir, "inn_decode_best.pt")
            to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state": to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ber": best_ber,
                "cfg": cfg.__dict__,
            }, ckpt_path)
            log(f"Saved best-by-BER to {ckpt_path}")

        if ((not is_distributed) or rank == 0) and (epoch % 5 == 0):
            ckpt_path = os.path.join(cfg.save_dir, f"inn_decode_e{epoch}.pt")
            to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state": to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ber": va["ber"],
                "cfg": cfg.__dict__,
            }, ckpt_path)
            log(f"Saved snapshot to {ckpt_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)


