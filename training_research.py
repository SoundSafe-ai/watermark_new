#!/usr/bin/env python3
"""
Research training script for the INN encoder/decoder, aligned with the
"Improving Payload Handling and Training for Audio Watermarking" document.

Key ideas implemented:
- 1 s segments at 44.1 kHz with an optional short (≈50 ms) sync marker
- EULDriver-based end-to-end encode/decode for supervision (RS(106,64) inside driver)
- INNWatermarker with RI-STFT front/back, invertible blocks, and message-spec embedding
- Composite perceptual loss (MR-STFT, MFCC, SNR) + byte-level objective proxy

This script is a lean, readable variant of the more feature-complete training_new.py,
intended for rapid research iteration and ablations.
"""

from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
import numpy as np

from models.inn_encoder_decoder import INNWatermarker
from pipeline.perceptual_losses import CombinedPerceptualLoss, MFCCCosineLoss
from pipeline.ingest_and_chunk import (
    EULDriver,
)
from pipeline.moore_galsberg import MooreGlasbergAnalyzer
from pipeline.adaptive_bit_allocation import PerceptualSignificanceMetric
import hashlib


# =========================
# Global constants
# =========================

TARGET_SR = 44100  # Updated to match INN model
CHUNK_SECONDS = 1
CHUNK_SAMPLES = TARGET_SR * CHUNK_SECONDS


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    # Data
    data_dir: str = os.environ.get("DATA_DIR", "data/train")
    val_dir: str = os.environ.get("VAL_DIR", "data/val")
    save_dir: str = os.environ.get("SAVE_DIR", "runs/research")
    num_workers: int = int(os.environ.get("NUM_WORKERS", 4))
    batch_size: int = int(os.environ.get("PER_DEVICE_BATCH", 4))

    # Training
    epochs: int = int(os.environ.get("EPOCHS", 20))
    lr: float = float(os.environ.get("LR", 2e-4))
    peak_lr: float = float(os.environ.get("PEAK_LR", 8e-4))
    min_lr: float = float(os.environ.get("MIN_LR", 1e-5))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 1000))
    mixed_precision: bool = os.environ.get("AMP", "1") == "1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 50

    # Payload / RS
    rs_payload_bytes: int = 64  # raw payload length (bytes); RS applied inside EULDriver (RS(106,64))
    interleave_depth: int = 4
    payload_seed: int = 1234
    use_fixed_payload: bool = True

    # EUL/embedding params
    base_symbol_amp: float = 0.08
    amp_safety: float = 1.0
    sync_strength: float = 0.05
    mapper_seed: int = 42

    # Loss weights with stability loss
    w_msg: float = 10.0  # Message loss weight
    w_perc: float = 1.0  # Perceptual loss weight
    w_inv: float = 1.0   # Invertibility loss weight
    w_stab: float = 0.05  # Stability loss weight (ramps to 0.5)
    
    # Stability loss parameters
    thr_step: float = 0.10  # Feature quantization step
    stability_ramp_start: float = 0.33  # Start ramping stability loss at 1/3 epochs
    stability_ramp_end: float = 1.0     # Finish ramping at end
    
    # Spectral normalization
    sn_enable: bool = True
    sn_target: float = 1.2
    sn_power_iter: int = 1
    sn_layers: List[str] = None  # Will be set in __post_init__
    
    # CSD (Channel Spectral Distortion) - placeholder for future implementation
    csd_start_frac: float = 0.6
    csd_final_scale: float = 0.7
    csd_schedule: str = "cosine"
    csd_feature_aware: bool = True
    
    # Decode precision
    decode_precision: str = "fp32"
    
    def __post_init__(self):
        if self.sn_layers is None:
            self.sn_layers = ["affine_coupling.*/proj.*", "coupling_net.*/fc.*", ".*1x1.*"]

    # IO
    ckpt_every: int = 1
    log_file: str = "train.log"
    save_name: str = "inn_research.pt"

    # Convenience
    max_train_files: Optional[int] = 25000
    max_val_files: Optional[int] = 10000


# =========================
# Dataset
# =========================

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
    def __init__(self, root: str, target_sr: int = TARGET_SR, max_files: Optional[int] = None):
        self.files = _list_audio_files(root)
        if max_files is not None and max_files > 0 and len(self.files) > max_files:
            self.files = self.files[:max_files]
        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {root}")
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(path)  # [C, T]
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
                    raise ImportError(
                        "Audio loading requires torchcodec/sox_io or soundfile."
                    ) from sf_err
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        return wav  # [1, T]

    def _random_1s_chunk(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(-1)
        if T < CHUNK_SAMPLES:
            pad = CHUNK_SAMPLES - T
            wav = F.pad(wav, (0, pad))
            return wav[:, :CHUNK_SAMPLES]
        start = random.randint(0, max(0, T - CHUNK_SAMPLES))
        return wav[:, start:start + CHUNK_SAMPLES]

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav = self._load_audio(path)
        chunk = self._random_1s_chunk(wav)
        return chunk  # [1, T]


# =========================
# Sync marker
# =========================

@torch.no_grad()
def embed_sync_marker(x_1s: torch.Tensor, strength: float, sr: int, seed: int) -> torch.Tensor:
    """Optional short sync marker at the start of the segment (~50 ms)."""
    if strength <= 0:
        return x_1s
    T = x_1s.size(-1)
    sync_len = int(round(0.050 * sr))  # 50 ms
    sync_len = min(sync_len, T)
    t = torch.linspace(0, sync_len / sr, steps=sync_len, device=x_1s.device, dtype=x_1s.dtype)
    freq = 1.0
    env = torch.hann_window(sync_len, device=x_1s.device, dtype=x_1s.dtype)
    sync = torch.sin(2 * math.pi * freq * t) * env
    sync = sync.view(1, 1, -1)
    out = x_1s.clone()
    out[..., :sync_len] = (out[..., :sync_len] + strength * sync).clamp(-1.0, 1.0)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


# =========================
# Payload helpers
# =========================

def _rand_payload_bytes(n: int, rng: random.Random) -> bytes:
    return bytes([rng.randrange(0, 256) for _ in range(n)])


def _bytes_to_bits_lsb_first(data: bytes) -> List[int]:
    bits: List[int] = []
    for b in data:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


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


def build_payload_bits_and_bytes(cfg: TrainConfig) -> Tuple[torch.Tensor, bytes, bytes]:
    rng = random.Random(cfg.payload_seed)
    if cfg.use_fixed_payload:
        # Deterministic but structured 64B for debugging; otherwise random
        fields = [
            b"SSAI",                  # magic 4
            b"RESRCH",               # tag 6
            os.urandom(8),           # nonce 8
            b"v1",                  # version 2
        ]
        pad_len = max(0, cfg.rs_payload_bytes - sum(len(f) for f in fields))
        payload = b"".join(fields) + bytes(pad_len)
    else:
        payload = _rand_payload_bytes(cfg.rs_payload_bytes, rng)

    # No RS/interleave here; EULDriver handles RS(106,64)+interleave internally
    bits = _bytes_to_bits_lsb_first(payload)
    return torch.tensor(bits, dtype=torch.long), payload, payload


# =========================
# Model and loss pipeline
# =========================

def _build_model(cfg: TrainConfig) -> INNWatermarker:
    # Match EULDriver training grid at 44.1 kHz: 882/441
    return INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 882, "hop_length": 441, "win_length": 882})


def compute_slot_stability_loss(
    x_ref: torch.Tensor, 
    x_wm: torch.Tensor, 
    cfg: TrainConfig,
    model: INNWatermarker
) -> torch.Tensor:
    """
    Compute slot-stability loss using quantized allocator features.
    
    Args:
        x_ref: Clean input audio [B, 1, T]
        x_wm: Watermarked output audio [B, 1, T]
        cfg: Training configuration
        model: INNWatermarker model for STFT computation
        
    Returns:
        Stability loss tensor
    """
    # Dataset enforces 44.1kHz; do not resample by tensor length
    
    # Compute STFT with exact INN parameters
    X_ref = model.stft(x_ref)  # [B, 2, F, T]
    X_wm = model.stft(x_wm)    # [B, 2, F, T]
    
    # Compute magnitude
    mag_ref = torch.sqrt(torch.clamp(X_ref[:, 0]**2 + X_ref[:, 1]**2, min=1e-12))
    mag_wm = torch.sqrt(torch.clamp(X_wm[:, 0]**2 + X_wm[:, 1]**2, min=1e-12))
    
    # Initialize Moore-Glasberg analyzer
    mg_analyzer = MooreGlasbergAnalyzer(
        sample_rate=TARGET_SR,
        n_fft=882,
        hop_length=441,
        n_critical_bands=24
    )
    
    # Compute quantized features for both clean and watermarked
    stability_losses = []
    
    for b in range(x_ref.size(0)):
        # Get magnitude for this batch item
        mag_ref_b = mag_ref[b].detach().cpu().numpy()  # [F, T]
        mag_wm_b = mag_wm[b].detach().cpu().numpy()    # [F, T]
        
        # Compute band thresholds using Moore-Glasberg
        band_thr_ref = mg_analyzer.band_thresholds(mag_ref_b)  # [BANDS, T]
        band_thr_wm = mg_analyzer.band_thresholds(mag_wm_b)    # [BANDS, T]
        
        # Quantize thresholds with same step as allocator
        quantized_thr_ref = np.floor(band_thr_ref / cfg.thr_step) * cfg.thr_step
        quantized_thr_wm = np.floor(band_thr_wm / cfg.thr_step) * cfg.thr_step
        
        # Compute perceptual significance using quantized features
        psm = PerceptualSignificanceMetric(method="inverse", use_median=True)
        sig_ref = psm.compute(quantized_thr_ref)  # [BANDS]
        sig_wm = psm.compute(quantized_thr_wm)    # [BANDS]
        
        # L1 loss between quantized features
        stability_loss = torch.mean(torch.abs(torch.from_numpy(sig_ref - sig_wm).float().to(x_ref.device)))
        stability_losses.append(stability_loss)
    
    return torch.stack(stability_losses).mean()


def apply_spectral_normalization(model: INNWatermarker, cfg: TrainConfig) -> None:
    """
    Apply spectral normalization to gain-critical layers.
    
    Args:
        model: INNWatermarker model
        cfg: Training configuration
    """
    if not cfg.sn_enable:
        return
    
    import re
    from torch.nn.utils import spectral_norm
    
    def should_normalize(name: str) -> bool:
        """Check if layer should be normalized based on pattern matching."""
        for pattern in cfg.sn_layers:
            if re.search(pattern, name):
                return True
        return False
    
    # Apply spectral normalization to matching layers
    matched = []
    for name, module in model.named_modules():
        # Light-touch targets: 1x1/proj/FC layers
        if should_normalize(name) and isinstance(module, (nn.Linear,)):
            try:
                spectral_norm(module, name='weight', n_power_iterations=cfg.sn_power_iter)
                # Optional: gently scale weights toward sn_target (>1.0)
                if hasattr(module, 'weight') and cfg.sn_target and cfg.sn_target != 1.0:
                    with torch.no_grad():
                        module.weight.data.mul_(float(cfg.sn_target))
                matched.append(name)
            except Exception as e:
                print(f"Warning: Could not apply spectral norm to {name}: {e}")
    if matched:
        print(f"SN attached to layers: {matched}")


def compute_losses_and_metrics(
    model: INNWatermarker,
    x_1s: torch.Tensor,
    cfg: TrainConfig,
    payload_bits: torch.Tensor,
    epoch: float = 0.0,
) -> Dict:
    base_model = getattr(model, "module", model)

    # STFT parameter assertions
    assert TARGET_SR == 44100, f"TARGET_SR must be 44100, got {TARGET_SR}"
    assert base_model.stft.n_fft == 882, f"STFT n_fft must be 882, got {base_model.stft.n_fft}"
    assert base_model.stft.hop == 441, f"STFT hop must be 441, got {base_model.stft.hop}"

    x_1s = torch.nan_to_num(x_1s, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    x_sync = embed_sync_marker(x_1s, cfg.sync_strength, TARGET_SR, cfg.mapper_seed)

    eul_driver = EULDriver(
        sr=TARGET_SR,
        n_fft=882,
        hop=441,
        rs_interleave=cfg.interleave_depth,
        per_eul_bits_target=106 * 8,
        base_symbol_amp=cfg.base_symbol_amp,
        amp_safety=cfg.amp_safety,
        thr_step=cfg.thr_step,
        use_anchor_seeding=True,
    )

    payload_bytes = _bits_to_bytes_lsb_first(payload_bits.tolist())
    x_wm = eul_driver.encode_eul(base_model, x_sync, payload_bytes)
    x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

    # Honor decode precision flag (float64 for QA)
    if cfg.decode_precision == "fp64":
        x_wm = x_wm.double()
        # EULDriver/INN use model dtype for STFT internally; this sets decode path to fp64
    decoded_bytes = eul_driver.decode_eul(base_model, x_wm, expected_bytes=cfg.rs_payload_bytes)
    original_payload_bytes = payload_bytes

    # Compute stability loss
    stability_loss = compute_slot_stability_loss(x_sync, x_wm, cfg, base_model)

    # Compute invertibility loss (placeholder - would need inverse pass)
    inv_loss = torch.tensor(0.0, device=x_1s.device)  # TODO: Implement actual invertibility loss

    if x_sync.dtype != torch.float32:
        x_sync_fp = x_sync.float()
    else:
        x_sync_fp = x_sync
    if x_wm.dtype != torch.float32:
        x_wm_fp = x_wm.float()
    else:
        x_wm_fp = x_wm

    perc = CombinedPerceptualLoss(mfcc=MFCCCosineLoss(sample_rate=TARGET_SR))(x_sync_fp, x_wm_fp)
    perc_total = perc["total_perceptual_loss"]

    with torch.no_grad():
        byte_acc = 0.0
        if len(decoded_bytes) > 0 and len(original_payload_bytes) > 0:
            L = min(len(decoded_bytes), len(original_payload_bytes))
            if L > 0:
                byte_acc = float(sum(1 for i in range(L) if decoded_bytes[i] == original_payload_bytes[i]) / L)
        payload_success = 1.0 if decoded_bytes == original_payload_bytes else 0.0
        byte_loss = 1.0 - byte_acc

    # Compute stability weight with ramping
    if epoch < cfg.stability_ramp_start:
        w_stab = cfg.w_stab
    elif epoch >= cfg.stability_ramp_end:
        w_stab = 0.5  # Final stability weight
    else:
        # Linear ramp from start to end
        ramp_progress = (epoch - cfg.stability_ramp_start) / (cfg.stability_ramp_end - cfg.stability_ramp_start)
        w_stab = cfg.w_stab + ramp_progress * (0.5 - cfg.w_stab)

    # Total loss with all components
    total = (cfg.w_msg * byte_loss + 
             cfg.w_perc * perc_total + 
             cfg.w_inv * inv_loss + 
             w_stab * stability_loss)

    return {
        "loss": total,
        "byte_loss": torch.tensor(byte_loss, device=x_1s.device, dtype=torch.float32),
        "perc": perc_total,
        "inv": inv_loss,
        "stability": stability_loss,
        "byte_acc": torch.tensor(byte_acc, device=x_1s.device, dtype=torch.float32),
        "payload_ok": torch.tensor(payload_success, device=x_1s.device, dtype=torch.float32),
        "w_stab": torch.tensor(w_stab, device=x_1s.device, dtype=torch.float32),
        "x_wm": x_wm.detach(),
    }


# =========================
# Train / Validate
# =========================

def _make_payload_bits_tensor(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    bits, _, _ = build_payload_bits_and_bytes(cfg)
    return bits.to(device)


def validate(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader, epoch: float = 0.0) -> Dict:
    model.eval()
    running = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "stability": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(cfg.device, non_blocking=True)
            bits = _make_payload_bits_tensor(cfg, x.device)
            out = compute_losses_and_metrics(model, x, cfg, bits, epoch)
            running["loss"] += float(out["loss"].detach().item()) * x.size(0)
            running["byte_loss"] += float(out["byte_loss"].detach().item()) * x.size(0)
            running["perc"] += float(out["perc"].detach().item()) * x.size(0)
            running["stability"] += float(out.get("stability", torch.tensor(0.0)).detach().item()) * x.size(0)
            running["byte_acc"] += float(out.get("byte_acc", torch.tensor(0.0)).detach().item()) * x.size(0)
            running["payload_ok"] += float(out.get("payload_ok", torch.tensor(0.0)).detach().item()) * x.size(0)
            n += int(x.size(0))
    n = max(1, n)
    for k in running:
        running[k] /= n
    return running


def train_one_epoch(model: INNWatermarker, cfg: TrainConfig, optimizer: torch.optim.Optimizer, loader: DataLoader, scaler, epoch: int) -> Dict:
    model.train()
    running = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "stability": 0.0, "byte_acc": 0.0, "payload_ok": 0.0, "w_stab": 0.0}
    local_samples = 0

    steps_per_epoch = len(loader)
    if not hasattr(cfg, "_global_step"):
        cfg._global_step = 0
    if not hasattr(cfg, "_total_steps_planned"):
        try:
            cfg._total_steps_planned = int(cfg.epochs) * int(steps_per_epoch)
        except Exception:
            cfg._total_steps_planned = steps_per_epoch

    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, batch in pbar:
        x = batch.to(cfg.device, non_blocking=True)
        bits = _make_payload_bits_tensor(cfg, x.device)
        optimizer.zero_grad(set_to_none=True)

        if hasattr(cfg, "warmup_steps"):
            if cfg._global_step < cfg.warmup_steps:
                lr_now = float(cfg.lr) + (float(cfg.peak_lr) - float(cfg.lr)) * (cfg._global_step / max(1, cfg.warmup_steps))
            else:
                denom = max(1, int(cfg._total_steps_planned) - int(cfg.warmup_steps))
                progress = min(1.0, max(0.0, (cfg._global_step - int(cfg.warmup_steps)) / denom))
                cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_now = float(cfg.min_lr) + (float(cfg.peak_lr) - float(cfg.min_lr)) * cos
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        use_amp = cfg.mixed_precision and torch.cuda.is_available()
        
        # Process each item in the batch individually (EULDriver requires B=1)
        batch_size = x.shape[0]
        total_loss = 0.0
        all_metrics = {}
        
        for i in range(batch_size):
            x_item = x[i:i+1]  # Keep batch dimension as 1
            bits_item = bits[i:i+1]  # Keep batch dimension as 1
            
            if use_amp and scaler is not None:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    out = compute_losses_and_metrics(model, x_item, cfg, bits_item, epoch)
                    loss = out["loss"]
            else:
                out = compute_losses_and_metrics(model, x_item, cfg, bits_item, epoch)
                loss = out["loss"]
            
            if not torch.isfinite(loss):
                continue
                
            total_loss += loss
            # Aggregate metrics (take mean for most metrics)
            for key, value in out.items():
                if key != "loss" and isinstance(value, torch.Tensor):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value.item() if value.numel() == 1 else value.mean().item())
        
        if total_loss == 0:
            cfg._global_step += 1
            continue
            
        # Average the loss across the batch
        loss = total_loss / batch_size
        
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Create final metrics dict
        out = {"loss": loss}
        for key, values in all_metrics.items():
            out[key] = sum(values) / len(values) if values else 0.0

        running["loss"] += float(loss.detach().item()) * x.size(0)
        running["byte_loss"] += float(out["byte_loss"]) * x.size(0)
        running["perc"] += float(out["perc"]) * x.size(0)
        running["stability"] += float(out.get("stability", 0.0)) * x.size(0)
        running["byte_acc"] += float(out.get("byte_acc", 0.0)) * x.size(0)
        running["payload_ok"] += float(out.get("payload_ok", 0.0)) * x.size(0)
        running["w_stab"] += float(out.get("w_stab", 0.0)) * x.size(0)
        local_samples += int(x.size(0))

        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['loss'] / ((step+1)*loader.batch_size):.4f}",
                "byte_loss": f"{running['byte_loss'] / ((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc'] / ((step+1)*loader.batch_size):.4f}",
                "stab": f"{running['stability'] / ((step+1)*loader.batch_size):.4f}",
                "w_stab": f"{running['w_stab'] / ((step+1)*loader.batch_size):.3f}",
                "byte_acc": f"{running['byte_acc'] / ((step+1)*loader.batch_size):.3f}",
                "payload_ok": f"{running['payload_ok'] / ((step+1)*loader.batch_size):.3f}",
            })

        cfg._global_step += 1

    global_samples = max(1, local_samples)
    for k in running:
        running[k] = running[k] / global_samples
    return running


def main(cfg: TrainConfig) -> None:
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

    train_ds = OneSecondDataset(cfg.data_dir, TARGET_SR, max_files=cfg.max_train_files)
    val_ds = OneSecondDataset(cfg.val_dir, TARGET_SR, max_files=cfg.max_val_files)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_distributed else None
    pin = (cfg.device.startswith("cuda"))
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

    model = _build_model(cfg).to(cfg.device)
    
    # Define log function before it's used
    log_path = os.path.join(cfg.save_dir, cfg.log_file)
    def log(msg: str) -> None:
        if (not is_distributed) or rank == 0:
            print(msg)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
    
    # Apply spectral normalization to gain-critical layers
    if cfg.sn_enable:
        apply_spectral_normalization(model, cfg)
        if (not is_distributed) or rank == 0:
            log(f"Applied spectral normalization to gain-critical layers")
    
    if is_distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.mixed_precision and torch.cuda.is_available())
    if (not is_distributed) or rank == 0:
        log(f"Files: train={len(train_ds)} | val={len(val_ds)} | SR={TARGET_SR}")

    best_byte_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if (not is_distributed) or rank == 0:
            log(f"\nEpoch {epoch}/{cfg.epochs}")

        train_metrics = train_one_epoch(model, cfg, optimizer, train_loader, scaler, epoch)
        if (not is_distributed) or rank == 0:
            log(f"train: loss={train_metrics['loss']:.4f} byte_loss={train_metrics['byte_loss']:.4f} perc={train_metrics['perc']:.4f} stab={train_metrics.get('stability',0.0):.4f} w_stab={train_metrics.get('w_stab',0.0):.3f} byte_acc={train_metrics.get('byte_acc',0.0):.3f} payload_ok={train_metrics.get('payload_ok',0.0):.3f}")

        val_metrics = validate(model, cfg, val_loader, epoch)
        if (not is_distributed) or rank == 0:
            log(f"val:   loss={val_metrics['loss']:.4f} byte_loss={val_metrics['byte_loss']:.4f} perc={val_metrics['perc']:.4f} stab={val_metrics.get('stability',0.0):.4f} byte_acc={val_metrics.get('byte_acc',0.0):.3f} payload_ok={val_metrics.get('payload_ok',0.0):.3f}")

            byte_acc_now = float(val_metrics.get("byte_acc", 0.0))
            is_best = byte_acc_now >= best_byte_acc
            if is_best:
                best_byte_acc = byte_acc_now
            if (epoch % cfg.ckpt_every == 0) or is_best:
                ckpt = {
                    "epoch": epoch,
                    "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "best_byte_acc": best_byte_acc,
                    "cfg": vars(cfg),
                }
                name = ("best_" if is_best else "") + cfg.save_name
                path = os.path.join(cfg.save_dir, name)
                torch.save(ckpt, path)
                log(f"Saved checkpoint: {path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="INN research training")
    parser.add_argument("--data_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--save_dir", type=str, default="new_run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_val_files", type=int, default=None)
    args, _ = parser.parse_known_args()

    _defaults = TrainConfig()
    cfg = TrainConfig(
        data_dir=(args.data_dir or os.environ.get("DATA_DIR", _defaults.data_dir)),
        val_dir=(args.val_dir or os.environ.get("VAL_DIR", _defaults.val_dir)),
        save_dir=(args.save_dir or os.environ.get("SAVE_DIR", _defaults.save_dir)),
        epochs=(args.epochs if args.epochs is not None else int(os.environ.get("EPOCHS", _defaults.epochs))),
        batch_size=(args.batch_size if args.batch_size is not None else int(os.environ.get("PER_DEVICE_BATCH", _defaults.batch_size))),
        max_train_files=(args.max_train_files if args.max_train_files is not None else (int(os.environ.get("MAX_TRAIN_FILES")) if os.environ.get("MAX_TRAIN_FILES") else _defaults.max_train_files)),
        max_val_files=(args.max_val_files if args.max_val_files is not None else (int(os.environ.get("MAX_VAL_FILES")) if os.environ.get("MAX_VAL_FILES") else _defaults.max_val_files)),
    )
    main(cfg)


