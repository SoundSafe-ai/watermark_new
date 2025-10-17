#!/usr/bin/env python3
"""
Research training script for the INN encoder/decoder, aligned with the
"Improving Payload Handling and Training for Audio Watermarking" document.

Key ideas implemented:
- 1 s segments at 22.05 kHz with a faint per-second sync marker
- EULDriver-based end-to-end encode/decode for supervision (RS(167,125) + interleave)
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

from models.inn_encoder_decoder import INNWatermarker
from pipeline.perceptual_losses import CombinedPerceptualLoss, MFCCCosineLoss
from pipeline.ingest_and_chunk import (
    EULDriver,
    rs_encode_167_125,
    deinterleave_bytes,
    interleave_bytes,
)


# =========================
# Global constants
# =========================

TARGET_SR = 22050
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
    rs_payload_bytes: int = 125  # pre-RS payload length (bytes)
    interleave_depth: int = 4
    payload_seed: int = 1234
    use_fixed_payload: bool = True

    # EUL/embedding params
    base_symbol_amp: float = 0.08
    amp_safety: float = 1.0
    sync_strength: float = 0.06
    mapper_seed: int = 42

    # Loss weights
    w_byte: float = 1.0
    w_perc: float = 0.1
    w_amp: float = 0.0

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
    B, C, T = 1, 1, x_1s.size(-1)
    t = torch.linspace(0, T / sr, steps=T, device=x_1s.device, dtype=x_1s.dtype)
    freq = 1.0
    env = torch.hann_window(T, device=x_1s.device, dtype=x_1s.dtype)
    sync = (torch.sin(2 * math.pi * freq * t) * env).unsqueeze(0).unsqueeze(0).to(dtype=x_1s.dtype)
    out = (x_1s + strength * sync)
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
            bytes(125 - (4 + 6 + 8 + 2))  # pad to 125
        ]
        payload = b"".join(fields[:3]) + fields[3] + fields[4]
    else:
        payload = _rand_payload_bytes(cfg.rs_payload_bytes, rng)

    coded = interleave_bytes(rs_encode_167_125(payload), cfg.interleave_depth)
    bits = _bytes_to_bits_lsb_first(coded)
    return torch.tensor(bits, dtype=torch.long), payload, coded


# =========================
# Model and loss pipeline
# =========================

def _build_model(cfg: TrainConfig) -> INNWatermarker:
    # Match EULDriver training grid at 22.05 kHz: 882/441
    return INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 882, "hop_length": 441, "win_length": 882})


def compute_losses_and_metrics(
    model: INNWatermarker,
    x_1s: torch.Tensor,
    cfg: TrainConfig,
    payload_bits: torch.Tensor,
) -> Dict:
    base_model = getattr(model, "module", model)

    x_1s = torch.nan_to_num(x_1s, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    x_sync = embed_sync_marker(x_1s, cfg.sync_strength, TARGET_SR, cfg.mapper_seed)

    eul_driver = EULDriver(
        sr=TARGET_SR,
        n_fft=882,
        hop=441,
        rs_interleave=cfg.interleave_depth,
        per_eul_bits_target=cfg.rs_payload_bytes * 8,
        base_symbol_amp=cfg.base_symbol_amp,
        amp_safety=cfg.amp_safety,
    )

    payload_bytes = _bits_to_bytes_lsb_first(payload_bits.tolist())
    x_wm = eul_driver.encode_eul(base_model, x_sync, payload_bytes)
    x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)

    decoded_bytes = eul_driver.decode_eul(base_model, x_wm, expected_bytes=cfg.rs_payload_bytes)
    original_payload_bytes = _bits_to_bytes_lsb_first(payload_bits.tolist())

    amp_penalty = torch.tensor(0.0, device=x_1s.device)

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

    total = cfg.w_byte * byte_loss + cfg.w_amp * amp_penalty + cfg.w_perc * perc_total

    return {
        "loss": total,
        "byte_loss": torch.tensor(byte_loss, device=x_1s.device, dtype=torch.float32),
        "amp_penalty": amp_penalty,
        "perc": perc_total,
        "byte_acc": torch.tensor(byte_acc, device=x_1s.device, dtype=torch.float32),
        "payload_ok": torch.tensor(payload_success, device=x_1s.device, dtype=torch.float32),
        "x_wm": x_wm.detach(),
    }


# =========================
# Train / Validate
# =========================

def _make_payload_bits_tensor(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    bits, _, _ = build_payload_bits_and_bytes(cfg)
    return bits.to(device)


def validate(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader) -> Dict:
    model.eval()
    running = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(cfg.device, non_blocking=True)
            bits = _make_payload_bits_tensor(cfg, x.device)
            out = compute_losses_and_metrics(model, x, cfg, bits)
            running["loss"] += float(out["loss"].detach().item()) * x.size(0)
            running["byte_loss"] += float(out["byte_loss"].detach().item()) * x.size(0)
            running["perc"] += float(out["perc"].detach().item()) * x.size(0)
            running["byte_acc"] += float(out.get("byte_acc", torch.tensor(0.0)).detach().item()) * x.size(0)
            running["payload_ok"] += float(out.get("payload_ok", torch.tensor(0.0)).detach().item()) * x.size(0)
            n += int(x.size(0))
    n = max(1, n)
    for k in running:
        running[k] /= n
    return running


def train_one_epoch(model: INNWatermarker, cfg: TrainConfig, optimizer: torch.optim.Optimizer, loader: DataLoader, scaler, epoch: int) -> Dict:
    model.train()
    running = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
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
        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                out = compute_losses_and_metrics(model, x, cfg, bits)
                loss = out["loss"]
            if not torch.isfinite(loss):
                cfg._global_step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = compute_losses_and_metrics(model, x, cfg, bits)
            loss = out["loss"]
            if not torch.isfinite(loss):
                cfg._global_step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running["loss"] += float(loss.detach().item()) * x.size(0)
        running["byte_loss"] += float(out["byte_loss"].detach().item()) * x.size(0)
        running["perc"] += float(out["perc"].detach().item()) * x.size(0)
        running["byte_acc"] += float(out.get("byte_acc", torch.tensor(0.0)).detach().item()) * x.size(0)
        running["payload_ok"] += float(out.get("payload_ok", torch.tensor(0.0)).detach().item()) * x.size(0)
        local_samples += int(x.size(0))

        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['loss'] / ((step+1)*loader.batch_size):.4f}",
                "byte_loss": f"{running['byte_loss'] / ((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc'] / ((step+1)*loader.batch_size):.4f}",
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
        pin_memory_device=("cuda" if pin else "cpu"),
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
        pin_memory_device=("cuda" if pin else "cpu"),
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

    model = _build_model(cfg).to(cfg.device)
    if is_distributed and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision and torch.cuda.is_available())

    log_path = os.path.join(cfg.save_dir, cfg.log_file)
    def log(msg: str) -> None:
        if (not is_distributed) or rank == 0:
            print(msg)
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

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
            log(f"train: loss={train_metrics['loss']:.4f} byte_loss={train_metrics['byte_loss']:.4f} perc={train_metrics['perc']:.4f} byte_acc={train_metrics.get('byte_acc',0.0):.3f} payload_ok={train_metrics.get('payload_ok',0.0):.3f}")

        val_metrics = validate(model, cfg, val_loader)
        if (not is_distributed) or rank == 0:
            log(f"val:   loss={val_metrics['loss']:.4f} byte_loss={val_metrics['byte_loss']:.4f} perc={val_metrics['perc']:.4f} byte_acc={val_metrics.get('byte_acc',0.0):.3f} payload_ok={val_metrics.get('payload_ok',0.0):.3f}")

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
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
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


