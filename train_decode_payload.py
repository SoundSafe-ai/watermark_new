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
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from models.inn_encoder_decoder import INNWatermarker, STFT
from pipeline.perceptual_losses import CombinedPerceptualLoss
from pipeline.ingest_and_chunk import (
    allocate_slots_and_amplitudes,
    bits_to_message_spec,
    message_spec_to_bits,
)

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
    def __init__(self, root: str, target_sr: int = TARGET_SR):
        self.files = list_audio_files(root)
        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in {root}")
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C, T]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        wav = wav / (wav.abs().max() + 1e-9)
        return wav  # [1,T]

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
        return self._random_1s_chunk(wav)


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
    batch_size: int = 6
    num_workers: int = 2
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    save_dir: str = "checkpoints"
    log_interval: int = 50
    n_fft: int = 1024
    hop: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Loss weights
    w_bits: float = 1.0
    w_mse: float = 0.25
    w_perc: float = 0.05
    # Symbol settings
    base_symbol_amp: float = 0.1
    target_bits: int = 512  # per 1s chunk (<= realistic slot count)


def make_bits(batch_size: int, S: int, device) -> torch.Tensor:
    return torch.randint(low=0, high=2, size=(batch_size, S), device=device, dtype=torch.long)


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
    # Plan on clean audio to avoid label leakage
    X = model.stft(x_wave)  # [B,2,F,T]
    assert X.size(0) == 1, "Call per item for deterministic plan"
    slots, amp_per_slot = allocate_slots_and_amplitudes(X, sr, n_fft, target_bits, amp_safety=1.0)
    # Normalize per-slot amplitude scaling around 1.0 (handled inside allocator)
    return slots, amp_per_slot


def validate(model: INNWatermarker, stft_cfg: dict, loss_perc: CombinedPerceptualLoss, cfg: TrainConfig, loader: DataLoader) -> dict:
    model.eval()
    metrics = {"ber": 0.0, "bits_ce": 0.0, "bits_mse": 0.0, "perc": 0.0}
    with torch.no_grad():
        for batch in loader:
            x = batch.to(cfg.device, non_blocking=True)
            B = x.size(0)
            # Plan slots on a per-item basis
            bers = []
            ce_sum = 0.0
            mse_sum = 0.0
            perc_sum = 0.0
            for i in range(B):
                xi = x[i:i+1]
                slots, amp_scale = plan_slots_and_amp(model, xi, cfg.n_fft, cfg.n_fft, cfg.target_bits, cfg.base_symbol_amp)
                S = min(len(slots), cfg.target_bits)
                if S == 0:
                    continue
                bits = make_bits(1, S, xi.device)
                # Build message spec
                X = model.stft(xi)
                F_, T_ = X.shape[-2], X.shape[-1]
                amp_vec = amp_scale[:S] if len(amp_scale) >= S else None
                M_spec = bits_to_message_spec(bits, slots[:S], F_, T_, base_amp=cfg.base_symbol_amp, amp_per_slot=amp_vec)
                # Encode -> Decode
                x_wm, _ = model.encode(xi, M_spec)
                x_wm = match_length(x_wm, xi.size(-1))
                M_rec = model.decode(x_wm)
                # Loss terms
                logits = gather_slots(M_rec, slots[:S]) * 8.0
                targets01 = bits.float()
                ce = F.binary_cross_entropy_with_logits(logits, targets01)
                target_sign = symbols_from_bits(bits, amp=cfg.base_symbol_amp)
                rec_vals = gather_slots(M_rec, slots[:S])
                mse = F.mse_loss(rec_vals, target_sign)
                perc = loss_perc(xi, x_wm)["total_perceptual_loss"]
                # BER
                pred_bits = (rec_vals > 0).long()
                ber = (pred_bits != bits).float().mean()
                bers.append(float(ber))
                ce_sum += float(ce)
                mse_sum += float(mse)
                perc_sum += float(perc)
            if len(bers) > 0:
                metrics["ber"] += sum(bers) / len(bers) * B
                metrics["bits_ce"] += ce_sum * 1.0
                metrics["bits_mse"] += mse_sum * 1.0
                metrics["perc"] += perc_sum * 1.0

    for k in metrics:
        metrics[k] = metrics[k] / len(loader.dataset)
    return metrics


def train_one_epoch(model: INNWatermarker, stft_cfg: dict, loss_perc: CombinedPerceptualLoss, optimizer: torch.optim.Optimizer, scaler, cfg: TrainConfig, loader: DataLoader) -> dict:
    model.train()
    running = {"obj": 0.0, "ber": 0.0, "bits_ce": 0.0, "bits_mse": 0.0, "perc": 0.0}
    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, batch in pbar:
        x = batch.to(cfg.device, non_blocking=True)
        B = x.size(0)
        obj_sum = 0.0; ber_sum = 0.0; ce_sum = 0.0; mse_sum = 0.0; perc_sum = 0.0

        for i in range(B):
            xi = x[i:i+1]
            slots, amp_scale = plan_slots_and_amp(model, xi, cfg.n_fft, cfg.n_fft, cfg.target_bits, cfg.base_symbol_amp)
            S = min(len(slots), cfg.target_bits)
            if S == 0:
                continue
            bits = make_bits(1, S, xi.device)
            X = model.stft(xi)
            F_, T_ = X.shape[-2], X.shape[-1]
            amp_vec = amp_scale[:S] if len(amp_scale) >= S else None
            M_spec = bits_to_message_spec(bits, slots[:S], F_, T_, base_amp=cfg.base_symbol_amp, amp_per_slot=amp_vec)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                x_wm, _ = model.encode(xi, M_spec)
                x_wm = match_length(x_wm, xi.size(-1))
                M_rec = model.decode(x_wm)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                logits = gather_slots(M_rec, slots[:S]) * 8.0
                targets01 = bits.float()
                L_bits_ce = F.binary_cross_entropy_with_logits(logits, targets01)
                target_sign = symbols_from_bits(bits, amp=cfg.base_symbol_amp)
                rec_vals = gather_slots(M_rec, slots[:S])
                L_bits_mse = F.mse_loss(rec_vals, target_sign)
                L_perc = loss_perc(xi.float(), x_wm.float())["total_perceptual_loss"]
                L_obj = cfg.w_bits * L_bits_ce + cfg.w_mse * L_bits_mse + cfg.w_perc * L_perc

            if scaler is not None:
                scaler.scale(L_obj).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                L_obj.backward()
                optimizer.step()

            # logging
            with torch.no_grad():
                pred_bits = (rec_vals > 0).long()
                ber = (pred_bits != bits).float().mean()
                obj_sum += float(L_obj)
                ber_sum += float(ber)
                ce_sum += float(L_bits_ce)
                mse_sum += float(L_bits_mse)
                perc_sum += float(L_perc)

        denom = max(1, B)
        running["obj"] += obj_sum / denom * B
        running["ber"] += ber_sum / denom * B
        running["bits_ce"] += ce_sum / denom * B
        running["bits_mse"] += mse_sum / denom * B
        running["perc"] += perc_sum / denom * B

        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "obj": f"{running['obj'] / ((step+1)*loader.batch_size):.4f}",
                "ber": f"{running['ber'] / ((step+1)*loader.batch_size):.4f}",
                "ce": f"{running['bits_ce'] / ((step+1)*loader.batch_size):.4f}",
                "mse": f"{running['bits_mse'] / ((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc'] / ((step+1)*loader.batch_size):.4f}",
            })

    for k in running:
        running[k] = running[k] / len(loader.dataset)
    return running


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    train_ds = AudioChunkDataset(cfg.data_dir)
    val_ds = AudioChunkDataset(cfg.val_dir)
    print(f"Found {len(train_ds)} training files | {len(val_ds)} validation files")

    pin = (cfg.device == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
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
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    print(f"Device: {cfg.device} | Batch: {cfg.batch_size} | Workers: {cfg.num_workers}")

    stft_cfg = {"n_fft": cfg.n_fft, "hop_length": cfg.hop, "win_length": cfg.n_fft}
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg=stft_cfg).to(cfg.device)
    loss_perc = CombinedPerceptualLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        GradScaler = torch.amp.GradScaler  # type: ignore[attr-defined]
    except Exception:
        from torch.cuda.amp import GradScaler  # type: ignore
    scaler = GradScaler(enabled=(cfg.mixed_precision and torch.cuda.is_available()))

    best_ber = math.inf
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr = train_one_epoch(model, stft_cfg, loss_perc, optimizer, scaler, cfg, train_loader)
        print(f"train: obj={tr['obj']:.4f} ber={tr['ber']:.4f} ce={tr['bits_ce']:.4f} mse={tr['bits_mse']:.4f} perc={tr['perc']:.4f}")
        va = validate(model, stft_cfg, loss_perc, cfg, val_loader)
        print(f"val  : ber={va['ber']:.4f} ce={va['bits_ce']:.4f} mse={va['bits_mse']:.4f} perc={va['perc']:.4f}")

        # Save by best BER
        if va["ber"] < best_ber:
            best_ber = va["ber"]
            ckpt_path = os.path.join(cfg.save_dir, "inn_decode_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ber": best_ber,
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"Saved best-by-BER to {ckpt_path}")

        if epoch % 5 == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"inn_decode_e{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ber": va["ber"],
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"Saved snapshot to {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)


