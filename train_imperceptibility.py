#!/usr/bin/env python3
"""
Imperceptibility-focused training script for SoundSafe INNWatermarker.

Dataset layout:
  ./data/  -> training wav files (any nested structure)
  ./val/   -> validation wav files

Training objective (imperceptibility only):
  Minimize CombinedPerceptualLoss(original, watermarked) which includes
  - MultiResSTFT loss
  - MFCC cosine loss (explicitly requested)
  - Time-domain SNR penalty

We do not optimize payload recovery here. A dummy message spectrogram is used
to drive the encoder pathway without influencing the loss directly.
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
from phm.perceptual_frontend import PerceptualFrontend
from phm.technical_frontend import TechnicalFrontend
from phm.fusion_head import FusionHead
from phm.telemetry import TechnicalTelemetry

# Silence specific torchaudio warnings mentioned by the user
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
warnings.filterwarnings(
    "ignore",
    message=r".*torch.cuda.amp.autocast\(.*\) is deprecated.*",
    category=FutureWarning,
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
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != self.target_sr:
            wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
        wav = wav / (wav.abs().max() + 1e-9)
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
        chunk = self._random_1s_chunk(wav)  # [1, T]
        return chunk


@dataclass
class TrainConfig:
    data_dir: str = "data/train"
    val_dir: str = "data/val"
    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    save_dir: str = "checkpoints"
    log_interval: int = 50
    n_fft: int = 1024
    hop: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Loss weights
    w_perc: float = 1.0
    w_msg_rec: float = 0.1
    w_msg_amp: float = 1e-4


def build_message_like_spec(x_wave: torch.Tensor, stft: STFT) -> torch.Tensor:
    """
    Build a dummy message spectrogram shaped like the audio STFT.
    We do not supervise message recovery here; we only need a non-zero
    message input to exercise the encoder path. We use small noise.
    """
    with torch.no_grad():
        X = stft(x_wave)  # [B,2,F,T]
    noise = torch.randn_like(X) * 0.01
    return noise


def match_length(y: torch.Tensor, target_T: int) -> torch.Tensor:
    """
    Ensure waveform y has time dimension == target_T by cropping or right-padding with zeros.
    y: [B,1,T'] -> [B,1,target_T]
    """
    T = y.size(-1)
    if T == target_T:
        return y
    if T > target_T:
        return y[..., :target_T]
    pad = target_T - T
    return F.pad(y, (0, pad))


def validate(model: INNWatermarker, loss_fn: CombinedPerceptualLoss, loader: DataLoader, device: str) -> dict:
    model.eval()
    metrics = {"total": 0.0, "mrstft": 0.0, "mfcc": 0.0, "snr": 0.0,
               "presence": 0.0, "reliability": 0.0, "artifact": 0.0}

    # PHM assessors (inference-only; not trained here)
    perc_model = PerceptualFrontend().to(device)
    tech_model = TechnicalFrontend(feat_dim=8).to(device)
    fusion_head = FusionHead().to(device)
    perc_model.eval(); tech_model.eval(); fusion_head.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)  # [B,1,T]
            m = build_message_like_spec(x, model.stft).to(device)
            x_wm, _ = model.encode(x, m)
            x_wm = match_length(x_wm, x.size(-1))
            L = loss_fn(x, x_wm)
            metrics["total"] += L["total_perceptual_loss"].detach().item() * x.size(0)
            metrics["mrstft"] += L["mrstft_total"].detach().item() * x.size(0)
            metrics["mfcc"] += L["mfcc_cos"].detach().item() * x.size(0)
            metrics["snr"] += L["snr"].detach().item() * x.size(0)

            # PHM metrics (no grad)
            X_ri = model.stft(x_wm)  # [B,2,F,T]
            M_rec = model.decode(x_wm)  # [B,2,F,T]
            # Build 8-feature telemetry sequence [B,T,8]
            mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-9))  # [B,F,T]
            conf = torch.sigmoid(torch.abs(M_rec[:,0]))  # [B,F,T]
            softbit_conf = conf.mean(dim=1).unsqueeze(-1)                     # [B,T,1]
            snr_proxy = (torch.abs(M_rec[:,0]) / (mag + 1e-9)).clamp(0, 10.0).mean(dim=1).unsqueeze(-1)
            frames = X_ri.size(-1)
            slot_fill = torch.ones(x.size(0), frames, 1, device=device)
            sync_drift = torch.zeros_like(slot_fill)
            rs_errata = torch.zeros_like(slot_fill)
            conf_max = conf.max(dim=1).values.unsqueeze(-1)
            conf_std = conf.std(dim=1, unbiased=False).unsqueeze(-1)
            mag_mean = mag.mean(dim=1).unsqueeze(-1)
            tech_seq = torch.cat([softbit_conf, snr_proxy, slot_fill, sync_drift, rs_errata, conf_max, conf_std, mag_mean], dim=-1)
            perc_vec, _ = perc_model.infer_features(x_wm)
            tech_vec, _ = tech_model.infer_features(tech_seq)
            fused = fusion_head(perc_vec, tech_vec)
            metrics["presence"] += fused["presence_p"].mean().detach().item() * x.size(0)
            metrics["reliability"] += fused["decode_reliability"].mean().detach().item() * x.size(0)
            metrics["artifact"] += fused["artifact_risk"].mean().detach().item() * x.size(0)

    for k in metrics.keys():
        metrics[k] = metrics[k] / len(loader.dataset)
    return metrics


def train_one_epoch(model: INNWatermarker, loss_fn: CombinedPerceptualLoss, optimizer: torch.optim.Optimizer, loader: DataLoader, device: str, scaler: torch.cuda.amp.GradScaler | None, log_interval: int) -> dict:
    model.train()
    running = {"total": 0.0, "mrstft": 0.0, "mfcc": 0.0, "snr": 0.0,
               "presence": 0.0, "reliability": 0.0, "artifact": 0.0}

    # PHM assessors (inference only)
    perc_model = PerceptualFrontend().to(device)
    tech_model = TechnicalFrontend(feat_dim=8).to(device)
    fusion_head = FusionHead().to(device)
    perc_model.eval(); tech_model.eval(); fusion_head.eval()

    pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
    for step, batch in pbar:
        x = batch.to(device)
        m = build_message_like_spec(x, model.stft).to(device)

        optimizer.zero_grad(set_to_none=True)
        # Use new torch.amp.autocast API to avoid deprecation warnings
        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            x_wm, _ = model.encode(x, m)
            x_wm = match_length(x_wm, x.size(-1))
            # Decode in AMP for speed
            M_rec = model.decode(x_wm)
        # Compute losses in full precision to avoid NaNs from fp16 logs/trigs
        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Imperceptibility (MRSTFT + MFCC + SNR)
            Lp = loss_fn(x.float(), x_wm.float())
            # Message self-consistency (decode should match the injected message)
            L_msg_rec = F.mse_loss(M_rec.float(), m.float())
            # Message energy regularizer (keep message small)
            L_msg_amp = m.float().abs().mean()
            # Total
            loss = cfg.w_perc * Lp["total_perceptual_loss"] + cfg.w_msg_rec * L_msg_rec + cfg.w_msg_amp * L_msg_amp

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running["total"] += Lp["total_perceptual_loss"].detach().item() * x.size(0)
        running["mrstft"] += Lp["mrstft_total"].detach().item() * x.size(0)
        running["mfcc"] += Lp["mfcc_cos"].detach().item() * x.size(0)
        running["snr"] += Lp["snr"].detach().item() * x.size(0)

        # PHM telemetry + metrics for logging
        with torch.no_grad():
            X_ri = model.stft(x_wm)
            mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-6))
            conf = torch.sigmoid(torch.abs(M_rec[:,0]))
            softbit_conf = conf.mean(dim=1).unsqueeze(-1)
            snr_proxy = (torch.abs(M_rec[:,0]) / (mag + 1e-6)).clamp(0, 10.0).mean(dim=1).unsqueeze(-1)
            frames = X_ri.size(-1)
            slot_fill = torch.ones(x.size(0), frames, 1, device=device)
            sync_drift = torch.zeros_like(slot_fill)
            rs_errata = torch.zeros_like(slot_fill)
            conf_max = conf.max(dim=1).values.unsqueeze(-1)
            conf_std = conf.std(dim=1, unbiased=False).unsqueeze(-1)
            mag_mean = mag.mean(dim=1).unsqueeze(-1)
            tech_seq = torch.cat([softbit_conf, snr_proxy, slot_fill, sync_drift, rs_errata, conf_max, conf_std, mag_mean], dim=-1)
            perc_vec, _ = perc_model.infer_features(x_wm)
            tech_vec, _ = tech_model.infer_features(tech_seq)
            fused = fusion_head(perc_vec, tech_vec)
            running["presence"] += fused["presence_p"].mean().detach().item() * x.size(0)
            running["reliability"] += fused["decode_reliability"].mean().detach().item() * x.size(0)
            running["artifact"] += fused["artifact_risk"].mean().detach().item() * x.size(0)

        if (step + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running['total'] / ((step+1)*loader.batch_size):.4f}",
                "mrstft": f"{running['mrstft'] / ((step+1)*loader.batch_size):.4f}",
                "mfcc": f"{running['mfcc'] / ((step+1)*loader.batch_size):.4f}",
                "snr": f"{running['snr'] / ((step+1)*loader.batch_size):.4f}",
                "P": f"{running['presence'] / ((step+1)*loader.batch_size):.3f}",
                "R": f"{running['reliability'] / ((step+1)*loader.batch_size):.3f}",
                "A": f"{running['artifact'] / ((step+1)*loader.batch_size):.3f}",
            })

    for k in running.keys():
        running[k] = running[k] / len(loader.dataset)
    return running


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Datasets
    train_ds = AudioChunkDataset(cfg.data_dir)
    val_ds = AudioChunkDataset(cfg.val_dir)

    # Report dataset stats
    print(f"Found {len(train_ds)} training files in '{cfg.data_dir}'")
    print(f"Found {len(val_ds)} validation files in '{cfg.val_dir}'")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print(f"Device: {cfg.device} | Batch size: {cfg.batch_size} | Workers: {cfg.num_workers}")
    print(f"Train steps/epoch: {len(train_loader)} | Val steps: {len(val_loader)}")

    # Model & loss
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": cfg.n_fft, "hop_length": cfg.hop, "win_length": cfg.n_fft}).to(cfg.device)
    loss_fn = CombinedPerceptualLoss()

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Prefer torch.amp.GradScaler when available
    try:
        GradScaler = torch.amp.GradScaler  # type: ignore[attr-defined]
    except Exception:
        from torch.cuda.amp import GradScaler  # type: ignore
    scaler = GradScaler(enabled=(cfg.mixed_precision and torch.cuda.is_available()))

    best_val = math.inf
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_metrics = train_one_epoch(model, loss_fn, optimizer, train_loader, cfg.device, scaler, cfg.log_interval)
        print(f"train: total={train_metrics['total']:.4f} mrstft={train_metrics['mrstft']:.4f} mfcc={train_metrics['mfcc']:.4f} snr={train_metrics['snr']:.4f}"
              f" | PHM P={train_metrics['presence']:.3f} R={train_metrics['reliability']:.3f} A={train_metrics['artifact']:.3f}")

        val_metrics = validate(model, loss_fn, val_loader, cfg.device)
        print(f"val  : total={val_metrics['total']:.4f} mrstft={val_metrics['mrstft']:.4f} mfcc={val_metrics['mfcc']:.4f} snr={val_metrics['snr']:.4f}"
              f" | PHM P={val_metrics['presence']:.3f} R={val_metrics['reliability']:.3f} A={val_metrics['artifact']:.3f}")

        # Save best on validation total perceptual loss
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            ckpt_path = os.path.join(cfg.save_dir, f"inn_imperc_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_total": best_val,
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

        # periodic save
        if epoch % 5 == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"inn_imperc_e{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_total": val_metrics["total"],
                "cfg": cfg.__dict__,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)


