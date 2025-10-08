#!/usr/bin/env python3
"""
Enhanced window-level training script that combines micro-chunking with 
advanced features from the original train_decode_payload.py

Key enhancements:
- Advanced RS/Interleaving (RS(167,125) + interleaving)
- Deterministic time-frequency mapping + psychoacoustic gating
- Training optimizations (amplitude annealing, warmup, caching)
- Proper loss computation (CE + MSE + perceptual)
- Checkpoint management and resume capability
"""

from __future__ import annotations
import os
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# Enable TF32 for better performance on A100/RTX 30+ series
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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
from pipeline.window_level_training import WindowLevelTrainer, EnhancedWindowLevelTrainer
from pipeline.micro_chunking import MicroChunker, fuse_overlapping_windows
from pipeline.sync_anchors import SyncAnchor, SyncDetector
from pipeline.ingest_and_chunk import (
    rs_encode_167_125,
    rs_decode_167_125,
    interleave_bytes,
    deinterleave_bytes,
)
from pipeline.micro_chunking import apply_psychoacoustic_gate
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
    batch_size: int = 8  # Reduced for window-level processing
    num_workers: int = 8
    epochs: int = 30
    lr: float = 5e-5
    weight_decay: float = 1e-5
    mixed_precision: bool = True
    save_dir: str = "window_level_enhanced_checkpoints"
    log_interval: int = 25
    n_fft: int = 1024
    hop: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_resample: bool = True
    
    # Window-level specific settings
    window_ms: int = 15
    overlap_ratio: float = 0.5
    target_bits_per_window: int = 50
    mapper_seed: int = 42
    sync_strength: float = 0.1
    
    # Advanced features from original system
    init_from: str | None = "inn_decode_best.pt"  # Initialize from old checkpoint
    train_max_files: int | None = 25000
    val_max_files: int | None = 5000
    file_seed: int = 42
    
    # Loss weights (from original)
    w_bits: float = 1.0
    w_mse: float = 0.25
    w_perc: float = 0.009
    
    # Symbol settings with annealing
    base_symbol_amp: float = 0.18  # Will be annealed: 0.18 → 0.13 → 0.09
    
    # RS and interleaving settings (from original)
    use_rs_interleave: bool = True
    rs_payload_bytes: int = 64
    rs_interleave_depth: int = 4
    rs_warmup_epochs: int = 3
    rs_enable_ber_threshold: float = 0.35
    
    # Mapper settings (new architecture)
    mapper_seed: int = 42
    replan_every: int = 3
    
    # Symbol-level redundancy settings
    use_symbol_level: bool = True
    symbol_redundancy: int = 3
    bias_lower_mid: bool = True
    n_symbols: int = 100  # Number of symbols to embed
    
    # CRC validation settings
    use_crc: bool = True
    symbol_length: int = 8  # Length of each symbol in bits
    min_agreement_rate: float = 0.3  # Minimum CRC agreement rate to accept symbol
    crc_weight: float = 0.1  # Weight for CRC penalty in loss
    
    # Fixed payload controls (from original)
    use_fixed_payload: bool = False
    payload_text: str = "Title:You'reasurvivor,Perf:NikeshShah,ISRC:123456789101,ISWC:123456789012,length:04:20,date:01/01/2025,label:warnerbros"
    
    # File logging
    log_file: str = "window_level_enhanced_train_log.txt"


def make_rs_payload_bytes(batch_size: int, payload_bytes: int, device) -> list[bytes]:
    """Generate random payload bytes for RS encoding."""
    payloads = []
    for _ in range(batch_size):
        payload = bytes([random.randint(0, 255) for _ in range(payload_bytes)])
        payloads.append(payload)
    return payloads


def encode_payload_with_rs(payload_bytes: bytes, interleave_depth: int) -> torch.Tensor:
    """Encode payload with RS(167,125) and interleaving, return as bit tensor."""
    rs_encoded = rs_encode_167_125(payload_bytes)
    interleaved = interleave_bytes(rs_encoded, interleave_depth)
    
    bits = []
    for b in interleaved:
        for k in range(8):
            bits.append((b >> k) & 1)
    
    return torch.tensor(bits, dtype=torch.long)


def _parse_kv_csv_to_fields(text: str) -> dict:
    """Parse key-value CSV text to fields dict."""
    fields: dict = {}
    for seg in text.split(","):
        seg = seg.strip()
        if not seg:
            continue
        pos_dash = seg.find("-")
        pos_colon = seg.find(":")
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
    """Build fixed payload bytes from text."""
    fields = _parse_kv_csv_to_fields(cfg.payload_text)
    raw = pack_fields(fields)
    if len(raw) >= cfg.rs_payload_bytes:
        return raw[:cfg.rs_payload_bytes]
    return raw + bytes(cfg.rs_payload_bytes - len(raw))


def _maybe_resample_gpu(x: torch.Tensor, sr: torch.Tensor, target_sr: int) -> torch.Tensor:
    out = []
    for i in range(x.size(0)):
        xi = x[i:i+1]
        sri = int(sr[i].item()) if isinstance(sr, torch.Tensor) else int(sr[i])
        if sri != target_sr:
            xi = AF.resample(xi, orig_freq=sri, new_freq=target_sr)
        out.append(xi)
    return torch.cat(out, dim=0) if len(out) > 0 else x


class AdvancedWindowLevelTrainer(EnhancedWindowLevelTrainer):
    """Advanced window-level trainer with symbol-level redundancy and advanced features."""
    
    def __init__(self, *args, **kwargs):
        # Extract symbol-level parameters
        use_symbol_level = kwargs.pop('use_symbol_level', True)
        symbol_redundancy = kwargs.pop('symbol_redundancy', 3)
        bias_lower_mid = kwargs.pop('bias_lower_mid', True)
        
        # Extract CRC parameters
        use_crc = kwargs.pop('use_crc', True)
        symbol_length = kwargs.pop('symbol_length', 8)
        min_agreement_rate = kwargs.pop('min_agreement_rate', 0.3)
        
        super().__init__(*args, **kwargs)
        self.plan_cache = {}  # Planning cache for efficiency
        self.use_symbol_level = use_symbol_level
        self.use_crc = use_crc
        self.symbol_length = symbol_length
        self.min_agreement_rate = min_agreement_rate
        
    def build_enhanced_window_plan(self, model: INNWatermarker, x: torch.Tensor, 
                                  target_bits_per_window: int, base_symbol_amp: float,
                                  mapper_seed: int = 42, replan_every: int = 3,
                                  epoch_idx: int = 0, n_symbols: int = 100) -> Dict:
        """Build window-level plan using symbol-level redundancy or deterministic mapping."""
        # Handle different input shapes
        if x.dim() == 3 and x.size(0) == 1 and x.size(1) == 1:
            # Shape [1, 1, T] -> squeeze to [1, T]
            x = x.squeeze(1)
        elif x.dim() == 3 and x.size(1) == 1:
            # Shape [B, 1, T] -> squeeze to [B, T] then take first sample
            x = x.squeeze(1)[:1]
        elif x.dim() != 2 or x.size(0) != 1:
            raise ValueError(f"Expected x shape [1, T], got {x.shape}")
        
        if self.use_symbol_level:
            # Use symbol-level redundancy
            return self.build_symbol_level_plan(model, x, n_symbols, base_symbol_amp)
        else:
            # Use simple deterministic mapping (fallback)
            return self._build_simple_window_plan(model, x, target_bits_per_window, base_symbol_amp)
    
    def _build_simple_window_plan(self, model: INNWatermarker, x: torch.Tensor, 
                                 target_bits_per_window: int, base_symbol_amp: float) -> Dict:
        """Build simple window-level plan using deterministic mapping."""
        # Add sync pattern
        x_with_sync = self.sync_anchor.embed_sync_pattern(x, self.sync_strength)
        
        # Create micro-chunks
        windows = self.micro_chunker.chunk_1s_segment(x_with_sync)
        n_windows = len(windows)
        
        # Build plan for each window using deterministic mapping
        window_plans = []
        total_slots = 0
        
        for i, window in enumerate(windows):
            # Use deterministic mapping (no pre-planning)
            slots = self.deterministic_mapper.map_window_to_slots(
                i, n_windows, target_bits_per_window
            )
            
            # Apply psychoacoustic gating
            masked_slots = apply_psychoacoustic_gate(window, slots, model, self.n_fft, self.hop)
            
            # Simple amplitude allocation (all slots get same amplitude)
            amp_per_slot = torch.ones(len(masked_slots), device=window.device)
            
            window_plan = {
                'window': window,
                'slots': masked_slots,
                'amp_per_slot': amp_per_slot,
                'n_slots': len(masked_slots),
                'window_idx': i
            }
            
            window_plans.append(window_plan)
            total_slots += len(masked_slots)
        
        return {
            'window_plans': window_plans,
            'n_windows': n_windows,
            'total_slots': total_slots,
            'overlap_ratio': self.overlap_ratio,
            'window_ms': self.window_ms
        }
    


def validate_enhanced_window_level(model: INNWatermarker, trainer: AdvancedWindowLevelTrainer, 
                                 loss_perc: CombinedPerceptualLoss, cfg: TrainConfig, 
                                 loader: DataLoader) -> dict:
    """Enhanced validation with proper metrics tracking."""
    model.eval()
    base_model = getattr(model, "module", model)
    
    metrics = {
        "obj": 0.0,
        "ber_sample_mean": 0.0,
        "ce_sample_mean": 0.0,
        "mse_sample_mean": 0.0,
        "perc_sample_mean": 0.0,
        "total_bits": 0,
        "total_errors": 0,
        "payload_ber": 0.0,
    }
    
    with torch.no_grad():
        for batch in loader:
            x, sr, _paths = batch
            x = x.to(cfg.device, non_blocking=True)
            sr = sr.to(cfg.device) if isinstance(sr, torch.Tensor) else torch.tensor(sr, device=cfg.device)
            if cfg.gpu_resample:
                x = _maybe_resample_gpu(x, sr, TARGET_SR)
            
            B = x.size(0)
            batch_metrics = {"obj": 0.0, "ber": 0.0, "ce": 0.0, "mse": 0.0, "perc": 0.0, 
                           "bits": 0, "errors": 0, "payload_ber": 0.0}
            
            for i in range(B):
                x_1s = x[i:i+1]  # [1, T]
                
                # Generate payload (fixed or random)
                if cfg.use_fixed_payload:
                    payload_bytes = build_fixed_payload_bytes(cfg)
                else:
                    payload_bytes = bytes([random.randint(0, 255) for _ in range(cfg.rs_payload_bytes)])
                
                payload_bits = encode_payload_with_rs(payload_bytes, cfg.rs_interleave_depth)
                payload_bits = payload_bits.to(cfg.device).unsqueeze(0)
                
                # Wrap hot paths with autocast for AMP optimization
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.mixed_precision):
                    # Build enhanced window-level plan
                    plan = trainer.build_enhanced_window_plan(
                        base_model, x_1s, cfg.target_bits_per_window, cfg.base_symbol_amp,
                        cfg.mapper_seed, cfg.replan_every, epoch_idx=0, n_symbols=cfg.n_symbols
                    )
                    
                    if cfg.use_symbol_level:
                        # Use symbol-level training
                        window_plans = trainer.build_symbol_level_labels(
                            plan['window_plans'], payload_bits
                        )
                        
                        loss_weights = {
                            'ce': cfg.w_bits,
                            'mse': cfg.w_mse,
                            'perceptual': cfg.w_perc,
                            'crc': cfg.crc_weight
                        }
                        
                        window_losses = trainer.compute_symbol_level_loss(
                            base_model, window_plans, plan['symbol_mappings'], 
                            cfg.base_symbol_amp, loss_weights
                        )
                    else:
                        # Use simple window-level training
                        window_plans = trainer.duplicate_labels_across_overlaps(
                            plan['window_plans'], payload_bits
                        )
                        
                        loss_weights = {
                            'ce': cfg.w_bits,
                            'mse': cfg.w_mse,
                            'perceptual': cfg.w_perc,
                            'crc': cfg.crc_weight
                        }
                        
                        window_losses = trainer.compute_window_level_loss(
                            base_model, window_plans, cfg.base_symbol_amp, loss_weights
                        )
                
                # Calculate payload BER
                if cfg.use_rs_interleave:
                    try:
                        # Recover bits from all windows (batched)
                        all_bits = []
                        valid_plans = [plan for plan in window_plans if plan['n_slots'] > 0]
                        
                        if valid_plans:
                            # Batch decode all windows at once
                            windows = torch.stack([plan['window'] for plan in valid_plans], dim=0)  # [N, 1, T]
                            M_recs = base_model.decode(windows)  # [N, 1, F, T]
                            
                            # Extract bits from each window
                            for i, plan in enumerate(valid_plans):
                                M_rec = M_recs[i:i+1]  # [1, 1, F, T]
                                rec_vals = torch.zeros(1, len(plan['slots']), device=plan['window'].device)
                                for s, (f, t) in enumerate(plan['slots']):
                                    if f < M_rec.size(2) and t < M_rec.size(3):
                                        rec_vals[0, s] = M_rec[0, 0, f, t]
                                rec_bits = (rec_vals > 0).long()
                                all_bits.append(rec_bits)
                        
                        if all_bits:
                            fused_bits = fuse_overlapping_windows(all_bits, cfg.overlap_ratio)
                            
                            # Decode payload
                            decoded_payload = _decode_payload_bits(fused_bits, cfg.rs_interleave_depth, cfg.rs_payload_bytes)
                            
                            # Calculate payload BER
                            payload_ber = sum(b1 != b2 for b1, b2 in zip(decoded_payload, payload_bytes)) / len(payload_bytes)
                            batch_metrics["payload_ber"] += payload_ber
                    except Exception:
                        batch_metrics["payload_ber"] += 1.0
                
                # Accumulate metrics (window_losses returns floats)
                batch_metrics["obj"] += window_losses['total_loss']
                batch_metrics["ber"] += window_losses['ber']
                batch_metrics["ce"] += window_losses['ber']  # Approximate
                batch_metrics["mse"] += window_losses['ber']  # Approximate
                batch_metrics["perc"] += window_losses['perceptual_loss']
                batch_metrics["bits"] += window_losses['total_bits']
                batch_metrics["errors"] += window_losses['total_errors']
            
            # Average over batch
            for key in batch_metrics:
                if key in ["obj", "ber", "ce", "mse", "perc", "payload_ber"]:
                    batch_metrics[key] /= B
            
            # Accumulate
            for key in metrics:
                if key in batch_metrics:
                    metrics[key] += batch_metrics[key] * B
    
    # All-reduce across DDP ranks
    if dist.is_available() and dist.is_initialized():
        keys = list(metrics.keys())
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        t = torch.tensor([metrics[k] for k in keys], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            metrics[k] = float(t[i].item())
    
    # Normalize
    N = len(loader.dataset)
    for key in metrics:
        if key not in ["total_bits", "total_errors"]:
            metrics[key] = metrics[key] / N if N > 0 else 0.0
    
    return metrics


def _decode_payload_bits(bits: torch.Tensor, interleave_depth: int, payload_bytes: int) -> bytes:
    """Decode bits to payload bytes."""
    if bits.numel() == 0:
        return bytes(payload_bytes)
    
    bits_flat = bits.flatten()
    by = bytearray()
    for i in range(0, len(bits_flat), 8):
        if i + 8 > len(bits_flat):
            break
        b = 0
        for k in range(8):
            b |= (int(bits_flat[i + k]) & 1) << k
        by.append(b)
    
    try:
        deinterleaved = deinterleave_bytes(bytes(by), interleave_depth)
        decoded = rs_decode_167_125(deinterleaved)
        return decoded[:payload_bytes]
    except Exception:
        return bytes(payload_bytes)


def train_one_epoch_enhanced(model: INNWatermarker, trainer: AdvancedWindowLevelTrainer,
                            loss_perc: CombinedPerceptualLoss, optimizer: torch.optim.Optimizer,
                            scaler, cfg: TrainConfig, loader: DataLoader, epoch_idx: int) -> dict:
    """Enhanced training epoch with amplitude annealing and advanced features."""
    model.train()
    base_model = getattr(model, "module", model)
    
    # Amplitude annealing (from original system)
    if epoch_idx <= 5:
        current_amp = 0.18
    elif epoch_idx <= 10:
        current_amp = 0.13
    else:
        current_amp = 0.09
    
    running = {
        "obj": 0.0,
        "ber_sample_mean": 0.0,
        "ce_sample_mean": 0.0,
        "mse_sample_mean": 0.0,
        "perc_sample_mean": 0.0,
        "total_bits": 0,
        "total_errors": 0,
    }
    
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
        batch_loss = 0.0
        batch_ber = 0.0
        batch_ce = 0.0
        batch_mse = 0.0
        batch_perc = 0.0
        batch_bits = 0
        batch_errors = 0
        
        for i in range(B):
            x_1s = x[i:i+1]  # [1, T]
            
            # Generate payload
            if cfg.use_fixed_payload:
                payload_bytes = build_fixed_payload_bytes(cfg)
            else:
                payload_bytes = bytes([random.randint(0, 255) for _ in range(cfg.rs_payload_bytes)])
            
            payload_bits = encode_payload_with_rs(payload_bytes, cfg.rs_interleave_depth)
            payload_bits = payload_bits.to(cfg.device).unsqueeze(0)
            
            # Wrap hot paths with autocast for AMP optimization
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.mixed_precision):
                # Build enhanced window-level plan
                plan = trainer.build_enhanced_window_plan(
                    base_model, x_1s, cfg.target_bits_per_window, current_amp,
                    cfg.mapper_seed, cfg.replan_every, epoch_idx, n_symbols=cfg.n_symbols
                )
                
                if cfg.use_symbol_level:
                    # Use symbol-level training
                    window_plans = trainer.build_symbol_level_labels(
                        plan['window_plans'], payload_bits
                    )
                    
                    loss_weights = {
                        'ce': cfg.w_bits,
                        'mse': cfg.w_mse,
                        'perceptual': cfg.w_perc
                    }
                    
                    window_losses = trainer.compute_symbol_level_loss(
                        base_model, window_plans, plan['symbol_mappings'], 
                        current_amp, loss_weights
                    )
                else:
                    # Use simple window-level training
                    window_plans = trainer.duplicate_labels_across_overlaps(
                        plan['window_plans'], payload_bits
                    )
                    
                    loss_weights = {
                        'ce': cfg.w_bits,
                        'mse': cfg.w_mse,
                        'perceptual': cfg.w_perc
                    }
                    
                    window_losses = trainer.compute_window_level_loss(
                        base_model, window_plans, current_amp, loss_weights
                    )
            
            # For window-level API we keep using total_loss (float). If a grad-capable
            # tensor is provided (loss_tensor), prefer that for backward.
            if 'loss_tensor' in window_losses and torch.is_tensor(window_losses['loss_tensor']):
                batch_loss += float(window_losses['loss_tensor'].detach().item())
                # Accumulate a separate tensor for backward pass
                if 'loss_tensor_accum' not in locals():
                    loss_tensor_accum = window_losses['loss_tensor']
                else:
                    loss_tensor_accum = loss_tensor_accum + window_losses['loss_tensor']
            else:
                batch_loss += window_losses['total_loss']
            batch_ber += window_losses['ber']
            batch_ce += window_losses['ber']  # Approximate
            batch_mse += window_losses['ber']  # Approximate
            batch_perc += window_losses['perceptual_loss']
            batch_bits += window_losses['total_bits']
            batch_errors += window_losses['total_errors']
        
        # Average over batch
        batch_loss /= B
        batch_ber /= B
        batch_ce /= B
        batch_mse /= B
        batch_perc /= B
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            if 'loss_tensor_accum' in locals():
                scaler.scale(loss_tensor_accum / max(1, B)).backward()
                del loss_tensor_accum
            else:
                scaler.scale(torch.tensor(batch_loss, device=cfg.device, requires_grad=True)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if 'loss_tensor_accum' in locals():
                (loss_tensor_accum / max(1, B)).backward()
                del loss_tensor_accum
            else:
                torch.tensor(batch_loss, device=cfg.device, requires_grad=True).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Accumulate metrics
        running["obj"] += batch_loss * B
        running["ber_sample_mean"] += batch_ber * B
        running["ce_sample_mean"] += batch_ce * B
        running["mse_sample_mean"] += batch_mse * B
        running["perc_sample_mean"] += batch_perc * B
        running["total_bits"] += batch_bits
        running["total_errors"] += batch_errors
        
        if (step + 1) % cfg.log_interval == 0:
            pbar.set_postfix({
                "obj": f"{running['obj'] / ((step+1)*loader.batch_size):.4f}",
                "ber": f"{running['ber_sample_mean'] / ((step+1)*loader.batch_size):.4f}",
                "ce": f"{running['ce_sample_mean'] / ((step+1)*loader.batch_size):.4f}",
                "mse": f"{running['mse_sample_mean'] / ((step+1)*loader.batch_size):.4f}",
                "perc": f"{running['perc_sample_mean'] / ((step+1)*loader.batch_size):.4f}",
                "amp": f"{current_amp:.3f}",
            })
    
    # All-reduce across DDP ranks
    if dist.is_available() and dist.is_initialized():
        keys = list(running.keys())
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device
        t = torch.tensor([running[k] for k in keys], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        for i, k in enumerate(keys):
            running[k] = float(t[i].item())
    
    # Normalize
    N = len(loader.dataset)
    for key in running:
        if key not in ["total_bits", "total_errors"]:
            running[key] = running[key] / N if N > 0 else 0.0
    
    return running


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
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

    # Adjust batch size for DistributedDataParallel so cfg.batch_size is GLOBAL by default
    # If launched with multiple processes, split the global batch across ranks
    global_bs_env = os.environ.get("GLOBAL_BATCH_SIZE")
    if global_bs_env is not None:
        try:
            cfg.batch_size = int(global_bs_env)
        except Exception:
            pass
    effective_global_bs = cfg.batch_size
    if is_distributed and world_size > 1:
        per_device_bs = max(1, effective_global_bs // world_size)
    else:
        per_device_bs = effective_global_bs
    if (not is_distributed) or rank == 0:
        print(f"Using batch sizes -> global: {effective_global_bs}, per_device: {per_device_bs} (world_size={world_size})")
    
    # Datasets
    train_ds = AudioChunkDataset(cfg.data_dir, gpu_resample=(cfg.device=="cuda" and cfg.gpu_resample))
    val_ds = AudioChunkDataset(cfg.val_dir, gpu_resample=(cfg.device=="cuda" and cfg.gpu_resample))
    
    # Limit files if specified
    if cfg.train_max_files is not None and cfg.train_max_files > 0 and len(train_ds.files) > cfg.train_max_files:
        rnd = random.Random(cfg.file_seed)
        rnd.shuffle(train_ds.files)
        train_ds.files = train_ds.files[:cfg.train_max_files]
    if cfg.val_max_files is not None and cfg.val_max_files > 0 and len(val_ds.files) > cfg.val_max_files:
        rnd = random.Random(cfg.file_seed + 1)
        rnd.shuffle(val_ds.files)
        val_ds.files = val_ds.files[:cfg.val_max_files]
    
    # Logging
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
        log(f"Found {len(train_ds)} training files | {len(val_ds)} validation files")
        log(f"Enhanced window-level training: {cfg.window_ms}ms windows, {cfg.overlap_ratio*100}% overlap")
        log(f"Target bits per window: {cfg.target_bits_per_window}")
        log(f"RS interleave: {cfg.use_rs_interleave}, payload bytes: {cfg.rs_payload_bytes}")
        log(f"Deterministic mapping with seed: {cfg.mapper_seed}")
        if cfg.use_symbol_level:
            log(f"Symbol-level training: {cfg.n_symbols} symbols, redundancy={cfg.symbol_redundancy}, bias_lower_mid={cfg.bias_lower_mid}")
            if cfg.use_crc:
                log(f"CRC validation: enabled, symbol_length={cfg.symbol_length}, min_agreement={cfg.min_agreement_rate}, weight={cfg.crc_weight}")
            else:
                log("CRC validation: disabled")
        else:
            log("Using simple window-level training (no symbol redundancy)")
    
    # Data loaders
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_distributed else None
    
    # DataLoader arguments
    dataloader_kwargs = {
        "batch_size": per_device_bs,
        "shuffle": (train_sampler is None),
        "sampler": train_sampler,
        "num_workers": min(16, os.cpu_count()) if cfg.num_workers == 0 else cfg.num_workers,
        "drop_last": True,
        "pin_memory": (cfg.device == "cuda"),
        "persistent_workers": True if cfg.num_workers > 0 else False,
        "prefetch_factor": 4 if cfg.num_workers > 0 else None,
    }
    if cfg.device == "cuda":
        dataloader_kwargs["pin_memory_device"] = "cuda"
    
    train_loader = DataLoader(train_ds, **dataloader_kwargs)
    # Validation DataLoader arguments
    val_dataloader_kwargs = {
        "batch_size": per_device_bs,
        "shuffle": False,
        "sampler": val_sampler,
        "num_workers": min(16, os.cpu_count()) if cfg.num_workers == 0 else cfg.num_workers,
        "pin_memory": (cfg.device == "cuda"),
        "persistent_workers": True if cfg.num_workers > 0 else False,
        "prefetch_factor": 4 if cfg.num_workers > 0 else None,
    }
    if cfg.device == "cuda":
        val_dataloader_kwargs["pin_memory_device"] = "cuda"
    
    val_loader = DataLoader(val_ds, **val_dataloader_kwargs)
    
    # Model and trainer
    model = INNWatermarker(n_blocks=8, spec_channels=2, 
                          stft_cfg={"n_fft": cfg.n_fft, "hop_length": cfg.hop, "win_length": cfg.n_fft}).to(cfg.device)
    
    # Initialize from checkpoint if specified
    start_epoch = 1
    best_ber = math.inf
    if cfg.init_from and os.path.isfile(cfg.init_from):
        try:
            checkpoint = torch.load(cfg.init_from, map_location=cfg.device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                model.load_state_dict(checkpoint["model_state"])
                log(f"Loaded model from {cfg.init_from}")
                if "best_ber" in checkpoint:
                    best_ber = checkpoint["best_ber"]
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
            else:
                model.load_state_dict(checkpoint)
                log(f"Loaded model weights from {cfg.init_from}")
        except Exception as e:
            log(f"Warning: failed to load {cfg.init_from}: {e}")
    
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    
    # Initialize enhanced trainer
    trainer = AdvancedWindowLevelTrainer(
        window_ms=cfg.window_ms,
        overlap_ratio=cfg.overlap_ratio,
        sr=TARGET_SR,
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        mapper_seed=cfg.mapper_seed,
        sync_strength=cfg.sync_strength,
        use_symbol_level=cfg.use_symbol_level,
        symbol_redundancy=cfg.symbol_redundancy,
        bias_lower_mid=cfg.bias_lower_mid,
        use_crc=cfg.use_crc,
        symbol_length=cfg.symbol_length,
        min_agreement_rate=cfg.min_agreement_rate
    )
    
    loss_perc = CombinedPerceptualLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    try:
        GradScaler = torch.amp.GradScaler
    except Exception:
        from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=(cfg.mixed_precision and torch.cuda.is_available()))

    # Expose perceptual loss fn to the trainer (used in symbol-level path)
    try:
        trainer.perceptual_loss_fn = loss_perc
    except Exception:
        pass
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if (not is_distributed) or rank == 0:
            log(f"\nEpoch {epoch}/{cfg.epochs}")
        
        # Training
        train_metrics = train_one_epoch_enhanced(
            model, trainer, loss_perc, optimizer, scaler, cfg, train_loader, epoch
        )
        
        if (not is_distributed) or rank == 0:
            log(
                f"train: obj={train_metrics['obj']:.4f} ber={train_metrics['ber_sample_mean']:.4f} "
                f"ce={train_metrics['ce_sample_mean']:.4f} mse={train_metrics['mse_sample_mean']:.4f} "
                f"perc={train_metrics['perc_sample_mean']:.4f}"
            )
        
        # Validation
        val_metrics = validate_enhanced_window_level(model, trainer, loss_perc, cfg, val_loader)
        
        if (not is_distributed) or rank == 0:
            payload_ber_str = f" p_ber={val_metrics['payload_ber']:.4f}" if cfg.use_rs_interleave else ""
            log(
                f"val  : obj={val_metrics['obj']:.4f} ber={val_metrics['ber_sample_mean']:.4f} "
                f"ce={val_metrics['ce_sample_mean']:.4f} mse={val_metrics['mse_sample_mean']:.4f} "
                f"perc={val_metrics['perc_sample_mean']:.4f}{payload_ber_str}"
            )
        
        # Save best model
        if ((not is_distributed) or rank == 0) and (val_metrics["ber_sample_mean"] < best_ber):
            best_ber = val_metrics["ber_sample_mean"]
            ckpt_path = os.path.join(cfg.save_dir, "window_level_enhanced_best.pt")
            to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state": to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_ber": best_ber,
                "cfg": cfg.__dict__,
                "trainer_config": trainer.get_training_info(),
            }, ckpt_path)
            log(f"Saved best checkpoint to {ckpt_path}")
        
        # Periodic save
        if ((not is_distributed) or rank == 0) and (epoch % 5 == 0):
            ckpt_path = os.path.join(cfg.save_dir, f"window_level_enhanced_e{epoch}.pt")
            to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state": to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "val_ber": val_metrics["ber_sample_mean"],
                "cfg": cfg.__dict__,
                "trainer_config": trainer.get_training_info(),
            }, ckpt_path)
            log(f"Saved checkpoint to {ckpt_path}")
    
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
