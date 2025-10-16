#!/usr/bin/env python3
"""
Phase-1 training outline for micro-windowed RS-coded payload recovery with INN.

Implements the requested pipeline skeleton:
- 1 s segmentation (self-contained seconds)
- Micro-windowing (10–20 ms, ~50% overlap)
- Adaptive STFT policy everywhere
- Deterministic time-frequency mapper with repetition r and spacing
- Psychoacoustic gate and amplitude budget per window
- Per-second sync marker embedding
- Encode (INN) and train-time decode (INN)
- Losses: bit BCE, amplitude penalty, perceptual composite
- Metrics: BER (window and symbol), RS payload success, perceptual

Note: This is an initial scaffold that relies on existing pipeline components
(e.g., INNWatermarker, perceptual losses). Mapper/gate implementations here
provide deterministic behavior and masks consistent with Phase-1 assumptions.
"""

from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import warnings
# Prefer legacy backends to avoid TorchCodec dependency unless explicitly installed
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from models.inn_encoder_decoder import INNWatermarker
from pipeline.perceptual_losses import CombinedPerceptualLoss, MFCCCosineLoss
from pipeline.moore_galsberg import MooreGlasbergAnalyzer
from pipeline.adaptive_bit_allocation import (
	PerceptualSignificanceMetric,
	AdaptiveBitAllocator,
	expand_allocation_to_slots,
)

# Select a backend that does not require TorchCodec
try:
	torchaudio.set_audio_backend("sox_io")
except Exception:
	pass

# Silence known TorchAudio warnings (2.9 migration and deprecations)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*use torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# Required RS utilities and EULDriver
from pipeline.ingest_and_chunk import (
	rs_encode_167_125,
	rs_decode_167_125,
	interleave_bytes,
	deinterleave_bytes,
	EULDriver,
)

# =========================
# Config
# =========================

TARGET_SR = 44100
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)


@dataclass
class TrainConfig:
	# Data
	data_dir: str = "data/train"
	val_dir: str = "data/val"
	batch_size: int = 8
	num_workers: int = 6
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	mixed_precision: bool = True
	log_interval: int = 25
	save_dir: str = "new_pipeline_test"

	# Micro-windowing
	window_ms: int = 20           # 10–20 ms (use 20 ms default)
	hop_ms: int = 10               # ~50% overlap

	# Adaptive STFT policy (applied everywhere)
	# Will be derived from window length dynamically; these are upper bounds
	n_fft_max: int = 1024

	# Payload
	rs_payload_bytes: int = 64     # per second
	repetition: int = 3            # r placements per symbol (redundancy)
	use_fixed_payload: bool = True # Phase-1 may use fixed payload for simplicity
	payload_seed: int = 420
	payload_variation: str = "per_epoch"  # "per_epoch", "per_batch", "per_sample"
	
	# EULDriver configuration
	interleave_depth: int = 4      # interleave depth for RS
	base_symbol_amp: float = 0.12  # base +/- amplitude per placement
	amp_safety: float = 1.0        # amplitude safety factor

	# Mapper and gating
	mapper_seed: int = 42
	min_time_spacing: int = 2      # min frames between placements for same symbol
	min_freq_spacing: int = 4      # min bins between placements for same symbol
	psy_mask_margin: float = 1.0   # margin below threshold to consider safe
	amp_budget_scale: float = 0.25 # fraction of masking margin per window

	# Sync
	sync_strength: float = 0.05    # faint, masked sync per second

	# Loss weights
	w_byte: float = 1.0             # byte-level accuracy loss (primary objective)
	w_amp: float = 0.0
	w_perc: float = 0.01            # start at 0.0 for bring-up; increase later
	# Early-easy settings and thresholds
	early_amp_budget_scale: float = 0.5  # use 0.4–0.6 initially
	early_psy_mask_margin: float = 1.8   # relax mask in epochs 1–2
	capacity_warn_threshold: float = 0.35
	warmup_perc_epochs: int = 2          # disable perceptual first N epochs

	# Optim
	lr: float = 9e-5
	peak_lr: float = 1e-4
	min_lr: float = 3e-5
	warmup_steps: int = 2000
	# Plateau and LR bump
	plateau_eps: float = 1e-3
	plateau_patience_epochs: int = 1
	lr_bump_mult: float = 1.5
	weight_decay: float = 1e-5
	epochs: int = 20

	# Logging
	log_file: str = "phase1_train_log.txt"

	# Dataset limits (optional)
	max_train_files: Optional[int] = None
	max_val_files: Optional[int] = None


# =========================
# Data
# =========================

def _list_audio_files(root: str) -> List[str]:
	exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
	files: List[str] = []
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if os.path.splitext(fn.lower())[1] in exts:
				files.append(os.path.join(dirpath, fn))
	return files


class OneSecondDataset(Dataset):
	def __init__(self, root: str, target_sr: int = TARGET_SR, max_files: Optional[int] = None):
		self.files = _list_audio_files(root)
		# Optionally limit dataset size to speed up experiments
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
			# Try forcing sox_io backend if available
			try:
				torchaudio.set_audio_backend("sox_io")
				wav, sr = torchaudio.load(path)
			except Exception as backend_err:
				# Optional fallback via soundfile if installed
				try:
					import soundfile as sf
					data, sr = sf.read(path, dtype="float32", always_2d=True)
					wav = torch.from_numpy(data.T)
				except Exception as sf_err:
					raise ImportError(
						"Audio loading requires either torchcodec installed, a working sox_io backend, or soundfile."
					) from sf_err
		if wav.size(0) > 1:
			wav = wav.mean(dim=0, keepdim=True)
		if sr != self.target_sr:
			wav = Resample(orig_freq=sr, new_freq=self.target_sr)(wav)
		# Do not normalize: preserve original mastering loudness and dynamics
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
# Micro-windowing and adaptive STFT helpers
# =========================

def ms_to_samples(ms: int, sr: int) -> int:
	return int(round(ms * 1e-3 * sr))


def adaptive_stft(x: torch.Tensor) -> torch.Tensor:
	"""Adaptive STFT used everywhere: for a given micro-window length T:
	- win_length = T
	- n_fft = next_pow2(T)
	- hop_length ≈ T/2 (preserve 50% overlap)
	- center=True with reflect padding only if n_fft/2 < T; else center=False
	Returns RI channels [B, 2, F, T_frames].
	"""
	if x.dim() == 2:  # [1, T]
		x = x.unsqueeze(0)
	elif x.dim() == 3 and x.size(1) == 1:
		pass
	else:
		raise ValueError(f"adaptive_stft expects [1,T] or [B,1,T], got {x.shape}")
	B, _, T = x.shape
	win_length = int(T)
	n_fft = 1 << (max(1, win_length) - 1).bit_length()
	hop_length = max(1, int(round(win_length * 0.5)))
	center = True
	pad_mode = "reflect"
	if (n_fft // 2) >= win_length:
		center = False
		pad_mode = "constant"
	window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
	X = torch.stft(
		x.squeeze(1),
		n_fft=n_fft,
		hop_length=hop_length,
		win_length=win_length,
		window=window,
		center=center,
		pad_mode=pad_mode,
		return_complex=True,
	)
	return torch.stack([X.real, X.imag], dim=1)






# =========================
# Sync marker (per second)
# =========================

@torch.no_grad()
def embed_sync_marker(x_1s: torch.Tensor, strength: float, sr: int, seed: int) -> torch.Tensor:
	"""Add a faint 1 Hz-like masked component by low-frequency sinusoid gated in time.
	This is a placeholder; Phase-2 can refine for alignment.
	"""
	B, C, T = 1, 1, x_1s.size(-1)
	t = torch.linspace(0, T / sr, steps=T, device=x_1s.device, dtype=x_1s.dtype)
	# Use a very low frequency tone and envelope
	freq = 1.0
	env = torch.hann_window(T, device=x_1s.device, dtype=x_1s.dtype)
	sync = (torch.sin(2 * math.pi * freq * t) * env).unsqueeze(0).unsqueeze(0).to(dtype=x_1s.dtype)
	out = (x_1s + strength * sync)
	return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


# =========================
# RS payload helpers
# =========================

def _rand_payload_bytes(n: int, rng: random.Random) -> bytes:
	return bytes([rng.randrange(0, 256) for _ in range(n)])


def _generate_structured_payload(rng: random.Random) -> bytes:
	"""Generate a structured 64-byte payload with variable content.
	
	Structure: "ISRC{12_digits}ISFR{12_digits}N{name_8_chars}D{4_digits}"
	- ISRC: 12 random digits
	- ISFR: 12 random digits  
	- N: 8-character name (letters/numbers)
	- D: 4-digit duration
	
	Total: 4 + 12 + 4 + 12 + 1 + 8 + 1 + 4 = 46 bytes
	Remaining 18 bytes: padded with random alphanumeric
	"""
	# ISRC: 12 random digits
	isrc = ''.join([str(rng.randint(0, 9)) for _ in range(12)])
	
	# ISFR: 12 random digits
	isfr = ''.join([str(rng.randint(0, 9)) for _ in range(12)])
	
	# Name: 8 random alphanumeric characters
	name_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
	name = ''.join([rng.choice(name_chars) for _ in range(8)])
	
	# Duration: 4 random digits (0000-9999)
	duration = ''.join([str(rng.randint(0, 9)) for _ in range(4)])
	
	# Construct payload
	payload_str = f"ISRC{isrc}ISFR{isfr}N{name}D{duration}"
	
	# Pad to 64 bytes with random alphanumeric
	while len(payload_str) < 64:
		payload_str += rng.choice(name_chars)
	
	# Truncate to exactly 64 bytes
	payload_str = payload_str[:64]
	
	return payload_str.encode('utf-8')


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
	"""Return (bits_tensor, gt_payload_bytes, coded_bytes).
	- bits_tensor: full coded bitstream (LSB-first) as torch.LongTensor
	- gt_payload_bytes: original (pre-RS) 64 bytes
	- coded_bytes: RS/interleaved bytes actually embedded
	"""
	rng = random.Random(cfg.payload_seed)
	if cfg.use_fixed_payload:
		payload = _generate_structured_payload(rng)
	else:
		payload = _rand_payload_bytes(cfg.rs_payload_bytes, rng)
	
	# Always use RS encoding + interleaving
	coded = interleave_bytes(rs_encode_167_125(payload), cfg.interleave_depth)
	bits = _bytes_to_bits_lsb_first(coded)
	return torch.tensor(bits, dtype=torch.long), payload, coded


# =========================
# Encode/decode path and loss
# =========================



def compute_losses_and_metrics(
	model: INNWatermarker,
	x_1s: torch.Tensor,
	cfg: TrainConfig,
	payload_bits: torch.Tensor,
	epoch: int = 0,
) -> Dict:
	"""Full forward for one 1 s segment using EULDriver for encode/decode."""
	# Unwrap DDP if present
	base_model = getattr(model, "module", model)
	
	# Sanitize inputs early
	x_1s = torch.nan_to_num(x_1s, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
	
	# Add sync marker
	x_sync = embed_sync_marker(x_1s, cfg.sync_strength, TARGET_SR, cfg.mapper_seed)
	
	# Create EULDriver instance with correct configuration
	eul_driver = EULDriver(
		sr=TARGET_SR,
		n_fft=882,  # Match model's STFT config
		hop=441,
		rs_interleave=cfg.interleave_depth,
		per_eul_bits_target=cfg.rs_payload_bytes * 8,  # Target bits for 64-byte payload
		base_symbol_amp=cfg.base_symbol_amp,
		amp_safety=cfg.amp_safety
	)
	
	# Get payload bytes from bits (reverse the bit conversion)
	payload_bytes = _bits_to_bytes_lsb_first(payload_bits.tolist())
	
	# Encode using EULDriver
	x_wm = eul_driver.encode_eul(base_model, x_sync, payload_bytes)
	x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
	
	# Decode using EULDriver (this will recompute slots from received audio)
	decoded_bytes = eul_driver.decode_eul(base_model, x_wm, expected_bytes=cfg.rs_payload_bytes)
	
	# Get original payload bytes for comparison
	original_payload_bytes = _bits_to_bytes_lsb_first(payload_bits.tolist())
	
	# Amplitude penalty (simplified - EULDriver handles amplitude allocation)
	amp_penalty = torch.tensor(0.0, device=x_1s.device)
	
	# Perceptual loss
	if x_sync.dtype != torch.float32:
		x_sync_fp = x_sync.float()
	else:
		x_sync_fp = x_sync
	if x_wm.dtype != torch.float32:
		x_wm_fp = x_wm.float()
	else:
		x_wm_fp = x_wm
		
	with torch.amp.autocast(device_type="cuda", enabled=False):
		perc = CombinedPerceptualLoss(mfcc=MFCCCosineLoss(sample_rate=TARGET_SR))(x_sync_fp, x_wm_fp)
		perc_total = perc["total_perceptual_loss"]
	
	# Byte-level and payload-level metrics
	with torch.no_grad():
		# Byte-level accuracy: compare decoded bytes with original payload bytes
		byte_acc = 0.0
		if len(decoded_bytes) > 0 and len(original_payload_bytes) > 0:
			L = min(len(decoded_bytes), len(original_payload_bytes))
			if L > 0:
				byte_acc = float(sum(1 for i in range(L) if decoded_bytes[i] == original_payload_bytes[i]) / L)
		
		# End-to-end payload success: perfect match of all bytes
		payload_success = 1.0 if decoded_bytes == original_payload_bytes else 0.0
		
		# Byte-level loss for optimization (1 - accuracy, so we minimize this)
		byte_loss = 1.0 - byte_acc
	
	# Total loss: focus on byte-level accuracy + perceptual quality
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
# Train/val loop
# =========================

def _build_model(cfg: TrainConfig) -> INNWatermarker:
	# Fixed STFT grid at 44.1 kHz: n_fft=882, hop=441, win=882
	# This matches EULDriver configuration for consistency
	return INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 882, "hop_length": 441, "win_length": 882})


def _make_payload_bits_tensor(cfg: TrainConfig, device, epoch: int = 0, batch_idx: int = 0) -> torch.Tensor:
	"""Create payload bits with configurable variation strategy (bits only)."""
	if cfg.payload_variation == "per_sample":
		# Most aggressive: different payload for each sample
		seed = cfg.payload_seed + epoch * 10000 + batch_idx * 100 + random.randint(0, 99)
	elif cfg.payload_variation == "per_batch":
		# Moderate: different payload for each batch
		seed = cfg.payload_seed + epoch * 1000 + batch_idx
	elif cfg.payload_variation == "per_epoch":
		# Conservative: different payload for each epoch
		seed = cfg.payload_seed + epoch
	else:
		# Fallback to original behavior
		seed = cfg.payload_seed
	
	# Create temporary config with modified seed
	temp_cfg = TrainConfig()
	for attr in dir(cfg):
		if not attr.startswith('_'):
			setattr(temp_cfg, attr, getattr(cfg, attr))
	temp_cfg.payload_seed = seed
	
	bits, _, _ = build_payload_bits_and_bytes(temp_cfg)
	return bits.to(device)


def _make_payload_bits_and_bytes(cfg: TrainConfig, device, epoch: int = 0, batch_idx: int = 0) -> Tuple[torch.Tensor, bytes, bytes]:
	"""Create bits plus GT/original bytes and coded bytes, matching training variation."""
	if cfg.payload_variation == "per_sample":
		seed = cfg.payload_seed + epoch * 10000 + batch_idx * 100 + random.randint(0, 99)
	elif cfg.payload_variation == "per_batch":
		seed = cfg.payload_seed + epoch * 1000 + batch_idx
	elif cfg.payload_variation == "per_epoch":
		seed = cfg.payload_seed + epoch
	else:
		seed = cfg.payload_seed
	# Clone cfg with modified seed
	temp_cfg = TrainConfig()
	for attr in dir(cfg):
		if not attr.startswith('_'):
			setattr(temp_cfg, attr, getattr(cfg, attr))
	temp_cfg.payload_seed = seed
	bits, gt_payload, coded = build_payload_bits_and_bytes(temp_cfg)
	return bits.to(device), gt_payload, coded


def validate(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader) -> Dict:
	model.eval()
	base_model = getattr(model, "module", model)
	metrics = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
	local_samples = 0
	with torch.no_grad():
		for batch_idx, batch in enumerate(loader):
			x = batch.to(cfg.device, non_blocking=True)
			bits = _make_payload_bits_tensor(cfg, x.device, epoch=0, batch_idx=batch_idx)
			out = compute_losses_and_metrics(base_model, x, cfg, bits, epoch=0)
			metrics["loss"] += float(out["loss"].detach().item()) * x.size(0)
			metrics["byte_loss"] += float(out["byte_loss"].detach().item()) * x.size(0)
			metrics["perc"] += float(out["perc"].detach().item()) * x.size(0)
			metrics["byte_acc"] += float(out.get("byte_acc", torch.tensor(0.0)).detach().item()) * x.size(0)
			metrics["payload_ok"] += float(out.get("payload_ok", torch.tensor(0.0)).detach().item()) * x.size(0)
			local_samples += int(x.size(0))
	# All-reduce across ranks for true global averages
	if dist.is_available() and dist.is_initialized():
		vals = torch.tensor([metrics["loss"], metrics["byte_loss"], metrics["perc"], metrics["byte_acc"], metrics["payload_ok"]], device=cfg.device, dtype=torch.float64)
		cnt = torch.tensor([float(local_samples)], device=cfg.device, dtype=torch.float64)
		dist.all_reduce(vals, op=dist.ReduceOp.SUM)
		dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
		global_samples = max(1.0, cnt.item())
		metrics["loss"], metrics["byte_loss"], metrics["perc"], metrics["byte_acc"], metrics["payload_ok"] = [float(v/global_samples) for v in vals.tolist()]
	else:
		global_samples = max(1, local_samples)
		for k in metrics:
			metrics[k] = metrics[k] / global_samples
	return metrics


def train_one_epoch(model: INNWatermarker, cfg: TrainConfig, optimizer: torch.optim.Optimizer, loader: DataLoader, scaler, epoch: int) -> Dict:
	model.train()
	base_model = getattr(model, "module", model)
	running = {"loss": 0.0, "byte_loss": 0.0, "perc": 0.0, "byte_acc": 0.0, "payload_ok": 0.0}
	local_samples = 0

	# Per-step LR scheduler with warmup then cosine decay over the whole run
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
		bits = _make_payload_bits_tensor(cfg, x.device, epoch=epoch, batch_idx=step)
		optimizer.zero_grad(set_to_none=True)
		# LR warmup and cosine
		if hasattr(cfg, "warmup_steps"):
			if cfg._global_step < cfg.warmup_steps:
				# linear warmup to peak_lr
				lr_now = float(cfg.lr) + (float(cfg.peak_lr) - float(cfg.lr)) * (cfg._global_step / max(1, cfg.warmup_steps))
			else:
				# cosine decay to min_lr over the rest of the total training schedule
				denom = max(1, int(cfg._total_steps_planned) - int(cfg.warmup_steps))
				progress = min(1.0, max(0.0, (cfg._global_step - int(cfg.warmup_steps)) / denom))
				cos = 0.5 * (1.0 + math.cos(math.pi * progress))
				lr_now = float(cfg.min_lr) + (float(cfg.peak_lr) - float(cfg.min_lr)) * cos
			for pg in optimizer.param_groups:
				pg["lr"] = lr_now
		use_amp = cfg.mixed_precision and torch.cuda.is_available()
		# Keep compute stable: do heavy math under AMP but sanitize outputs
		with torch.amp.autocast(device_type="cuda", enabled=use_amp):
			out = compute_losses_and_metrics(base_model, x, cfg, bits, epoch=epoch)
			loss = out["loss"]
		# Skip non-finite losses
		if not torch.isfinite(loss):
			continue
		if scaler is not None and use_amp:
			scaler.scale(loss).backward()
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			scaler.step(optimizer)
			scaler.update()
		else:
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
	# All-reduce across ranks for true global averages
	if dist.is_available() and dist.is_initialized():
		vals = torch.tensor([running["loss"], running["byte_loss"], running["perc"], running["byte_acc"], running["payload_ok"]], device=cfg.device, dtype=torch.float64)
		cnt = torch.tensor([float(local_samples)], device=cfg.device, dtype=torch.float64)
		dist.all_reduce(vals, op=dist.ReduceOp.SUM)
		dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
		global_samples = max(1.0, cnt.item())
		running["loss"], running["byte_loss"], running["perc"], running["byte_acc"], running["payload_ok"] = [float(v/global_samples) for v in vals.tolist()]
	else:
		global_samples = max(1, local_samples)
		for k in running:
			running[k] = running[k] / global_samples
	return running


def main(cfg: TrainConfig) -> None:
	os.makedirs(cfg.save_dir, exist_ok=True)
	# DDP init (gloo on Windows)
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
	# Dataset & loaders
	train_ds = OneSecondDataset(cfg.data_dir, TARGET_SR, max_files=getattr(cfg, "max_train_files", None))
	val_ds = OneSecondDataset(cfg.val_dir, TARGET_SR, max_files=getattr(cfg, "max_val_files", None))
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
	# Model
	model = _build_model(cfg).to(cfg.device)
	# Resume logic
	start_epoch = 1
	best_byte_acc = 0.0
	if hasattr(cfg, "resume") and isinstance(cfg.resume, str) and os.path.isfile(cfg.resume):
		try:
			ckpt = torch.load(cfg.resume, map_location=cfg.device)
			state = ckpt.get("model_state", ckpt)
			model.load_state_dict(state, strict=False)
			if "best_byte_acc" in ckpt:
				best_byte_acc = float(ckpt["best_byte_acc"])  # carry forward best metric if available
			if "epoch" in ckpt:
				start_epoch = int(ckpt["epoch"]) + 1
			# optimizer/scaler will be restored after creation below
			_resume_blob = ckpt  # stash for later
			if (not is_distributed) or rank == 0:
				print(f"Resumed model weights from {cfg.resume} (start_epoch={start_epoch})")
		except Exception as e:
			if (not is_distributed) or rank == 0:
				print(f"Warning: failed to load resume checkpoint {cfg.resume}: {e}")
	if is_distributed:
		model = DDP(
			model,
			device_ids=[local_rank] if torch.cuda.is_available() else None,
			output_device=local_rank if torch.cuda.is_available() else None,
			broadcast_buffers=False,
			find_unused_parameters=False,
		)
	# Optimizer & scaler
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
	try:
		GradScaler = torch.amp.GradScaler  # type: ignore[attr-defined]
	except Exception:
		from torch.cuda.amp import GradScaler  # type: ignore
	scaler = GradScaler(enabled=(cfg.mixed_precision and torch.cuda.is_available()))
	# Restore optimizer/scaler if available in resume
	if "_resume_blob" in locals():
		try:
			if _resume_blob.get("optimizer_state") is not None:
				optimizer.load_state_dict(_resume_blob["optimizer_state"])
			if _resume_blob.get("scaler_state") is not None and scaler is not None:
				scaler.load_state_dict(_resume_blob["scaler_state"])  # type: ignore[arg-type]
		except Exception:
			pass
	# Logger
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
		log(f"Files: train={len(train_ds)} | val={len(val_ds)} | SR={TARGET_SR} | win={cfg.window_ms}ms hop={cfg.hop_ms}ms r={cfg.repetition}")
	# Epochs
	best_byte_acc = 0.0
	for epoch in range(start_epoch, cfg.epochs + 1):
		if is_distributed and train_sampler is not None:
			train_sampler.set_epoch(epoch)
		if (not is_distributed) or rank == 0:
			log(f"\nEpoch {epoch}/{cfg.epochs}")
		train_metrics = train_one_epoch(model, cfg, optimizer, train_loader, scaler, epoch)
		if (not is_distributed) or rank == 0:
			log(f"train: loss={train_metrics['loss']:.4f} byte_loss={train_metrics['byte_loss']:.4f} perc={train_metrics['perc']:.4f} byte_acc={train_metrics.get('byte_acc',0.0):.3f} payload_ok={train_metrics.get('payload_ok',0.0):.3f}")
		val_metrics = validate(model, cfg, val_loader)
		if (not is_distributed) or rank == 0:
			log(f"val  : loss={val_metrics['loss']:.4f} byte_loss={val_metrics['byte_loss']:.4f} perc={val_metrics['perc']:.4f} byte_acc={val_metrics.get('byte_acc',0.0):.3f} payload_ok={val_metrics.get('payload_ok',0.0):.3f}")
			# LR bump on plateau: if no improvement > epsilon over patience, bump LR briefly
			if "_last_val_loss" not in locals():
				_last_val_loss = val_metrics["loss"]
				_stagnant_epochs = 0
			else:
				if abs(val_metrics["loss"] - _last_val_loss) < cfg.plateau_eps:
					_stagnant_epochs += 1
				else:
					_stagnant_epochs = 0
				_last_val_loss = val_metrics["loss"]
			if _stagnant_epochs >= cfg.plateau_patience_epochs:
				for pg in optimizer.param_groups:
					pg["lr"] = min(cfg.peak_lr * cfg.lr_bump_mult, pg["lr"] * cfg.lr_bump_mult)
				log(f"LR bump applied due to plateau (new lr ~ {optimizer.param_groups[0]['lr']:.2e})")
			# Save best by byte accuracy
			if val_metrics["byte_acc"] > best_byte_acc:
				best_byte_acc = val_metrics["byte_acc"]
				ckpt_path = os.path.join(cfg.save_dir, "phase1_best.pt")
				to_save = model.module if hasattr(model, "module") else model
				torch.save({
					"epoch": epoch,
					"model_state": to_save.state_dict(),
					"optimizer_state": optimizer.state_dict(),
					"best_byte_acc": best_byte_acc,
					"cfg": cfg.__dict__,
				}, ckpt_path)
				log(f"Saved best checkpoint to {ckpt_path} (byte_acc={best_byte_acc:.3f})")
	if dist.is_available() and dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	# Optional CLI/env overrides so you can do:
	# python training_new.py --batch_size 8 --epochs 10
	parser = argparse.ArgumentParser(description="Phase-1 INN training (DDP-ready)")
	parser.add_argument("--data_dir", type=str, default=None)
	parser.add_argument("--val_dir", type=str, default=None)
	parser.add_argument("--save_dir", type=str, default=None)
	parser.add_argument("--epochs", type=int, default=None)
	parser.add_argument("--batch_size", type=int, default=None, help="Per-process batch size")
	parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume from")
	parser.add_argument("--max_train_files", type=int, default=None, help="Limit number of training files")
	parser.add_argument("--max_val_files", type=int, default=None, help="Limit number of validation files")
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
	# Attach resume path if provided
	if args.resume:
		setattr(cfg, "resume", args.resume)
	main(cfg)
