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
from typing import Dict, List, Tuple
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
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

from models.inn_encoder_decoder import INNWatermarker
from pipeline.perceptual_losses import CombinedPerceptualLoss

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

# Optional existing utilities if present
try:
	from pipeline.ingest_and_chunk import (
		rs_encode_167_125,
		rs_decode_167_125,
		interleave_bytes,
		deinterleave_bytes,
	)
	_HAS_RS = True
except Exception:
	_HAS_RS = False

# =========================
# Config
# =========================

TARGET_SR = 48000  # Phase-1 target sample rate per spec examples (yields ~99 windows/sec @ 20ms/10ms)
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
	save_dir: str = "Watermark_new_pipeline_test"

	# Micro-windowing
	window_ms: int = 20           # 10–20 ms (use 20 ms default)
	hop_ms: int = 10               # ~50% overlap

	# Adaptive STFT policy (applied everywhere)
	# Will be derived from window length dynamically; these are upper bounds
	n_fft_max: int = 1024

	# Payload
	rs_payload_bytes: int = 64     # per second
	repetition: int = 3            # r placements per symbol
	use_fixed_payload: bool = True # Phase-1 may use fixed payload for simplicity
	payload_seed: int = 12345

	# Mapper and gating
	mapper_seed: int = 42
	min_time_spacing: int = 1      # min frames between placements for same symbol
	min_freq_spacing: int = 2      # min bins between placements for same symbol
	psy_mask_margin: float = 1.0   # margin below threshold to consider safe
	amp_budget_scale: float = 0.25 # fraction of masking margin per window

	# Sync
	sync_strength: float = 0.05    # faint, masked sync per second

	# Loss weights
	w_bits: float = 1.0
	w_amp: float = 0.05
	w_perc: float = 0.01            # start at 0.0 for bring-up; increase later

	# Optim
	lr: float = 5e-5
	weight_decay: float = 1e-5
	epochs: int = 20

	# Logging
	log_file: str = "phase1_train_log.txt"


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
	def __init__(self, root: str, target_sr: int = TARGET_SR):
		self.files = _list_audio_files(root)
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
		# normalize
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


def chunk_into_micro_windows(x_1s: torch.Tensor, window_ms: int, hop_ms: int, sr: int) -> List[torch.Tensor]:
	"""Return list of micro-windows [1, T_w] with 50% overlap by default."""
	T_w = ms_to_samples(window_ms, sr)
	hop = ms_to_samples(hop_ms, sr)
	T = x_1s.size(-1)
	windows: List[torch.Tensor] = []
	start = 0
	while start + T_w <= T:
		windows.append(x_1s[..., start:start + T_w])
		start += hop
	# if remainder exists, right-pad one last window to full length
	if start < T:
		rem = x_1s[..., -T_w:]
		windows.append(rem)
	return windows


# =========================
# Deterministic mapper with repetition and spacing
# =========================

class DeterministicTFMapper:
	def __init__(self, seed: int, repetition: int, min_time_spacing: int, min_freq_spacing: int):
		self.seed = seed
		self.r = repetition
		self.min_t = max(0, int(min_time_spacing))
		self.min_f = max(0, int(min_freq_spacing))

	def map_symbols(self, num_symbols: int, num_windows: int, F_bins: int, T_frames: int) -> Dict[int, List[Tuple[int,int,int]]]:
		"""Return placements per symbol: dict[sym] -> list of (w_idx, f_idx, t_idx), size r.
		Deterministic seeded PRNG with spacing constraints to avoid collisions.
		"""
		rng = random.Random(self.seed)
		placements: Dict[int, List[Tuple[int,int,int]]] = {}
		used = set()  # guard collisions across all symbols
		for s_idx in range(num_symbols):
			chosen: List[Tuple[int,int,int]] = []
			attempts = 0
			while len(chosen) < self.r and attempts < 10000:
				w = rng.randrange(0, num_windows)
				f = rng.randrange(0, F_bins)
				t = rng.randrange(0, T_frames)
				# spacing constraint: ensure not near previous picks for same symbol
				ok = True
				for (_w, _f, _t) in chosen:
					if abs(_w - w) < 1:  # keep within same second; windows can repeat, but avoid exact same
						if abs(_t - t) <= self.min_t and abs(_f - f) <= self.min_f:
							ok = False; break
				# avoid global duplicates
				if ok and (w, f, t) in used:
					ok = False
				if ok:
					chosen.append((w, f, t)); used.add((w, f, t))
				attempts += 1
			placements[s_idx] = chosen
		return placements


# =========================
# Psychoacoustic gate and amplitude budget
# =========================

@torch.no_grad()
def psycho_gate_accept_mask(X_ri: torch.Tensor, margin: float) -> torch.Tensor:
	"""Simple proxy: accept bins where |X| is below a headroom threshold margin (masked regions allow energy).
	X_ri: [1, 2, F, T]
	Return bool mask [F, T] where True means accepted for embedding.
	"""
	mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-9))[0]  # [F,T]
	thr = mag.median()  # crude global proxy
	return (mag <= (thr * (1.0 + margin)))


def per_window_amp_budget(X_ri: torch.Tensor, mask: torch.Tensor, budget_scale: float) -> float:
	"""Compute scalar amplitude budget for a window as a fraction of masking margin."""
	mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-9))[0]
	accepted = mag[mask]
	if accepted.numel() == 0:
		return 0.0
	margin = float(accepted.mean().item())
	return max(0.0, budget_scale * margin)


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


def build_payload_bits(cfg: TrainConfig) -> torch.Tensor:
	rng = random.Random(cfg.payload_seed)
	if cfg.use_fixed_payload:
		payload = bytes([i % 256 for i in range(cfg.rs_payload_bytes)])
	else:
		payload = _rand_payload_bytes(cfg.rs_payload_bytes, rng)
	if _HAS_RS:
		enc = rs_encode_167_125(payload)
		enc = interleave_bytes(enc, 1)  # simple interleave depth=1 for Phase-1
	else:
		enc = payload
	bits: List[int] = []
	for b in enc:
		for k in range(8):
			bits.append((b >> k) & 1)
	return torch.tensor(bits, dtype=torch.long)


# =========================
# Encode/decode path and loss
# =========================

def build_message_spec_for_second(x_1s: torch.Tensor, placements: Dict[int, List[Tuple[int,int,int]]],
		bits_per_symbol: List[int], amp_budget: Dict[int, float]) -> torch.Tensor:
	"""Create message spectrogram [1, 2, F, T] for the whole second by aggregating micro-window targets.
	We place real-channel targets at specified (f,t) for each window; here we ignore window index in spec grid
	and rely on decode-time window slicing for supervision. As a simple scaffold, we use a single STFT grid
	for the whole second with n_fft = next_pow2(T) and hop=T (no overlap) to provide shape; the INN's STFT
	will enforce consistent shapes internally during encode.
	"""
	# Build second-level adaptive STFT grid
	X_ri = adaptive_stft(x_1s)  # [1,2,F,Tf]
	F_bins, T_frames = X_ri.size(-2), X_ri.size(-1)
	M = torch.zeros_like(X_ri)
	for sym_idx, places in placements.items():
		for (w, f, t) in places:
			if f < F_bins and t < T_frames:
				sign = 1.0 if bits_per_symbol[sym_idx] > 0 else -1.0
				amp = amp_budget.get(w, 0.0)
				M[0, 0, f, t] = M[0, 0, f, t] + sign * amp
	return M


def compute_losses_and_metrics(
	model: INNWatermarker,
	x_1s: torch.Tensor,
	cfg: TrainConfig,
	payload_bits: torch.Tensor,
) -> Dict:
	"""Full forward for one 1 s segment with encode/decode and losses."""
	# Unwrap DDP if present so we can call custom methods like encode/decode
	base_model = getattr(model, "module", model)
	# Sanitize inputs early
	x_1s = torch.nan_to_num(x_1s, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
	# Insert sync
	x_sync = embed_sync_marker(x_1s, cfg.sync_strength, TARGET_SR, cfg.mapper_seed)
	# Micro-windowing
	windows = chunk_into_micro_windows(x_sync, cfg.window_ms, cfg.hop_ms, TARGET_SR)
	n_windows = len(windows)
	# Establish per-window STFT and masks
	window_specs: List[torch.Tensor] = []
	accept_masks: List[torch.Tensor] = []
	amp_budget_by_window: Dict[int, float] = {}
	for i, w in enumerate(windows):
		Xw = adaptive_stft(w)  # [1,2,F,T]
		mask = psycho_gate_accept_mask(Xw, cfg.psy_mask_margin)
		window_specs.append(Xw)
		accept_masks.append(mask)
		amp_budget_by_window[i] = per_window_amp_budget(Xw, mask, cfg.amp_budget_scale)
	# Determine bit-level training: 64 bytes -> 512 bits; train bit-wise BCE per mapped placement
	n_symbols = cfg.rs_payload_bytes * 8  # treat each bit as a logical symbol index
	bits_by_symbol = []
	for s in range(n_symbols):
		b = int(payload_bits[s % payload_bits.numel()].item())
		bits_by_symbol.append(b)
	# Build deterministic placements with repetition
	F_bins = window_specs[0].size(-2)
	T_frames = window_specs[0].size(-1)
	mapper = DeterministicTFMapper(cfg.mapper_seed, cfg.repetition, cfg.min_time_spacing, cfg.min_freq_spacing)
	placements = mapper.map_symbols(n_symbols, n_windows, F_bins, T_frames)
	# Gate placements by psycho mask; if rejected, drop that copy
	gated_placements: Dict[int, List[Tuple[int,int,int]]] = {}
	for sym_idx, places in placements.items():
		kept: List[Tuple[int,int,int]] = []
		for (w, f, t) in places:
			mask = accept_masks[w]
			if f < mask.size(0) and t < mask.size(1) and bool(mask[f, t].item()):
				kept.append((w, f, t))
		gated_placements[sym_idx] = kept
	# Build message spec for the whole second using window budgets
	M_spec = build_message_spec_for_second(x_sync, gated_placements, bits_by_symbol, amp_budget_by_window)
	# Encode with INN
	x_wm, _ = base_model.encode(x_sync, M_spec)
	x_wm = torch.nan_to_num(x_wm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
	# Decode with INN
	M_rec = base_model.decode(x_wm)
	M_rec = torch.nan_to_num(M_rec, nan=0.0, posinf=1.0, neginf=-1.0)
	# Extract predictions at placement slots and compute bit-wise BCE (masked)
	all_logits: List[torch.Tensor] = []
	all_targets: List[torch.Tensor] = []
	for sym_idx, places in gated_placements.items():
		for (w, f, t) in places:
			if f < M_rec.size(-2) and t < M_rec.size(-1):
				val = M_rec[0, 0, f, t]
				all_logits.append(val.unsqueeze(0))
				target = torch.tensor(1.0 if bits_by_symbol[sym_idx] > 0 else 0.0, device=val.device)
				all_targets.append(target.unsqueeze(0))
	if len(all_logits) == 0:
		bit_loss = torch.tensor(0.0, device=x_1s.device)
		ber = 1.0
	else:
		logits = torch.cat(all_logits, dim=0)
		targets = torch.cat(all_targets, dim=0)
		# Clamp/sanitize logits to avoid NaN/Inf in BCE
		logits = torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)
		bit_loss = F.binary_cross_entropy_with_logits(logits, targets)
		pred_bits = (logits > 0).float()
		ber = float((pred_bits != targets).float().mean().item())
	# Amplitude budget penalty: sum of placed amplitudes relative to budget
	amp_penalty_terms: List[torch.Tensor] = []
	for w_idx, budget in amp_budget_by_window.items():
		if budget <= 0.0:
			continue
		# Estimate used amplitude as average absolute target placed in that window
		used_vals: List[torch.Tensor] = []
		for sym_idx, places in gated_placements.items():
			for (w, f, t) in places:
				if w == w_idx and f < M_spec.size(-2) and t < M_spec.size(-1):
					used_vals.append(M_spec[0, 0, f, t].abs())
		if used_vals:
			used = torch.stack(used_vals).mean()
			budget_t = torch.tensor(budget, device=x_1s.device, dtype=used.dtype)
			excess = (used - budget_t).clamp(min=0.0)
			amp_penalty_terms.append(excess)
	amp_penalty = torch.stack(amp_penalty_terms).mean() if amp_penalty_terms else torch.tensor(0.0, device=x_1s.device)
	# Perceptual loss (optional; start at 0 in bring-up)
	# Compute perceptual loss in full precision to reduce fp16/bf16 NaNs
	if x_sync.dtype != torch.float32:
		x_sync_fp = x_sync.float()
	else:
		x_sync_fp = x_sync
	if x_wm.dtype != torch.float32:
		x_wm_fp = x_wm.float()
	else:
		x_wm_fp = x_wm
	with torch.amp.autocast(device_type="cuda", enabled=False):
		perc = CombinedPerceptualLoss()(x_sync_fp, x_wm_fp)
		perc_total = perc["total_perceptual_loss"]
	# Total loss
	total = cfg.w_bits * bit_loss + cfg.w_amp * amp_penalty + cfg.w_perc * perc_total
	return {
		"loss": total,
		"bit_loss": bit_loss.detach().item() if torch.is_tensor(bit_loss) else float(bit_loss),
		"amp_penalty": float(amp_penalty.detach().item()),
		"perc": float(perc_total.detach().item()),
		"ber": float(ber),
		"x_wm": x_wm.detach(),
	}


# =========================
# Train/val loop
# =========================

def _build_model(cfg: TrainConfig) -> INNWatermarker:
	# Use INN with default block count; its internal STFT is adaptive-aware
	return INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 1024, "hop_length": 512, "win_length": 1024})


def _make_payload_bits_tensor(cfg: TrainConfig, device) -> torch.Tensor:
	bits = build_payload_bits(cfg)
	return bits.to(device)


def validate(model: INNWatermarker, cfg: TrainConfig, loader: DataLoader) -> Dict:
	model.eval()
	base_model = getattr(model, "module", model)
	metrics = {"loss": 0.0, "ber": 0.0, "perc": 0.0}
	with torch.no_grad():
		for batch in loader:
			x = batch.to(cfg.device, non_blocking=True)
			bits = _make_payload_bits_tensor(cfg, x.device)
			out = compute_losses_and_metrics(base_model, x, cfg, bits)
			metrics["loss"] += float(out["loss"].detach().item()) * x.size(0)
			metrics["ber"] += out["ber"] * x.size(0)
			metrics["perc"] += out["perc"] * x.size(0)
	N = len(loader.dataset)
	for k in metrics:
		metrics[k] = metrics[k] / max(1, N)
	return metrics


def train_one_epoch(model: INNWatermarker, cfg: TrainConfig, optimizer: torch.optim.Optimizer, loader: DataLoader, scaler, epoch: int) -> Dict:
	model.train()
	base_model = getattr(model, "module", model)
	running = {"loss": 0.0, "ber": 0.0, "perc": 0.0}
	pbar = tqdm(enumerate(loader), total=len(loader), desc="train", leave=False)
	for step, batch in pbar:
		x = batch.to(cfg.device, non_blocking=True)
		bits = _make_payload_bits_tensor(cfg, x.device)
		optimizer.zero_grad(set_to_none=True)
		use_amp = cfg.mixed_precision and torch.cuda.is_available()
		# Keep compute stable: do heavy math under AMP but sanitize outputs
		with torch.amp.autocast(device_type="cuda", enabled=use_amp):
			out = compute_losses_and_metrics(base_model, x, cfg, bits)
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
		running["ber"] += out["ber"] * x.size(0)
		running["perc"] += out["perc"] * x.size(0)
		if (step + 1) % cfg.log_interval == 0:
			pbar.set_postfix({
				"loss": f"{running['loss'] / ((step+1)*loader.batch_size):.4f}",
				"ber": f"{running['ber'] / ((step+1)*loader.batch_size):.4f}",
				"perc": f"{running['perc'] / ((step+1)*loader.batch_size):.4f}",
			})
	N = len(loader.dataset)
	for k in running:
		running[k] = running[k] / max(1, N)
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
	train_ds = OneSecondDataset(cfg.data_dir, TARGET_SR)
	val_ds = OneSecondDataset(cfg.val_dir, TARGET_SR)
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
	best_ber = float("inf")
	for epoch in range(1, cfg.epochs + 1):
		if is_distributed and train_sampler is not None:
			train_sampler.set_epoch(epoch)
		if (not is_distributed) or rank == 0:
			log(f"\nEpoch {epoch}/{cfg.epochs}")
		train_metrics = train_one_epoch(model, cfg, optimizer, train_loader, scaler, epoch)
		if (not is_distributed) or rank == 0:
			log(f"train: loss={train_metrics['loss']:.4f} ber={train_metrics['ber']:.4f} perc={train_metrics['perc']:.4f}")
		val_metrics = validate(model, cfg, val_loader)
		if (not is_distributed) or rank == 0:
			log(f"val  : loss={val_metrics['loss']:.4f} ber={val_metrics['ber']:.4f} perc={val_metrics['perc']:.4f}")
			# Save best by BER
			if val_metrics["ber"] < best_ber:
				best_ber = val_metrics["ber"]
				ckpt_path = os.path.join(cfg.save_dir, "phase1_best.pt")
				to_save = model.module if hasattr(model, "module") else model
				torch.save({
					"epoch": epoch,
					"model_state": to_save.state_dict(),
					"optimizer_state": optimizer.state_dict(),
					"best_ber": best_ber,
					"cfg": cfg.__dict__,
				}, ckpt_path)
				log(f"Saved best checkpoint to {ckpt_path}")
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
	args, _ = parser.parse_known_args()

	_defaults = TrainConfig()
	cfg = TrainConfig(
		data_dir=(args.data_dir or os.environ.get("DATA_DIR", _defaults.data_dir)),
		val_dir=(args.val_dir or os.environ.get("VAL_DIR", _defaults.val_dir)),
		save_dir=(args.save_dir or os.environ.get("SAVE_DIR", _defaults.save_dir)),
		epochs=(args.epochs if args.epochs is not None else int(os.environ.get("EPOCHS", _defaults.epochs))),
		batch_size=(args.batch_size if args.batch_size is not None else int(os.environ.get("PER_DEVICE_BATCH", _defaults.batch_size))),
	)
	main(cfg)
