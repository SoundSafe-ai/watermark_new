#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

# Ensure repo root is importable when running as `python inference/encode.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)

# Reuse the exact pipeline pieces from training to ensure consistency
from training_new import (
    INNWatermarker,
    TrainConfig,
    TARGET_SR,
    adaptive_stft,
    chunk_into_micro_windows,
    DeterministicTFMapper,
    psycho_gate_accept_mask,
    per_window_amp_budget,
    embed_sync_marker,
    build_message_spec_for_second,
)
# RS/Interleave helpers (same as training)
from pipeline.ingest_and_chunk import (
    rs_encode_167_125,
    interleave_bytes,
)


def _load_audio(path: str, target_sr: int) -> torch.Tensor:
	wav, sr = torchaudio.load(path)
	if wav.size(0) > 1:
		wav = wav.mean(dim=0, keepdim=True)
	if sr != target_sr:
		wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
	wav = wav / (wav.abs().max() + 1e-9)
	return wav  # [1,T]


def _save_audio(path: str, wav: torch.Tensor, sr: int) -> None:
	# torchaudio.save expects [channels, samples]
	wav = wav.detach().cpu()
	if wav.dim() == 3:  # [B,C,T] -> squeeze batch (Phase-1 uses B=1)
		if wav.size(0) != 1:
			# If somehow multiple in batch, collapse to first (or mixdown). Keep first for determinism.
			wav = wav[0]
		else:
			wav = wav.squeeze(0)
	elif wav.dim() == 1:  # [T] -> add channel
		wav = wav.unsqueeze(0)
	elif wav.dim() != 2:
		raise ValueError(f"Expected audio tensor of shape [C,T] or [B,C,T], got dim={wav.dim()}")
	# Sanitize and clamp to valid range
	wav = torch.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
	torchaudio.save(path, wav, sample_rate=sr, encoding="PCM_F32")


def _load_model(ckpt_path: str, device: str) -> INNWatermarker:
	# Use adaptive STFT like training instead of fixed parameters
	model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 1024, "hop_length": 512, "win_length": 1024}).to(device)
	if ckpt_path and os.path.isfile(ckpt_path):
		ckpt = torch.load(ckpt_path, map_location=device)
		state = ckpt.get("model_state", ckpt)
		model.load_state_dict(state, strict=False)
	return model.eval()


def _bytes_to_bits_lsb_first(by: bytes) -> list[int]:
	bits: list[int] = []
	for b in by:
		for k in range(8):
			bits.append((b >> k) & 1)
	return bits


def main() -> None:
	parser = argparse.ArgumentParser(description="Encode watermark using the same pipeline as training_new.py")
	parser.add_argument("--in", dest="inp", required=True, help="Input clean audio file")
	parser.add_argument("--out", dest="out", required=True, help="Output watermarked audio file")
	parser.add_argument("--ckpt", dest="ckpt", required=False, default="phase1_best.pt", help="Checkpoint path")
	parser.add_argument("--sr", type=int, default=TARGET_SR)
	parser.add_argument("--payload", type=str, required=True, help="UTF-8 string (<=64 bytes), will be zero-padded to 64 bytes")
	parser.add_argument("--window_ms", type=int, default=20)
	parser.add_argument("--hop_ms", type=int, default=10)
	parser.add_argument("--mapper_seed", type=int, default=42)
	parser.add_argument("--repetition", type=int, default=3)
	parser.add_argument("--min_time_spacing", type=int, default=1)
	parser.add_argument("--min_freq_spacing", type=int, default=2)
	parser.add_argument("--psy_mask_margin", type=float, default=1.0)
	parser.add_argument("--amp_budget_scale", type=float, default=0.25)
	parser.add_argument("--sync_strength", type=float, default=0.05)
	parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
	args = parser.parse_args()

	device = args.device
	model = _load_model(args.ckpt, device)

	# Load audio and split into 1s chunks, process each independently (Phase-1 contract)
	wav = _load_audio(args.inp, args.sr).to(device)
	T = wav.size(-1)
	sec_len = int(args.sr * 1.0)
	segments = []
	for start in range(0, T, sec_len):
		seg = wav[..., start:start + sec_len]
		if seg.size(-1) == 0:
			continue
		if seg.size(-1) < sec_len:
			seg = F.pad(seg, (0, sec_len - seg.size(-1)))
		segments.append(seg)

	# Build user payload bytes (64 bytes, zero-padded), then RS-encode + interleave
	payload_bytes = args.payload.encode("utf-8")
	if len(payload_bytes) > 64:
		raise ValueError("--payload exceeds 64 bytes when UTF-8 encoded")
	if len(payload_bytes) < 64:
		payload_bytes = payload_bytes + bytes(64 - len(payload_bytes))
	# RS encode + interleave to match training
	payload_bytes_rs = rs_encode_167_125(payload_bytes)
	payload_bytes_rs = interleave_bytes(payload_bytes_rs, 1)
	payload_bits_list = _bytes_to_bits_lsb_first(payload_bytes_rs)

	outs = []
	for seg in segments:
		# 1) Add sync
		seg_sync = embed_sync_marker(seg, args.sync_strength, args.sr, args.mapper_seed)
		# 2) Micro-windows and psycho gate
		windows = chunk_into_micro_windows(seg_sync, args.window_ms, args.hop_ms, args.sr)
		accept_masks = []
		amp_budget_by_window = {}
		window_specs = []
		for i, w in enumerate(windows):
			Xw = adaptive_stft(w)
			mask = psycho_gate_accept_mask(Xw, args.psy_mask_margin)
			accept_masks.append(mask)
			amp_budget_by_window[i] = per_window_amp_budget(Xw, mask, args.amp_budget_scale)
			window_specs.append(Xw)
		# 3) Deterministic placements with repetition (exactly like training)
		F_bins = window_specs[0].size(-2)
		T_frames = window_specs[0].size(-1)
		mapper = DeterministicTFMapper(
			args.mapper_seed, args.repetition, args.min_time_spacing, args.min_freq_spacing
		)
		n_symbols = 512
		placements = mapper.map_symbols(n_symbols, len(windows), F_bins, T_frames)
		# Gate placements
		gated = {}
		for sym_idx, places in placements.items():
			kept = []
			for (w_idx, f, t) in places:
				m = accept_masks[w_idx]
				if f < m.size(0) and t < m.size(1) and bool(m[f, t].item()):
					kept.append((w_idx, f, t))
			gated[sym_idx] = kept
		# 4) Build bits-per-symbol vector exactly as training expects
		# If encoded bitstream is shorter than n_symbols, repeat; otherwise truncate
		if len(payload_bits_list) < n_symbols:
			reps = (n_symbols + len(payload_bits_list) - 1) // len(payload_bits_list)
			bits_stream = (payload_bits_list * reps)[:n_symbols]
		else:
			bits_stream = payload_bits_list[:n_symbols]
		bits_by_symbol = bits_stream
		# 5) Build message spec via the same helper used in training
		M_spec = build_message_spec_for_second(seg_sync, gated, bits_by_symbol, amp_budget_by_window)
		# 6) Encode
		with torch.no_grad():
			seg_wm, _ = model.encode(seg_sync, M_spec)
			seg_wm = torch.nan_to_num(seg_wm, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
		outs.append(seg_wm)

	if len(outs) == 0:
		raise RuntimeError("No segments produced during encoding")
	wm = torch.cat(outs, dim=-1)[..., :T]  # trim back to original length
	_save_audio(args.out, wm, args.sr)
	print(f"Saved watermarked audio to {os.path.abspath(args.out)}")


if __name__ == "__main__":
	main()


