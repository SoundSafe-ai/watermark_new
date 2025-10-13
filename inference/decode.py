#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.transforms import Resample

# Ensure repo root is importable when running as `python inference/decode.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
	sys.path.append(REPO_ROOT)

from training_new import (
	INNWatermarker,
	TARGET_SR,
	adaptive_stft,
	chunk_into_micro_windows,
	DeterministicTFMapper,
	psycho_gate_accept_mask,
)


def _bits_to_bytes_lsb_first(bits: list[int]) -> bytes:
	by = bytearray()
	for i in range(0, len(bits), 8):
		acc = 0
		for k in range(8):
			if i + k < len(bits):
				acc |= (int(bits[i + k]) & 1) << k
		by.append(acc)
	return bytes(by)


def _load_audio(path: str, target_sr: int) -> torch.Tensor:
	wav, sr = torchaudio.load(path)
	if wav.size(0) > 1:
		wav = wav.mean(dim=0, keepdim=True)
	if sr != target_sr:
		wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
	wav = wav / (wav.abs().max() + 1e-9)
	return wav  # [1,T]


def _load_model(ckpt_path: str, device: str) -> INNWatermarker:
	model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": 1024, "hop_length": 512, "win_length": 1024}).to(device)
	if ckpt_path and os.path.isfile(ckpt_path):
		ckpt = torch.load(ckpt_path, map_location=device)
		state = ckpt.get("model_state", ckpt)
		model.load_state_dict(state, strict=False)
	return model.eval()


def main() -> None:
	parser = argparse.ArgumentParser(description="Decode watermark using the same pipeline as training_new.py")
	parser.add_argument("--in", dest="inp", required=True, help="Input watermarked audio file")
	parser.add_argument("--ckpt", dest="ckpt", required=False, default="phase1_best.pt", help="Checkpoint path")
	parser.add_argument("--sr", type=int, default=TARGET_SR)
	parser.add_argument("--window_ms", type=int, default=20)
	parser.add_argument("--hop_ms", type=int, default=10)
	parser.add_argument("--mapper_seed", type=int, default=42)
	parser.add_argument("--repetition", type=int, default=3)
	parser.add_argument("--psy_mask_margin", type=float, default=1.0)
	parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
	args = parser.parse_args()

	device = args.device
	model = _load_model(args.ckpt, device)
	wav = _load_audio(args.inp, args.sr).to(device)
	T = wav.size(-1)
	sec_len = int(args.sr * 1.0)

	all_logits = []
	for start in range(0, T, sec_len):
		seg = wav[..., start:start + sec_len]
		if seg.size(-1) == 0:
			continue
		if seg.size(-1) < sec_len:
			seg = F.pad(seg, (0, sec_len - seg.size(-1)))

		windows = chunk_into_micro_windows(seg, args.window_ms, args.hop_ms, args.sr)
		accept_masks = []
		window_specs = []
		for w in windows:
			Xw = adaptive_stft(w)
			mask = psycho_gate_accept_mask(Xw, args.psy_mask_margin)
			accept_masks.append(mask)
			window_specs.append(Xw)

		F_bins = window_specs[0].size(-2)
		T_frames = window_specs[0].size(-1)
		mapper = DeterministicTFMapper(args.mapper_seed, args.repetition, 1, 2)
		placements = mapper.map_symbols(512, len(windows), F_bins, T_frames)
		gated = {}
		for sym_idx, places in placements.items():
			kept = []
			for (w_idx, f, t) in places:
				m = accept_masks[w_idx]
				if f < m.size(0) and t < m.size(1) and bool(m[f, t].item()):
					kept.append((w_idx, f, t))
			gated[sym_idx] = kept

		with torch.no_grad():
			M_rec = model.decode(seg)
			M_rec = torch.nan_to_num(M_rec, nan=0.0, posinf=1.0, neginf=-1.0)

		ordered_slots = []
		for sym_idx in sorted(gated.keys()):
			kept_sorted = sorted(gated[sym_idx], key=lambda t: (t[0], t[1], t[2]))
			ordered_slots.extend(kept_sorted)
		logits = []
		for (w_idx, f, t) in ordered_slots:
			if f < M_rec.size(-2) and t < M_rec.size(-1):
				logits.append(M_rec[0, 0, f, t].unsqueeze(0))
		if logits:
			all_logits.append(torch.cat(logits, dim=0))

	if not all_logits:
		print("No gated placements found; cannot decode.")
		return

	logits_cat = torch.cat(all_logits, dim=0)
	pred_bits = (logits_cat > 0).long().cpu().tolist()
	pred_bits = pred_bits[:512]
	decoded_bytes = _bits_to_bytes_lsb_first(pred_bits)
	trimmed = decoded_bytes.rstrip(b"\x00")
	try:
		decoded_text = trimmed.decode("utf-8", errors="strict")
	except Exception:
		decoded_text = trimmed.decode("utf-8", errors="replace")
	print(f"Decoded bytes ({len(decoded_bytes)}): {decoded_bytes.hex()}")
	print(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
	main()


