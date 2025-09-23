#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import argparse
import warnings

import torch
import torchaudio
from torchaudio.transforms import Resample

# Ensure project root on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.payload_codec import unpack_fields


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


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
    wav = wav / (wav.abs().max() + 1e-9)
    return wav


def match_length(y: torch.Tensor, target_T: int) -> torch.Tensor:
    T = y.size(-1)
    if T == target_T:
        return y
    if T > target_T:
        return y[..., :target_T]
    return torch.nn.functional.pad(y, (0, target_T - T))


def chunk_indices(total_T: int, sr: int, eul_seconds: float = 1.0) -> list[tuple[int, int]]:
    step = int(sr * eul_seconds)
    out: list[tuple[int, int]] = []
    for s in range(0, total_T, step):
        e = min(total_T, s + step)
        out.append((s, e))
    if len(out) == 0:
        out = [(0, min(total_T, step))]
    return out


def bits_to_bytes(bits: list[int]) -> bytes:
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for k in range(8):
            if i + k < len(bits):
                b |= (int(bits[i + k]) & 1) << k
        by.append(b)
    return bytes(by)


def load_model(ckpt_path: str, n_fft: int = 1024, hop: int = 512, device: str | torch.device = None) -> INNWatermarker:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("model_state", state)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided or file not found; using randomly initialized model.")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode watermarked audio using saved meta, recover payload text.")
    parser.add_argument("watermarked", type=str, help="Path to watermarked audio file")
    parser.add_argument("meta", type=str, help="Path to metadata JSON produced by embed script")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "inn_decode_best.pt"), help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load meta
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sr = int(meta.get("sr", TARGET_SR))
    n_fft = int(meta.get("n_fft", 1024))
    hop = int(meta.get("hop", 512))
    num_bits = int(meta.get("num_bits_embedded", 0))
    slots_per_chunk = [ [(int(f), int(t)) for f, t in chunk] for chunk in meta.get("slots_per_chunk", []) ]

    # Load audio and model
    wav = load_audio_mono(args.watermarked, target_sr=sr)
    x_wm = wav.unsqueeze(0).to(device)
    model = load_model(args.ckpt, n_fft=n_fft, hop=hop, device=device)

    # Recover bits chunk-by-chunk using the same slots
    recovered_bits: list[int] = []
    bit_cursor = 0
    for (s, e), slots in zip(chunk_indices(x_wm.size(-1), sr, eul_seconds=1.0), slots_per_chunk):
        if bit_cursor >= num_bits:
            break
        x_seg = match_length(x_wm[:, :, s:e], int(sr * 1.0))
        with torch.no_grad():
            M_rec = model.decode(x_seg)
        for (f, t) in slots:
            if bit_cursor >= num_bits:
                break
            val = M_rec[:, 0, f, t]
            recovered_bits.append(int((val > 0).item()))
            bit_cursor += 1

    recovered_bytes = bits_to_bytes(recovered_bits)
    try:
        fields = unpack_fields(recovered_bytes, ["msg"])  # type: ignore[arg-type]
        text = fields.get("msg", "")
    except Exception:
        text = "<decode/parse failed>"

    print(text)


if __name__ == "__main__":
    main()


