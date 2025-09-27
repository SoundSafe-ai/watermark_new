#!/usr/bin/env python3
"""
Decode watermark payload from a watermarked audio file using the same
planner/STFT/RS parameters saved in the training checkpoint.

Steps:
- Load checkpoint cfg: n_fft, hop, target_bits, planner, RS params
- Resample audio to 22050 Hz, split into 1s chunks
- For each chunk: plan slots with the checkpoint's planner, run model.decode,
  gather values at planned slots, threshold to bits
- Aggregate bits across chunks and attempt RS(167,125) decode for one codeword

Outputs:
- Prints number of extracted bits and recovered payload (utf-8 best effort)
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import torch
import torchaudio

# Make project root importable when running this script from repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.psychoacoustic import mel_proxy_threshold
from pipeline.ingest_and_chunk import (
    rs_decode_167_125,
    deinterleave_bytes,
    allocate_slots_and_amplitudes,
)


TARGET_SR = 22050
CHUNK_SECONDS = 1.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)


def _resample_if_needed(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    if sr == TARGET_SR:
        return wav, sr
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(wav)
    return wav, TARGET_SR


def _load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)  # [C,T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav / (wav.abs().max() + 1e-9)
    wav, sr = _resample_if_needed(wav, sr)
    return wav, sr


def _chunk_audio_1s(wav: torch.Tensor) -> list[torch.Tensor]:
    T = wav.size(-1)
    chunks: list[torch.Tensor] = []
    cursor = 0
    while cursor < T:
        end = min(cursor + CHUNK_SAMPLES, T)
        ch = wav[..., cursor:end]
        if ch.size(-1) < CHUNK_SAMPLES:
            ch = torch.nn.functional.pad(ch, (0, CHUNK_SAMPLES - ch.size(-1)))
        chunks.append(ch)
        cursor += CHUNK_SAMPLES
    if len(chunks) == 0:
        chunks = [torch.nn.functional.pad(wav[..., :0], (0, CHUNK_SAMPLES))]
    return chunks


@torch.no_grad()
def _fast_gpu_plan_slots(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int) -> tuple[list[tuple[int,int]], torch.Tensor]:
    X = model.stft(x_wave)  # [1,2,F,T]
    B, _, F, T = X.shape
    assert B == 1
    mag = torch.sqrt(torch.clamp(X[:,0]**2 + X[:,1]**2, min=1e-6))
    thr = mel_proxy_threshold(X, n_mels=64)
    score = mag / (thr + 1e-6)
    k = int(min(target_bits, F*T))
    _vals, idx = score[0].flatten().topk(k)
    f = (idx // T).to(torch.int64)
    t = (idx % T).to(torch.int64)
    slots: list[tuple[int,int]] = [(int(f[i]), int(t[i])) for i in range(k)]
    amp = thr[0].flatten()[idx]
    med = torch.median(amp)
    amp = amp / (med + 1e-9)
    return slots, amp.detach()


@torch.no_grad()
def _mg_plan_slots(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int, n_fft: int) -> tuple[list[tuple[int,int]], torch.Tensor]:
    X = model.stft(x_wave)
    slots, amp = allocate_slots_and_amplitudes(X, TARGET_SR, n_fft, target_bits, amp_safety=1.0)
    if isinstance(amp, torch.Tensor):
        amp_t = amp
    else:
        import numpy as np
        if isinstance(amp, np.ndarray):
            amp_t = torch.from_numpy(amp)
        else:
            amp_t = torch.ones(len(slots), dtype=torch.float32)
    return slots, amp_t.to(x_wave.device)


def _bits_to_bytes(bits: torch.Tensor) -> bytes:
    if bits.dim() == 2:
        bits = bits[0]
    by = bytearray()
    for i in range(0, len(bits), 8):
        if i + 8 > len(bits):
            break
        b = 0
        for k in range(8):
            b |= (int(bits[i + k]) & 1) << k
        by.append(b)
    return bytes(by)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to decode checkpoint (.pt)")
    parser.add_argument("--audio", type=str, required=True, help="Path to watermarked audio file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--planner", type=str, default=None, choices=[None, "gpu", "mg"], help="Override planner (default: use checkpoint cfg)")
    parser.add_argument("--out_text", type=str, default="recovered.txt", help="Where to save recovered utf-8 text (best effort)")
    parser.add_argument("--slots", type=str, default=None, help="Path to JSON with slots_per_chunk from encoding (recommended)")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location=args.device)
    cfg = state.get("cfg", {})
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop", 512))
    target_bits = int(cfg.get("target_bits", 1336))
    rs_payload_bytes = int(cfg.get("rs_payload_bytes", 125))
    rs_interleave_depth = int(cfg.get("rs_interleave_depth", 4))
    planner = args.planner or cfg.get("planner", "mg")

    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(args.device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()

    wav, sr = _load_audio_mono(args.audio)
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected SR after resample: {sr}")
    chunks = _chunk_audio_1s(wav)

    # Optional: load exact slots used at embed time
    slots_per_chunk: list[list[list[int]]] | None = None
    if args.slots and os.path.isfile(args.slots):
        try:
            with open(args.slots, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Expect {"slots_per_chunk": [[[f,t],...], ...]}
            if isinstance(data, dict) and "slots_per_chunk" in data:
                slots_per_chunk = data["slots_per_chunk"]
        except Exception:
            slots_per_chunk = None

    rec_bits_list = []
    for idx, ch in enumerate(chunks):
        x = ch.to(args.device).unsqueeze(0)  # [1,1,T]
        # Prefer exact slots from encoder if provided; fallback to planner otherwise
        if slots_per_chunk is not None and idx < len(slots_per_chunk) and isinstance(slots_per_chunk[idx], list):
            slots = [(int(ft[0]), int(ft[1])) for ft in slots_per_chunk[idx] if isinstance(ft, (list, tuple)) and len(ft) == 2]
        else:
            if planner == "gpu":
                slots, _ = _fast_gpu_plan_slots(model, x, target_bits)
            else:
                slots, _ = _mg_plan_slots(model, x, target_bits, n_fft)
        S = min(len(slots), target_bits)
        if S == 0:
            continue
        # Decode
        M_rec = model.decode(x)
        rec_vals = torch.stack([M_rec[0, 0, f, t] for (f, t) in slots[:S]], dim=0).unsqueeze(0)
        rec_bits = (rec_vals > 0).long()
        rec_bits_list.append(rec_bits)

    if len(rec_bits_list) == 0:
        print("No bits recovered (no slots or empty audio)")
        return

    all_bits = torch.cat(rec_bits_list, dim=1)  # [1, N]
    # Limit to a single RS codeword by default; this matches encode/decode unit
    rs_code_bits = 167 * 8
    used_len = min(all_bits.size(1), rs_code_bits)
    print(f"Extracted bits: {used_len}")

    # RS decode first codeword if available
    recovered_text = ""
    if all_bits.size(1) >= rs_code_bits:
        byte_stream = _bits_to_bytes(all_bits[0, :rs_code_bits])
        deint = deinterleave_bytes(byte_stream, rs_interleave_depth)
        try:
            payload = rs_decode_167_125(deint)[:rs_payload_bytes]
            try:
                recovered_text = payload.decode("utf-8", errors="ignore")
            except Exception:
                recovered_text = ""
        except Exception:
            recovered_text = ""

    if recovered_text:
        print(f"Recovered payload (utf-8, truncated): {recovered_text[:128]}")
        try:
            with open(args.out_text, "w", encoding="utf-8") as f:
                f.write(recovered_text)
            print(f"Saved recovered text to: {args.out_text}")
        except Exception:
            pass
    else:
        print("Recovered payload unavailable (insufficient bits or RS decode failed)")


if __name__ == "__main__":
    main()


