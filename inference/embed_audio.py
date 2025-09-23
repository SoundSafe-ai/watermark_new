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
from pipeline.payload_codec import pack_fields
from pipeline.psychoacoustic import mel_proxy_threshold


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


def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
    wav = wav / (wav.abs().max() + 1e-9)
    return wav


def save_audio(path: str, wav: torch.Tensor, sr: int = TARGET_SR) -> None:
    wav = wav.clamp(-1.0, 1.0)
    torchaudio.save(path, wav.cpu(), sr)


def save_combined_spectrogram_png(path: str, X_orig: torch.Tensor, X_wm: torch.Tensor) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    mag_o = torch.sqrt(torch.clamp(X_orig[:,0]**2 + X_orig[:,1]**2, min=1e-9))[0].cpu().numpy()
    mag_w = torch.sqrt(torch.clamp(X_wm[:,0]**2 + X_wm[:,1]**2, min=1e-9))[0].cpu().numpy()
    db_o = 20 * np.log10(mag_o + 1e-9)
    db_w = 20 * np.log10(mag_w + 1e-9)
    max_db = max(db_o.max(), db_w.max())
    vmin = max_db - 80.0
    vmin = min(vmin, max_db)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(db_o, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=max_db)
    axes[0].set_title('Original')
    im1 = axes[1].imshow(db_w, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=max_db)
    axes[1].set_title('Watermarked')
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('dB')
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


@torch.no_grad()
def plan_slots_gpu(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int) -> list[tuple[int,int]]:
    X = model.stft(x_wave)
    _, _, F, T = X.shape
    mag = torch.sqrt(torch.clamp(X[:,0]**2 + X[:,1]**2, min=1e-6))
    thr = mel_proxy_threshold(X, n_mels=64)
    score = mag / (thr + 1e-6)
    k = int(min(target_bits, F*T))
    vals, idx = score[0].flatten().topk(k)
    f = (idx // T).to(torch.int64)
    t = (idx % T).to(torch.int64)
    return [(int(f[i]), int(t[i])) for i in range(k)]


def bytes_to_bits(by: bytes) -> list[int]:
    bits: list[int] = []
    for b in by:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


def build_message_spec_for_slots(model: INNWatermarker, x_wave: torch.Tensor, slots: list[tuple[int,int]], bits: list[int], base_symbol_amp: float) -> torch.Tensor:
    X = model.stft(x_wave)
    F_, T_ = X.shape[-2], X.shape[-1]
    S = min(len(slots), len(bits))
    M = torch.zeros(1, 2, F_, T_, device=x_wave.device)
    for s in range(S):
        f, t = slots[s]
        sign = (int(bits[s]) * 2 - 1)
        M[:, 0, f, t] = float(sign) * float(base_symbol_amp)
    return M


def encode_full_audio(model: INNWatermarker, x_full: torch.Tensor, sr: int, payload_bits: list[int], *, target_bits_per_eul: int, base_symbol_amp: float) -> tuple[torch.Tensor, list[list[tuple[int,int]]], int]:
    B, C, T = x_full.shape
    assert B == 1 and C == 1
    out_chunks = []
    slots_per_chunk: list[list[tuple[int,int]]] = []
    bit_cursor = 0
    for (s, e) in chunk_indices(T, sr, eul_seconds=1.0):
        x_seg = x_full[:, :, s:e]
        x_seg_eul = match_length(x_seg, int(sr * 1.0))
        slots = plan_slots_gpu(model, x_seg_eul, target_bits=target_bits_per_eul)
        S = min(len(slots), len(payload_bits) - bit_cursor)
        if S <= 0:
            x_wm_seg = x_seg_eul
        else:
            M_spec = build_message_spec_for_slots(model, x_seg_eul, slots[:S], payload_bits[bit_cursor:bit_cursor+S], base_symbol_amp)
            x_wm_seg, _ = model.encode(x_seg_eul, M_spec)
            bit_cursor += S
        x_wm_seg = match_length(x_wm_seg, x_seg_eul.size(-1))
        x_wm_seg = x_wm_seg[:, :, : (e - s)]
        out_chunks.append(x_wm_seg)
        slots_per_chunk.append(slots[:S])
    x_wm_full = torch.cat(out_chunks, dim=-1)
    x_wm_full = x_wm_full[:, :, :T]
    return x_wm_full, slots_per_chunk, bit_cursor


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
    parser = argparse.ArgumentParser(description="Embed payload into audio, output spectrogram and watermarked audio (+meta).")
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument("payload", type=str, help="Payload text to embed")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "inn_decode_best.pt"), help="Path to checkpoint")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="Target sample rate")
    parser.add_argument("--out_dir", type=str, default="inference_outputs", help="Directory to save outputs")
    parser.add_argument("--bits_per_eul", type=int, default=512, help="Target number of bits per 1s chunk (as in training)")
    parser.add_argument("--base_symbol_amp", type=float, default=0.1, help="Symbol amplitude used for BPSK embed (training default)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    wav = load_audio_mono(args.audio, target_sr=args.sr)
    x = wav.unsqueeze(0).to(device)  # [1,1,T_full]

    model = load_model(args.ckpt, n_fft=1024, hop=512, device=device)

    # Original spectrogram (full)
    with torch.no_grad():
        X_orig = model.stft(x)

    # Build payload bits
    payload_bytes = pack_fields({"msg": args.payload})
    payload_bits = bytes_to_bits(payload_bytes)

    # Encode full audio
    x_wm, slots_per_chunk, num_bits_embedded = encode_full_audio(
        model, x, args.sr, payload_bits,
        target_bits_per_eul=args.bits_per_eul,
        base_symbol_amp=args.base_symbol_amp,
    )

    # Save outputs
    wm_path = os.path.join(args.out_dir, "watermarked.wav")
    save_audio(wm_path, x_wm[0].cpu(), sr=args.sr)
    with torch.no_grad():
        X_wm = model.stft(x_wm)
    save_combined_spectrogram_png(os.path.join(args.out_dir, "spectrograms.png"), X_orig, X_wm)

    # Save metadata for decoding
    meta = {
        "sr": args.sr,
        "n_fft": 1024,
        "hop": 512,
        "bits_per_eul": args.bits_per_eul,
        "base_symbol_amp": args.base_symbol_amp,
        "num_bits_embedded": int(num_bits_embedded),
        "slots_per_chunk": [[[int(f), int(t)] for (f,t) in slots] for slots in slots_per_chunk],
        "planner": "gpu_mel_proxy",
        "payload_text": args.payload,
    }
    meta_path = os.path.join(args.out_dir, "watermarked.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {wm_path}\nMeta: {meta_path}\nSpectrograms: {os.path.join(args.out_dir, 'spectrograms.png')}")


if __name__ == "__main__":
    main()


