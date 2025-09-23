#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import warnings
from typing import Tuple

import torch
import torchaudio
from torchaudio.transforms import Resample

# Ensure project root is on sys.path so we can import `models`, `pipeline`, etc. when running directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.inn_encoder_decoder import INNWatermarker, STFT
from pipeline.payload_codec import pack_fields, unpack_fields
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
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
    wav = wav / (wav.abs().max() + 1e-9)
    return wav  # [1, T]


def save_audio(path: str, wav: torch.Tensor, sr: int = TARGET_SR) -> None:
    wav = wav.clamp(-1.0, 1.0)
    torchaudio.save(path, wav.cpu(), sr)


def save_spectrogram_png(path: str, X_ri: torch.Tensor) -> None:
    """
    Save magnitude spectrogram as PNG using matplotlib (lazy import).
    X_ri: [1,2,F,T]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-9))[0].cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(20 * np.log10(mag + 1e-9), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar(label='dB')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    return torch.nn.functional.pad(y, (0, pad))


def save_combined_spectrogram_png(path: str, X_orig: torch.Tensor, X_wm: torch.Tensor) -> None:
    """
    Save a single image with original and watermarked magnitude spectrograms side-by-side.
    X_orig/X_wm: [1,2,F,T]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    mag_o = torch.sqrt(torch.clamp(X_orig[:,0]**2 + X_orig[:,1]**2, min=1e-9))[0].cpu().numpy()
    mag_w = torch.sqrt(torch.clamp(X_wm[:,0]**2 + X_wm[:,1]**2, min=1e-9))[0].cpu().numpy()
    db_o = 20 * np.log10(mag_o + 1e-9)
    db_w = 20 * np.log10(mag_w + 1e-9)
    max_db = max(db_o.max(), db_w.max())
    vmin = max_db - 80.0
    vmin = min(vmin, max_db)  # guard

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    im0 = axes[0].imshow(db_o, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=max_db)
    axes[0].set_title('Original')
    im1 = axes[1].imshow(db_w, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=max_db)
    axes[1].set_title('Watermarked')
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('dB')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chunk_indices(total_T: int, sr: int, eul_seconds: float = 1.0) -> list[tuple[int, int]]:
    step = int(sr * eul_seconds)
    idx: list[tuple[int, int]] = []
    for start in range(0, total_T, step):
        end = min(total_T, start + step)
        idx.append((start, end))
    if len(idx) == 0:
        idx = [(0, min(total_T, step))]
    return idx


def bytes_to_bits(by: bytes) -> list[int]:
    bits: list[int] = []
    for b in by:
        for k in range(8):
            bits.append((b >> k) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for k in range(8):
            if i + k < len(bits):
                b |= (int(bits[i + k]) & 1) << k
        by.append(b)
    return bytes(by)


@torch.no_grad()
def plan_slots_gpu(model: INNWatermarker, x_wave: torch.Tensor, target_bits: int) -> list[tuple[int, int]]:
    """
    Mirror training's mel-proxy GPU planner: choose top-k (f,t) by mag/threshold.
    Returns a list of (f, t) slots, length S<=target_bits.
    """
    X = model.stft(x_wave)  # [1,2,F,T]
    _, _, F, T = X.shape
    mag = torch.sqrt(torch.clamp(X[:,0]**2 + X[:,1]**2, min=1e-6))  # [1,F,T]
    thr = mel_proxy_threshold(X, n_mels=64)                          # [1,F,T]
    score = mag / (thr + 1e-6)
    k = int(min(target_bits, F * T))
    vals, idx = score[0].flatten().topk(k)
    f = (idx // T).to(torch.int64)
    t = (idx % T).to(torch.int64)
    return [(int(f[i]), int(t[i])) for i in range(k)]


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
    """
    Encode watermark over entire audio by splitting into 1s EULs and concatenating.
    Returns (x_wm_full [1,1,T], slots_per_chunk, num_bits_embedded).
    """
    B, C, T = x_full.shape
    assert B == 1 and C == 1
    out_chunks = []
    slots_per_chunk: list[list[tuple[int,int]]] = []
    bit_cursor = 0
    for (s, e) in chunk_indices(T, sr, eul_seconds=1.0):
        x_seg = x_full[:, :, s:e]
        x_seg_eul = match_length(x_seg, int(sr * 1.0))
        # Plan slots on clean segment
        slots = plan_slots_gpu(model, x_seg_eul, target_bits=target_bits_per_eul)
        S = min(len(slots), len(payload_bits) - bit_cursor)
        if S <= 0:
            # No bits left; pass-through
            x_wm_seg = x_seg_eul
        else:
            M_spec = build_message_spec_for_slots(model, x_seg_eul, slots[:S], payload_bits[bit_cursor:bit_cursor+S], base_symbol_amp)
            with torch.no_grad():
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


def make_payload_bytes(payload_text: str) -> bytes:
    # Simple example: pack a single field under key 'msg'
    return pack_fields({"msg": payload_text})[:125]  # ensure 125 bytes max for RS(167,125)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed payload into audio, save specs/audio, then decode.")
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument("payload", type=str, help="Payload text to embed")
    parser.add_argument("--ckpt_imperc", type=str, default=os.path.join("checkpoints", "inn_imperc_best.pt"), help="Path to imperceptibility checkpoint")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help="Target sample rate")
    parser.add_argument("--out_dir", type=str, default="inference_outputs", help="Directory to save outputs")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    parser.add_argument("--bits_per_eul", type=int, default=512, help="Target number of bits per 1s chunk (as in training)")
    parser.add_argument("--base_symbol_amp", type=float, default=0.1, help="Symbol amplitude used for BPSK embed (training default)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load audio (full length)
    wav = load_audio_mono(args.audio, target_sr=args.sr)
    x = wav.unsqueeze(0).to(device)  # [1,1,T_full]

    # Load model
    model = load_model(args.ckpt_imperc, n_fft=1024, hop=512, device=device)

    # Original spectrogram (full audio)
    X_orig = model.stft(x)  # [1,2,F,T]

    # Build payload bytes and convert to bitstream
    payload_bytes = make_payload_bytes(args.payload)
    payload_bits = bytes_to_bits(payload_bytes)

    # Encode (embed) over entire file via 1s EUL chunks (mirror training, no RS)
    x_wm, slots_per_chunk, num_bits_embedded = encode_full_audio(
        model, x, args.sr, payload_bits,
        target_bits_per_eul=args.bits_per_eul,
        base_symbol_amp=args.base_symbol_amp,
    )

    # Save watermarked audio
    save_audio(os.path.join(args.out_dir, "watermarked.wav"), x_wm[0].cpu(), sr=args.sr)

    # Watermarked spectrogram (full) and combined figure
    with torch.no_grad():
        X_wm = model.stft(x_wm)
    save_combined_spectrogram_png(os.path.join(args.out_dir, "spectrograms.png"), X_orig, X_wm)

    # Decode by mirroring training: recover message spectrogram and read bits at the same planned slots
    with torch.no_grad():
        M_rec_full = model.decode(x_wm)
    recovered_bits: list[int] = []
    bit_goal = num_bits_embedded
    cursor = 0
    for chunk_i, (s, e) in enumerate(chunk_indices(x.size(-1), args.sr, eul_seconds=1.0)):
        if cursor >= bit_goal:
            break
        slots = slots_per_chunk[chunk_i]
        for (f, t) in slots:
            if cursor >= bit_goal:
                break
            # Map chunk-local t to global frame index: since we padded to full EUL in encode, STFT frames align per-second
            # Simpler approach: recompute decode per chunk to avoid frame indexing mismatch
            pass
    # To avoid global frame mapping complexity, decode per chunk like encode
    recovered_bits = []
    bit_cursor = 0
    for (s, e), slots in zip(chunk_indices(x.size(-1), args.sr, eul_seconds=1.0), slots_per_chunk):
        if bit_cursor >= bit_goal:
            break
        x_wm_seg = match_length(x_wm[:, :, s:e], int(args.sr * 1.0))
        with torch.no_grad():
            M_rec = model.decode(x_wm_seg)
        for (f, t) in slots:
            if bit_cursor >= bit_goal:
                break
            val = M_rec[:, 0, f, t]  # [1]
            recovered_bits.append(int((val > 0).item()))
            bit_cursor += 1
    recovered_bytes = bits_to_bytes(recovered_bits)

    # Try to unpack assuming same field order
    try:
        rec_fields = unpack_fields(recovered_bytes, ["msg"])  # type: ignore[arg-type]
        rec_msg = rec_fields.get("msg", "")
    except Exception:
        rec_msg = "<decode/parse failed>"

    # Save small report
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Input: {args.audio}\n")
        f.write(f"Payload (text): {args.payload}\n")
        f.write(f"Recovered (text): {rec_msg}\n")
        f.write(f"Embedded bits: {num_bits_embedded}\n")
        f.write(f"Recovered bytes (hex): {recovered_bytes.hex()}\n")
        f.write("Files: spectrograms.png, watermarked.wav\n")
        f.write("Decode note: mirrored training (no RS), planned per 1s EUL on clean audio.\n")

    print(f"Saved outputs to {args.out_dir}")
    print(f"Recovered text: {rec_msg}")


if __name__ == "__main__":
    main()


