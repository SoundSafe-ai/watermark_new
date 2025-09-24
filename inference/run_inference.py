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
from pipeline.ingest_and_chunk import (
    EULDriver,
    allocate_slots_and_amplitudes,
    message_spec_to_bits,
    rs_encode_167_125,
    rs_decode_167_125,
    interleave_bytes,
    deinterleave_bytes,
)


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
    torchaudio.save(path, wav.detach().cpu(), sr)


def save_spectrogram_png(path: str, X_ri: torch.Tensor) -> None:
    """
    Save magnitude spectrogram as PNG using matplotlib (lazy import).
    X_ri: [1,2,F,T]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    mag = torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-9))[0].detach().cpu().numpy()
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
    mag_o = torch.sqrt(torch.clamp(X_orig[:,0]**2 + X_orig[:,1]**2, min=1e-9))[0].detach().cpu().numpy()
    mag_w = torch.sqrt(torch.clamp(X_wm[:,0]**2 + X_wm[:,1]**2, min=1e-9))[0].detach().cpu().numpy()
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


def encode_full_audio_eul_driver(model: INNWatermarker, x_full: torch.Tensor, sr: int, payload_bytes: bytes, eul_driver: EULDriver) -> tuple[torch.Tensor, list[list[tuple[int,int]]], int]:
    """
    Encode watermark over entire audio using EULDriver (includes RS encoding and full psychoacoustic allocation).
    Returns (x_wm_full [1,1,T], slots_per_chunk, num_bytes_embedded).
    """
    B, C, T = x_full.shape
    assert B == 1 and C == 1
    out_chunks = []
    slots_per_chunk: list[list[tuple[int,int]]] = []
    bytes_cursor = 0
    payload_size = len(payload_bytes)

    for (s, e) in chunk_indices(T, sr, eul_seconds=1.0):
        x_seg = x_full[:, :, s:e]
        x_seg_eul = match_length(x_seg, int(sr * 1.0))

        # Use EULDriver for this chunk
        if bytes_cursor < payload_size:
            # Take next 125 bytes for this EUL
            chunk_bytes = payload_bytes[bytes_cursor:bytes_cursor+125]
            if len(chunk_bytes) < 125:
                # Pad with zeros if needed
                chunk_bytes = chunk_bytes + b'\x00' * (125 - len(chunk_bytes))

            x_wm_seg = eul_driver.encode_eul(model, x_seg_eul, chunk_bytes)
            bytes_cursor += 125

            # Get slots used by EULDriver (need to recompute for consistency)
            X = model.stft(x_seg_eul)
            from pipeline.ingest_and_chunk import allocate_slots_and_amplitudes
            slots, _ = allocate_slots_and_amplitudes(
                X, sr, 1024, target_bits=eul_driver.per_eul_bits_target, amp_safety=1.0
            )
            slots_per_chunk.append(slots[:eul_driver.per_eul_bits_target])
        else:
            # No payload left, pass through
            x_wm_seg = x_seg_eul
            slots_per_chunk.append([])

        x_wm_seg = match_length(x_wm_seg, x_seg_eul.size(-1))
        x_wm_seg = x_wm_seg[:, :, : (e - s)]
        out_chunks.append(x_wm_seg)

    x_wm_full = torch.cat(out_chunks, dim=-1)
    x_wm_full = x_wm_full[:, :, :T]
    return x_wm_full, slots_per_chunk, min(bytes_cursor, payload_size)


def decode_full_audio_eul_driver(model: INNWatermarker, x_wm_full: torch.Tensor, sr: int, eul_driver: EULDriver, num_bytes_embedded: int) -> bytes:
    """
    Decode watermark from entire audio using EULDriver with Acoustic DNA pilots and slot mapping.
    Returns recovered payload bytes.
    """
    B, C, T = x_wm_full.shape
    assert B == 1 and C == 1

    recovered_chunks = []
    # RS expands 125B -> 167B, but we embed per-EUL fixed 125B pre-RS payload
    chunks_needed = (num_bytes_embedded + 124) // 125

    chunk_idx = 0
    for (s, e) in chunk_indices(T, sr, eul_seconds=1.0):
        if chunk_idx >= chunks_needed:
            break

        x_wm_seg = match_length(x_wm_full[:, :, s:e], int(sr * 1.0))
        try:
            chunk_bytes = eul_driver.decode_eul(model, x_wm_seg, expected_bytes=125)
            recovered_chunks.append(chunk_bytes)
            chunk_idx += 1
            print(f"Decoded chunk {chunk_idx}/{chunks_needed} at {s}-{e}s: {len(chunk_bytes)} bytes")
        except Exception as e:
            print(f"Warning: EUL decode failed at {s}-{e}s: {e}")
            recovered_chunks.append(b'\x00' * 125)
            chunk_idx += 1

    recovered_payload = b''.join(recovered_chunks)
    return recovered_payload[:num_bytes_embedded]


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
    parser.add_argument("--base_symbol_amp", type=float, default=0.2, help="Symbol amplitude used for BPSK embed (increased for better recovery)")
    parser.add_argument("--pilot_fraction", type=float, default=0.08, help="Fraction of per-EUL bit budget reserved for pilot PN (0.0-0.25). Typical 0.05-0.12; higher fraction reduces payload slightly but improves robustness.")
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

    # Build payload bytes using proper encoding pipeline
    payload_bytes = make_payload_bytes(args.payload)
    print(f"Original payload: {len(payload_bytes)} bytes")

    # Use EULDriver for proper RS encoding/decoding pipeline
    eul_driver = EULDriver(
        sr=args.sr,
        n_fft=1024,
        hop=512,
        rs_interleave=4,
        per_eul_bits_target=167 * 8,  # 1336 bits for RS(167,125)
        base_symbol_amp=args.base_symbol_amp,
        amp_safety=1.0,
        pilot_fraction=args.pilot_fraction,
    )

    # Encode using EULDriver (includes RS encoding and full psychoacoustic allocation)
    print("Encoding with RS(167,125) + interleaving + full psychoacoustic allocation...")
    x_wm, slots_per_chunk, num_bits_embedded = encode_full_audio_eul_driver(
        model, x, args.sr, payload_bytes, eul_driver
    )

    # Save watermarked audio
    save_audio(os.path.join(args.out_dir, "watermarked.wav"), x_wm[0].cpu(), sr=args.sr)

    # Watermarked spectrogram (full) and combined figure
    with torch.no_grad():
        X_wm = model.stft(x_wm)
    save_combined_spectrogram_png(os.path.join(args.out_dir, "spectrograms.png"), X_orig, X_wm)

    # Decode using EULDriver (includes RS decoding and full psychoacoustic allocation)
    print("Decoding with RS(167,125) + deinterleaving + full psychoacoustic allocation...")
    recovered_payload = decode_full_audio_eul_driver(model, x_wm, args.sr, eul_driver, num_bits_embedded)

    # Try to unpack assuming same field order
    try:
        rec_fields = unpack_fields(recovered_payload, ["msg"])  # type: ignore[arg-type]
        rec_msg = rec_fields.get("msg", "")
    except Exception as e:
        rec_msg = f"<decode/parse failed: {e}>"
        recovered_payload = b""

    # Save small report
    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Input: {args.audio}\n")
        f.write(f"Payload (text): {args.payload}\n")
        f.write(f"Recovered (text): {rec_msg}\n")
        f.write(f"Embedded bytes: {num_bits_embedded}\n")
        f.write(f"Recovered bytes (hex): {recovered_payload.hex() if recovered_payload else 'N/A'}\n")
        f.write("Files: spectrograms.png, watermarked.wav\n")
        f.write("Pipeline: RS(167,125) + interleaving + full psychoacoustic allocation + INN encoding/decoding\n")

    print(f"Saved outputs to {args.out_dir}")
    print(f"Recovered text: {rec_msg}")


if __name__ == "__main__":
    main()


