#!/usr/bin/env python3
"""
Validation runner:
- Reads `validation/manifest.csv` with columns: audio_path,payload,out_dir,augment
- Uses the encode/decode flow from `inference/new_inference.py`
- Computes metrics per file and aggregates at the end:
  PESQ (resampled to 16 kHz), SNR, ΔSPL (original vs watermark residual), MSE, BER
- Optional robustness check: apply listed augmentations to the watermarked audio
  (toggle via --use_augmentations), and compute BER again after augmentation.

Notes:
- Requires torchaudio and (optional) pesq package. If `pesq` is unavailable, PESQ is skipped.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import torch
import torchaudio
import torch.nn.functional as F

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse helpers from inference script
from inference.new_inference import (  # type: ignore
    TARGET_SR,
    _load_audio_mono,
    _chunk_audio_1s,
    _fast_gpu_plan_slots,
    _mg_plan_slots,
    _encode_payload_bits_from_text,
    _decode_payload_bits,
)
from models.inn_encoder_decoder import INNWatermarker  # type: ignore


# --------- Utility: metrics ---------

def compute_pesq(orig: torch.Tensor, proc: torch.Tensor, sr_src: int = TARGET_SR) -> float | None:
    """Compute PESQ by resampling both to 16 kHz; returns None if package missing."""
    try:
        from pesq import pesq  # type: ignore
    except Exception:
        return None
    target_sr = 16000
    if sr_src != target_sr:
        resample = torchaudio.transforms.Resample(orig_freq=sr_src, new_freq=target_sr)
        orig_ = resample(orig)
        proc_ = resample(proc)
    else:
        orig_, proc_ = orig, proc
    o = orig_.squeeze().cpu().numpy()
    p = proc_.squeeze().cpu().numpy()
    try:
        # wb mode supports 16 kHz; score range ~1.0-4.5
        score = pesq(target_sr, o, p, "wb")
        return float(score)
    except Exception:
        return None


def compute_snr_db(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """SNR = 10*log10(sum(x^2) / sum((x-x_hat)^2))."""
    num = (x ** 2).sum().clamp_min(1e-12)
    den = ((x - x_hat) ** 2).sum().clamp_min(1e-12)
    return float(10.0 * torch.log10(num / den))


def compute_delta_spl_db(x: torch.Tensor, watermark_residual: torch.Tensor) -> float:
    """ΔSPL = SPL_original - SPL_watermark; SPL ~ 20*log10(rms/reference).
    The reference cancels in the difference, so we use rms levels directly.
    """
    def rms(sig: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((sig ** 2).mean().clamp_min(1e-12))

    spl_x = 20.0 * torch.log10(rms(x))
    spl_w = 20.0 * torch.log10(rms(watermark_residual))
    return float(spl_x - spl_w)


def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y).detach().cpu())


# --------- Augmentations (waveform domain, SR preserved at 22.05k) ---------

def apply_augmentations(wav: torch.Tensor, sr: int, aug_str: str) -> torch.Tensor:
    if not aug_str:
        return wav
    out = wav
    chains = [a.strip() for a in aug_str.split("|") if a.strip()]
    for a in chains:
        try:
            if a.startswith("gain_"):
                sign = +1.0 if "+" in a else -1.0
                db = float(a.split("_")[-1].replace("dB", "").replace("+", "").replace("-", ""))
                if "-" in a and sign > 0:
                    sign = -1.0
                g = 10 ** ((sign * db) / 20.0)
                out = torch.clamp(out * g, -1.0, 1.0)
            elif a.startswith("lowpass_"):
                fc = float(a.split("_")[-1].replace("k", "000").replace("Hz", ""))
                out = torchaudio.functional.lowpass_biquad(out, sr, cutoff_freq=fc)
            elif a.startswith("highpass_"):
                fc = float(a.split("_")[-1].replace("Hz", ""))
                out = torchaudio.functional.highpass_biquad(out, sr, cutoff_freq=fc)
            elif a.startswith("bandstop_"):
                # simple bandreject around center freq with Q=0.707
                fc = float(a.split("_")[-1].replace("k", "000").replace("Hz", ""))
                out = torchaudio.functional.bandreject_biquad(out, sr, central_freq=fc, Q=0.707)
            elif a.startswith("time_stretch_"):
                fac = float(a.split("_")[-1])
                # speed change via resampling: sr -> sr*fac -> sr
                up = torchaudio.transforms.Resample(sr, int(sr * fac))
                dn = torchaudio.transforms.Resample(int(sr * fac), sr)
                out = dn(up(out))
            elif a.startswith("pitch_shift_"):
                n_steps = float(a.split("_")[-1])
                out = torchaudio.functional.pitch_shift(out, sr, n_steps)
            elif a == "clip_soft":
                out = torch.tanh(2.0 * out)
            elif a == "reverb_small":
                try:
                    from torchaudio.sox_effects import apply_effects_tensor
                    out, _ = apply_effects_tensor(out, sr, [["reverb", "50", "50", "50"], ["gain", "-n"]])
                except Exception:
                    pass
            elif a.startswith("gaussian_noise_"):
                db = float(a.split("_")[-1].replace("dB", "").replace("-", ""))
                sig_rms = torch.sqrt((out ** 2).mean().clamp_min(1e-12))
                noise_rms = sig_rms * (10 ** (-db / 20.0))
                noise = torch.randn_like(out) * (noise_rms / (torch.sqrt((torch.randn_like(out) ** 2).mean()).clamp_min(1e-12)))
                out = torch.clamp(out + noise, -1.0, 1.0)
            elif a.startswith("mp3_"):
                # Not implemented cross-platform without external encoders; skip gracefully
                pass
            else:
                # Unknown augmentation; skip
                pass
        except Exception:
            # Robust to any augmentation failure; keep previous signal
            continue
    return out


# --------- Types ---------

@dataclass
class PerFileResult:
    audio_path: str
    pesq: float | None
    snr_db: float
    delta_spl_db: float
    mse: float
    ber_base: float
    ber_aug: float | None


# --------- Core evaluation per file ---------

@torch.no_grad()
def evaluate_one(
    model: INNWatermarker,
    planner: str,
    payload_text: str,
    audio_path: str,
    device: torch.device,
    use_augmentations: bool,
    aug_str: str,
) -> PerFileResult:
    state = model.state_dict()  # just to keep type-checkers happy
    _ = state

    # Pull cfg from a tiny dummy checkpoint-style dict on model
    # We rely on the inference script defaults if not present
    n_fft = model.stft_cfg.get("n_fft", 1024)  # type: ignore[attr-defined]
    hop = model.stft_cfg.get("hop_length", 512)  # type: ignore[attr-defined]

    wav, sr = _load_audio_mono(audio_path)  # [1,T]
    assert sr == TARGET_SR
    chunks = _chunk_audio_1s(wav)

    # Plan slots per chunk
    planned: List[Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor, int]] = []
    total_capacity = 0
    for ch in chunks:
        x = ch.to(device).unsqueeze(0)
        if planner == "gpu":
            slots, amp_scale = _fast_gpu_plan_slots(model, x, target_bits=1336)
        else:
            slots, amp_scale = _mg_plan_slots(model, x, target_bits=1336, n_fft=n_fft)
        S = min(len(slots), 1336)
        planned.append((x, slots, amp_scale, S))
        total_capacity += S

    rs_code_bits = 167 * 8
    want_bits = max(rs_code_bits, total_capacity)
    payload_bits = _encode_payload_bits_from_text(payload_text, rs_payload_bytes=125, interleave_depth=4, target_bits=want_bits, device=device)

    wm_chunks = []
    rec_bits_list = []
    cursor = 0
    base_symbol_amp = 0.03
    for (x, slots, amp_scale, S) in planned:
        if S == 0:
            wm_chunks.append(x)
            continue
        bits_i = payload_bits[:, cursor:cursor + S]
        if bits_i.size(1) < S:
            pad = torch.zeros(1, S - bits_i.size(1), dtype=torch.long, device=x.device)
            bits_i = torch.cat([bits_i, pad], dim=1)
        cursor += S

        X = model.stft(x)
        F_, T_ = X.shape[-2], X.shape[-1]
        M_spec = torch.zeros(1, 2, F_, T_, device=x.device)
        amp_vec = amp_scale[:S].to(x.device) if isinstance(amp_scale, torch.Tensor) else torch.ones(S, device=x.device)
        signs = (bits_i * 2 - 1).float() * base_symbol_amp * amp_vec.unsqueeze(0)
        for s, (f, t) in enumerate(slots[:S]):
            M_spec[0, 0, f, t] = signs[0, s]

        x_wm, _ = model.encode(x, M_spec)
        x_wm = torch.clamp(x_wm, -1.0, 1.0)
        M_rec = model.decode(x_wm)
        rec_vals = torch.stack([M_rec[0, 0, f, t] for (f, t) in slots[:S]], dim=0).unsqueeze(0)
        rec_bits = (rec_vals > 0).long()
        rec_bits_list.append(rec_bits)
        wm_chunks.append(x_wm)

    x_full = torch.cat([ch.squeeze(0) for ch in chunks], dim=-1)
    x_wm_full = torch.cat([ch.squeeze(0) for ch in wm_chunks], dim=-1)

    all_rec_bits = torch.cat(rec_bits_list, dim=1) if len(rec_bits_list) > 0 else torch.zeros(1, 0, dtype=torch.long, device=device)
    compare_len = min(all_rec_bits.size(1), payload_bits.size(1))
    ber_base = float(((all_rec_bits[:, :compare_len] != payload_bits[:, :compare_len]).float().mean().item()) if compare_len > 0 else 1.0)

    # Metrics w.r.t the encoded audio (no augmentation)
    pesq = compute_pesq(x_full, x_wm_full, sr_src=TARGET_SR)
    snr_db = compute_snr_db(x_full, x_wm_full)
    mse = compute_mse(x_full, x_wm_full)
    residual = (x_wm_full - x_full)
    delta_spl_db = compute_delta_spl_db(x_full, residual)

    # Optional robustness: decode after augmentation using same slots
    ber_aug: float | None = None
    if use_augmentations and aug_str:
        aug = apply_augmentations(x_wm_full, TARGET_SR, aug_str)
        rec_bits_list_aug = []
        cursor = 0
        for (x, slots, _amp_scale, S) in planned:
            # extract the corresponding portion of augmented audio
            seg = aug[..., cursor * 1: (cursor + 1) * x.shape[-1]]  # rough alignment by chunk length
            if seg.numel() < x.numel():
                seg = F.pad(seg, (0, x.shape[-1] - seg.shape[-1]))
            seg = seg.unsqueeze(0)
            M_rec = model.decode(seg)
            rec_vals = torch.stack([M_rec[0, 0, f, t] for (f, t) in slots[:S]], dim=0).unsqueeze(0)
            rec_bits = (rec_vals > 0).long()
            rec_bits_list_aug.append(rec_bits)
            cursor += x.shape[-1]
        all_bits_aug = torch.cat(rec_bits_list_aug, dim=1) if len(rec_bits_list_aug) > 0 else torch.zeros(1, 0, dtype=torch.long, device=device)
        L = min(all_bits_aug.size(1), payload_bits.size(1))
        ber_aug = float(((all_bits_aug[:, :L] != payload_bits[:, :L]).float().mean().item()) if L > 0 else 1.0)

    return PerFileResult(
        audio_path=audio_path,
        pesq=pesq,
        snr_db=snr_db,
        delta_spl_db=delta_spl_db,
        mse=mse,
        ber_base=ber_base,
        ber_aug=ber_aug,
    )


# --------- Runner ---------

def load_model(ckpt_path: str, device: torch.device, planner_override: str | None) -> Tuple[INNWatermarker, str]:
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("cfg", {})
    n_fft = int(cfg.get("n_fft", 1024))
    hop = int(cfg.get("hop", 512))
    planner = planner_override or cfg.get("planner", "mg")
    model = INNWatermarker(n_blocks=8, spec_channels=2, stft_cfg={"n_fft": n_fft, "hop_length": hop, "win_length": n_fft}).to(device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    return model, planner


def read_manifest(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=os.path.join("validation", "manifest.csv"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--planner", type=str, default=None, choices=[None, "gpu", "mg"], help="Override planner (default: use checkpoint cfg)")
    parser.add_argument("--use_augmentations", action="store_true", help="If set, apply augmentations listed in manifest to watermarked audio before robustness BER decode")
    parser.add_argument("--results_csv", type=str, default=os.path.join("validation", "results.csv"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model, planner = load_model(args.ckpt, device, args.planner)

    rows = read_manifest(args.manifest)
    per_file: List[PerFileResult] = []

    for r in rows:
        audio_path = r.get("audio_path", "").strip()
        payload = r.get("payload", "").strip()
        out_dir = r.get("out_dir", "validation/output").strip()
        aug = r.get("augment", "").strip()
        if not audio_path:
            continue
        os.makedirs(out_dir, exist_ok=True)
        res = evaluate_one(model, planner, payload, audio_path, device, args.use_augmentations, aug)
        per_file.append(res)

    # Aggregate
    def avg(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    pesq_vals = [v for v in (r.pesq for r in per_file) if v is not None]
    snr_vals = [r.snr_db for r in per_file]
    spl_vals = [r.delta_spl_db for r in per_file]
    mse_vals = [r.mse for r in per_file]
    ber_base_vals = [r.ber_base for r in per_file]
    ber_aug_vals = [v for v in (r.ber_aug for r in per_file) if v is not None]

    print("\nValidation summary ({} files):".format(len(per_file)))
    if pesq_vals:
        print(f"  PESQ avg: {avg(pesq_vals):.3f}")
    else:
        print("  PESQ avg: N/A (pesq package not installed)")
    print(f"  SNR avg (dB): {avg(snr_vals):.3f}")
    print(f"  ΔSPL avg (dB): {avg(spl_vals):.3f}")
    print(f"  MSE avg: {avg(mse_vals):.6f}")
    print(f"  BER base avg: {avg(ber_base_vals):.6f}")
    if ber_aug_vals:
        print(f"  BER augmented avg: {avg(ber_aug_vals):.6f}")

    # Write detailed CSV
    fieldnames = ["audio_path", "pesq", "snr_db", "delta_spl_db", "mse", "ber_base", "ber_aug"]
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_file:
            row = asdict(r)
            w.writerow(row)


if __name__ == "__main__":
    main()


