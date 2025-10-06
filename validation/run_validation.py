#!/usr/bin/env python3
"""Validation harness for SoundSafe watermark decoder checkpoints.

Reads a manifest CSV (audio_path,payload,out_dir,augment), embeds payloads
into each clean clip with the trained decoder, applies optional audio attacks,
decodes the (clean + attacked) outputs, and records metrics in a summary CSV.

The script is intentionally defensive:
  * Missing audio files or load failures are logged and skipped.
  * Augmentation steps that cannot be executed (e.g., ffmpeg not available)
    are skipped gracefully without aborting the run.
  * Embed/decode subprocess failures are captured in the results table.

Usage example:

    python validation/run_validation.py \
        --manifest validation/manifest.csv \
        --ckpt checkpoints/inn_decode_best.pt \
        --results validation/results.csv

"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Helpers


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug or "clean"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"audio_path", "payload", "out_dir", "augment"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Embedding + decoding helpers


@dataclass
class EmbedResult:
    exit_code: int
    ber: float | None
    stdout: str
    stderr: str
    watermarked_audio: Path | None
    spec_image: Path | None


@dataclass
class DecodeResult:
    exit_code: int
    extracted_bits: int | None
    recovered_payload: str | None
    stdout: str
    stderr: str


def run_embed(
    inference_script: Path,
    ckpt: Path,
    audio_path: Path,
    payload: str,
    out_audio: Path,
    spec_path: Path,
    planner: str = "mg",
) -> EmbedResult:
    cmd = [
        sys.executable,
        str(inference_script),
        "--ckpt",
        str(ckpt),
        "--audio",
        str(audio_path),
        "--out",
        str(out_audio),
        "--spec_out",
        str(spec_path),
        "--payload",
        payload,
        "--planner",
        planner,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ber = None
    match = re.search(r"BER=([0-9.]+)", proc.stdout)
    if match:
        try:
            ber = float(match.group(1))
        except ValueError:
            ber = None
    return EmbedResult(
        exit_code=proc.returncode,
        ber=ber,
        stdout=proc.stdout,
        stderr=proc.stderr,
        watermarked_audio=out_audio if out_audio.exists() else None,
        spec_image=spec_path if spec_path.exists() else None,
    )


def run_decode(
    decode_script: Path,
    ckpt: Path,
    audio_path: Path,
    out_text: Path,
    planner: str = "mg",
) -> DecodeResult:
    cmd = [
        sys.executable,
        str(decode_script),
        "--ckpt",
        str(ckpt),
        "--audio",
        str(audio_path),
        "--out_text",
        str(out_text),
        "--planner",
        planner,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    bits = None
    recovered = None
    match_bits = re.search(r"Extracted bits:\s*(\d+)", proc.stdout)
    if match_bits:
        bits = int(match_bits.group(1))
    match_payload = re.search(r"Recovered payload \(utf-8, truncated\):\s*(.*)", proc.stdout)
    if match_payload:
        recovered = match_payload.group(1).strip()
    if recovered is None and out_text.exists():
        try:
            recovered = out_text.read_text(encoding="utf-8")[:128]
        except Exception:
            recovered = None
    return DecodeResult(
        exit_code=proc.returncode,
        extracted_bits=bits,
        recovered_payload=recovered,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# ---------------------------------------------------------------------------
# Augmentations


def _load_audio(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    return wav, sr


def _save_audio(path: Path, wav: torch.Tensor, sr: int) -> None:
    torchaudio.save(str(path), wav, sr)


def _apply_gain(wav: torch.Tensor, db: float) -> torch.Tensor:
    gain = 10.0 ** (db / 20.0)
    return torch.clamp(wav * gain, -1.0, 1.0)


def _apply_noise(wav: torch.Tensor, db: float) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt()
    if rms <= 1e-9:
        noise_scale = 10.0 ** (db / 20.0)
    else:
        target_noise = rms * (10.0 ** (db / 20.0))
        noise_scale = target_noise
    noise = torch.randn_like(wav) * noise_scale
    return torch.clamp(wav + noise, -1.0, 1.0)


def _apply_time_stretch(wav: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    if factor <= 0:
        return wav
    new_sr = int(sr * factor)
    resampled = torchaudio.functional.resample(wav, sr, new_sr)
    stretched = torchaudio.functional.resample(resampled, new_sr, sr)
    return stretched


def _apply_pitch_shift(wav: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    return torchaudio.functional.pitch_shift(wav, sr, n_steps=float(semitones))


def _apply_lowpass(wav: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    return torchaudio.functional.lowpass_biquad(wav, sr, cutoff_hz)


def _apply_highpass(wav: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    return torchaudio.functional.highpass_biquad(wav, sr, cutoff_hz)


def _apply_bandstop(wav: torch.Tensor, sr: int, center_hz: float) -> torch.Tensor:
    q = 0.707
    return torchaudio.functional.band_biquad(wav, sr, center_hz, q=q, noise=True)


def _apply_soft_clip(wav: torch.Tensor) -> torch.Tensor:
    return torch.tanh(wav * 2.0)


def _apply_reverb_small(wav: torch.Tensor) -> torch.Tensor:
    impulse = torch.tensor([[1.0, 0.6, 0.3, 0.15]], dtype=wav.dtype, device=wav.device)
    convolved = torchaudio.functional.fftconvolve(wav, impulse)
    return torch.clamp(convolved[..., : wav.shape[-1]], -1.0, 1.0)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _apply_mp3_roundtrip(src: Path, dst: Path, bitrate: str) -> bool:
    if not _ffmpeg_available():
        logging.warning("ffmpeg not available; skipping mp3 compression attack")
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mp3 = Path(tmpdir) / "temp.mp3"
        cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", bitrate, str(tmp_mp3)]
        cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_mp3), str(dst)]
        for cmd in (cmd1, cmd2):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                logging.warning("ffmpeg failed (%s): %s", " ".join(cmd), proc.stderr.strip())
                return False
    return True


def apply_augment_chain(src: Path, dst: Path, chain: Sequence[str]) -> bool:
    if not chain:
        shutil.copyfile(src, dst)
        return True

    current = src
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for idx, token in enumerate(chain):
            token = token.strip()
            if not token:
                continue
            next_path = tmpdir_path / f"step_{idx}.wav"
            if token.startswith("mp3_"):
                bitrate = token.split("_", 1)[1].replace("kbps", "k")
                bitrate = bitrate.rstrip("b") if bitrate.endswith("bps") else bitrate
                bitrate = bitrate if bitrate.endswith("k") else f"{bitrate}"
                success = _apply_mp3_roundtrip(current, next_path, bitrate)
                if not success:
                    return False
            else:
                try:
                    wav, sr = _load_audio(current)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to load audio for augment '%s': %s", token, exc)
                    return False

                wav = _apply_token(wav, sr, token)
                try:
                    _save_audio(next_path, wav, sr)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to save augmented audio '%s': %s", token, exc)
                    return False
            current = next_path
        shutil.copyfile(current, dst)
    return True


def _apply_token(wav: torch.Tensor, sr: int, token: str) -> torch.Tensor:
    try:
        if token.startswith("gaussian_noise_"):
            level = token.split("_", 2)[-1].replace("dB", "")
            return _apply_noise(wav, float(level))
        if token.startswith("gain_"):
            level = token.split("_", 1)[1].replace("dB", "")
            return _apply_gain(wav, float(level))
        if token.startswith("time_stretch_"):
            factor = float(token.split("_", 1)[1])
            return _apply_time_stretch(wav, sr, factor)
        if token.startswith("pitch_shift_"):
            steps = float(token.split("_", 1)[1])
            return _apply_pitch_shift(wav, sr, steps)
        if token.startswith("lowpass_"):
            val = token.split("_", 1)[1]
            cutoff = _parse_hz(val)
            return _apply_lowpass(wav, sr, cutoff)
        if token.startswith("highpass_"):
            val = token.split("_", 1)[1]
            cutoff = _parse_hz(val)
            return _apply_highpass(wav, sr, cutoff)
        if token.startswith("bandstop_"):
            val = token.split("_", 1)[1]
            center = _parse_hz(val)
            return _apply_bandstop(wav, sr, center)
        if token == "clip_soft":
            return _apply_soft_clip(wav)
        if token == "reverb_small":
            return _apply_reverb_small(wav)
    except Exception as exc:
        logging.warning("Augmentation '%s' failed: %s", token, exc)
        return wav

    logging.warning("Unknown augmentation '%s'; skipping", token)
    return wav


def _parse_hz(text: str) -> float:
    text = text.lower()
    if text.endswith("hz"):
        text = text[:-2]
    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1000.0
        text = text[:-1]
    return float(text) * multiplier


# ---------------------------------------------------------------------------
# Result tracking


@dataclass
class ValidationRecord:
    audio_path: str
    payload: str
    attack_label: str
    augment_chain: str
    embed_exit: int
    embed_ber: float | None
    decode_exit: int | None
    decode_bits: int | None
    decode_payload: str | None
    notes: str
    watermarked_audio: str | None
    attacked_audio: str | None


def write_results(path: Path, records: Iterable[ValidationRecord]) -> None:
    fieldnames = [
        "audio_path",
        "payload",
        "attack_label",
        "augment_chain",
        "embed_exit",
        "embed_ber",
        "decode_exit",
        "decode_bits",
        "decode_payload",
        "notes",
        "watermarked_audio",
        "attacked_audio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "audio_path": rec.audio_path,
                "payload": rec.payload,
                "attack_label": rec.attack_label,
                "augment_chain": rec.augment_chain,
                "embed_exit": rec.embed_exit,
                "embed_ber": f"{rec.embed_ber:.6f}" if rec.embed_ber is not None else "",
                "decode_exit": rec.decode_exit if rec.decode_exit is not None else "",
                "decode_bits": rec.decode_bits if rec.decode_bits is not None else "",
                "decode_payload": rec.decode_payload or "",
                "notes": rec.notes,
                "watermarked_audio": rec.watermarked_audio or "",
                "attacked_audio": rec.attacked_audio or "",
            })


# ---------------------------------------------------------------------------
# Main driver


def process_manifest(
    manifest: Path,
    ckpt: Path,
    inference_script: Path,
    decode_script: Path,
    results_path: Path,
    planner: str = "mg",
) -> list[ValidationRecord]:
    rows = _read_manifest(manifest)
    root = manifest.parent.parent if manifest.parts[-2] == "validation" else Path.cwd()
    records: list[ValidationRecord] = []

    for idx, row in enumerate(rows, start=1):
        audio_rel = row.get("audio_path", "")
        payload = row.get("payload", "")
        out_dir_rel = row.get("out_dir", "validation/output")
        augment_str = row.get("augment", "")

        audio_path = Path(audio_rel)
        if not audio_path.is_absolute():
            audio_path = Path.cwd() / audio_path
        if not audio_path.exists():
            logging.warning("[%03d] Missing audio file: %s", idx, audio_path)
            records.append(
                ValidationRecord(
                    audio_path=str(audio_path),
                    payload=payload,
                    attack_label="missing",
                    augment_chain=augment_str,
                    embed_exit=-1,
                    embed_ber=None,
                    decode_exit=None,
                    decode_bits=None,
                    decode_payload=None,
                    notes="audio_not_found",
                    watermarked_audio=None,
                    attacked_audio=None,
                )
            )
            continue

        clip_out_dir = Path(out_dir_rel)
        if not clip_out_dir.is_absolute():
            clip_out_dir = Path.cwd() / clip_out_dir
        clip_out_dir = clip_out_dir / audio_path.stem
        _ensure_dir(clip_out_dir)

        wm_audio = clip_out_dir / "clean_wm.wav"
        spec_path = clip_out_dir / "clean_spec.png"

        # If source is not WAV, try to transcode to WAV to avoid codec/backend issues
        src_for_embed = audio_path
        if audio_path.suffix.lower() != ".wav" and _ffmpeg_available():
            trans_wav = clip_out_dir / "source_converted.wav"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(audio_path), str(trans_wav)
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and trans_wav.exists():
                src_for_embed = trans_wav
            else:
                logging.warning("[%03d] ffmpeg failed to convert %s to wav: %s", idx, audio_path, proc.stderr.strip())

        logging.info("[%03d] Embedding payload into %s", idx, src_for_embed)
        embed_res = run_embed(
            inference_script=inference_script,
            ckpt=ckpt,
            audio_path=src_for_embed,
            payload=payload,
            out_audio=wm_audio,
            spec_path=spec_path,
            planner=planner,
        )

        record_base = ValidationRecord(
            audio_path=str(audio_path),
            payload=payload,
            attack_label="clean",
            augment_chain="",
            embed_exit=embed_res.exit_code,
            embed_ber=embed_res.ber,
            decode_exit=None,
            decode_bits=None,
            decode_payload=None,
            notes="",
            watermarked_audio=str(embed_res.watermarked_audio) if embed_res.watermarked_audio else None,
            attacked_audio=str(embed_res.watermarked_audio) if embed_res.watermarked_audio else None,
        )

        if embed_res.exit_code != 0 or not embed_res.watermarked_audio:
            record_base.notes = "embed_failed"
            record_base.decode_exit = None
            # Persist logs to assist debugging
            try:
                (clip_out_dir / "embed_stdout.txt").write_text(embed_res.stdout, encoding="utf-8")
                (clip_out_dir / "embed_stderr.txt").write_text(embed_res.stderr, encoding="utf-8")
            except Exception:
                pass
            # Log a short snippet for quick triage
            err_snip = (embed_res.stderr or "").strip().splitlines()[-1:] or [""]
            logging.warning("[%03d] Embed stderr tail: %s", idx, err_snip[0])
            records.append(record_base)
            logging.warning(
                "[%03d] Embed failed for %s (exit %s)", idx, src_for_embed, embed_res.exit_code
            )
            continue

        # Decode clean watermarked audio
        decode_clean_path = clip_out_dir / "clean_recovered.txt"
        logging.info("[%03d] Decoding clean output", idx)
        decode_res = run_decode(
            decode_script=decode_script,
            ckpt=ckpt,
            audio_path=wm_audio,
            out_text=decode_clean_path,
            planner=planner,
        )
        record_clean = ValidationRecord(
            audio_path=record_base.audio_path,
            payload=payload,
            attack_label="clean",
            augment_chain="",
            embed_exit=embed_res.exit_code,
            embed_ber=embed_res.ber,
            decode_exit=decode_res.exit_code,
            decode_bits=decode_res.extracted_bits,
            decode_payload=decode_res.recovered_payload,
            notes="clean_decode_ok" if decode_res.exit_code == 0 else "clean_decode_failed",
            watermarked_audio=record_base.watermarked_audio,
            attacked_audio=record_base.attacked_audio,
        )
        records.append(record_clean)

        # Apply augmentation chain if provided
        augment_chain = [tok for tok in augment_str.split("|") if tok]
        if augment_chain:
            attack_label = _slugify("_".join(augment_chain))
            attack_audio_path = clip_out_dir / f"attack_{attack_label}.wav"
            logging.info("[%03d] Applying augmentations %s", idx, augment_chain)
            success = apply_augment_chain(wm_audio, attack_audio_path, augment_chain)
            if not success or not attack_audio_path.exists():
                logging.warning("[%03d] Augmentation failed for %s", idx, augment_chain)
                records.append(
                    ValidationRecord(
                        audio_path=str(audio_path),
                        payload=payload,
                        attack_label=attack_label,
                        augment_chain="|".join(augment_chain),
                        embed_exit=embed_res.exit_code,
                        embed_ber=embed_res.ber,
                        decode_exit=None,
                        decode_bits=None,
                        decode_payload=None,
                        notes="augment_failed",
                        watermarked_audio=str(wm_audio),
                        attacked_audio=str(attack_audio_path) if attack_audio_path.exists() else None,
                    )
                )
            else:
                logging.info("[%03d] Decoding attacked output %s", idx, attack_label)
                decode_attack_path = clip_out_dir / f"attack_{attack_label}_recovered.txt"
                decode_attack = run_decode(
                    decode_script=decode_script,
                    ckpt=ckpt,
                    audio_path=attack_audio_path,
                    out_text=decode_attack_path,
                    planner=planner,
                )
                records.append(
                    ValidationRecord(
                        audio_path=str(audio_path),
                        payload=payload,
                        attack_label=attack_label,
                        augment_chain="|".join(augment_chain),
                        embed_exit=embed_res.exit_code,
                        embed_ber=embed_res.ber,
                        decode_exit=decode_attack.exit_code,
                        decode_bits=decode_attack.extracted_bits,
                        decode_payload=decode_attack.recovered_payload,
                        notes="attack_decode_ok" if decode_attack.exit_code == 0 else "attack_decode_failed",
                        watermarked_audio=str(wm_audio),
                        attacked_audio=str(attack_audio_path),
                    )
                )
        else:
            logging.debug("[%03d] No augmentations listed; skipping attack phase", idx)

    write_results(results_path, records)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run validation suite for decoder checkpoint")
    parser.add_argument("--manifest", type=Path, default=Path("validation/manifest.csv"))
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint to validate")
    parser.add_argument("--inference-script", type=Path, default=Path("inference/new_inference.py"))
    parser.add_argument("--decode-script", type=Path, default=Path("inference/decode_watermark.py"))
    parser.add_argument("--results", type=Path, default=Path("validation/results.csv"))
    parser.add_argument("--planner", type=str, default="mg", choices=["mg", "gpu"], help="Planner to pass through")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        records = process_manifest(
            manifest=args.manifest,
            ckpt=args.ckpt,
            inference_script=args.inference_script,
            decode_script=args.decode_script,
            results_path=args.results,
            planner=args.planner,
        )
    except Exception as exc:
        logging.exception("Validation run failed: %s", exc)
        return 1

    logging.info("Validation complete: %d records written to %s", len(records), args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


