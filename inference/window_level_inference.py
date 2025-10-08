#!/usr/bin/env python3
"""
Window-level inference script for micro-chunking system.
Implements micro-windower, sync alignment, and coordinate-independent decoding.

Usage:
    python window_level_inference.py --input audio.wav --output watermarked.wav --payload "test message"
"""

from __future__ import annotations
import argparse
import os
import sys
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.inn_encoder_decoder import INNWatermarker
from pipeline.micro_chunking import MicroChunker, DeterministicMapper, fuse_overlapping_windows
from pipeline.sync_anchors import SyncAnchor, SyncDetector
from pipeline.ingest_and_chunk import (
    rs_encode_167_125,
    rs_decode_167_125,
    interleave_bytes,
    deinterleave_bytes,
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
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav, sr = _resample_if_needed(wav, sr)
    return wav, sr


def _chunk_audio_1s(wav: torch.Tensor) -> list[torch.Tensor]:
    """Split audio into 1-second chunks."""
    T = wav.size(-1)
    chunks: list[torch.Tensor] = []
    cursor = 0
    while cursor < T:
        end = min(cursor + CHUNK_SAMPLES, T)
        ch = wav[..., cursor:end]
        if ch.size(-1) < CHUNK_SAMPLES:
            ch = F.pad(ch, (0, CHUNK_SAMPLES - ch.size(-1)))
        chunks.append(ch)
        cursor += CHUNK_SAMPLES
    if len(chunks) == 0:
        chunks = [F.pad(wav[..., :0], (0, CHUNK_SAMPLES))]
    return chunks


def _encode_payload_bits_from_text(text: str, payload_bytes: int, 
                                  interleave_depth: int, target_bits: int,
                                  device: torch.device) -> torch.Tensor:
    """Encode text payload to bits with RS encoding."""
    # Convert text to bytes
    text_bytes = text.encode('utf-8')
    
    # Pad or truncate to payload_bytes
    if len(text_bytes) >= payload_bytes:
        payload = text_bytes[:payload_bytes]
    else:
        payload = text_bytes + b'\x00' * (payload_bytes - len(text_bytes))
    
    # RS encode and interleave
    rs_encoded = rs_encode_167_125(payload)
    interleaved = interleave_bytes(rs_encoded, interleave_depth)
    
    # Convert to bits
    bits = []
    for b in interleaved:
        for k in range(8):
            bits.append((b >> k) & 1)
    
    bits_tensor = torch.tensor(bits, dtype=torch.long, device=device)
    
    # Pad or truncate to target_bits
    if bits_tensor.size(0) >= target_bits:
        return bits_tensor[:target_bits].unsqueeze(0)
    else:
        pad = torch.zeros(target_bits - bits_tensor.size(0), dtype=torch.long, device=device)
        return torch.cat([bits_tensor, pad], dim=0).unsqueeze(0)


def _decode_payload_bits_to_text(bits: torch.Tensor, interleave_depth: int,
                                payload_bytes: int) -> str:
    """Decode bits to text with RS decoding."""
    if bits.numel() == 0:
        return ""
    
    # Convert bits to bytes
    bits_flat = bits.flatten()
    by = bytearray()
    for i in range(0, len(bits_flat), 8):
        if i + 8 > len(bits_flat):
            break
        b = 0
        for k in range(8):
            b |= (int(bits_flat[i + k]) & 1) << k
        by.append(b)
    
    # Deinterleave and RS decode
    try:
        deinterleaved = deinterleave_bytes(bytes(by), interleave_depth)
        decoded = rs_decode_167_125(deinterleaved)
        text = decoded[:payload_bytes].decode('utf-8', errors='ignore').rstrip('\x00')
        return text
    except Exception:
        return ""


class WindowLevelInference:
    """Window-level inference with micro-chunking and sync alignment."""
    
    def __init__(self, model: INNWatermarker, window_ms: int = 15, 
                 overlap_ratio: float = 0.5, mapper_seed: int = 42,
                 sync_strength: float = 0.1, n_fft: int = 1024, hop: int = 512):
        """
        Args:
            model: Trained INNWatermarker model
            window_ms: Micro-chunk duration in milliseconds
            overlap_ratio: Overlap between consecutive chunks
            mapper_seed: Seed for deterministic mapping
            sync_strength: Strength of sync pattern
            n_fft: FFT size
            hop: Hop length
        """
        self.model = model
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.mapper_seed = mapper_seed
        self.sync_strength = sync_strength
        self.n_fft = n_fft
        self.hop = hop
        
        # Initialize components
        self.micro_chunker = MicroChunker(window_ms, overlap_ratio, TARGET_SR)
        self.mapper = DeterministicMapper(mapper_seed, n_fft, hop, TARGET_SR, window_ms)
        self.sync_anchor = SyncAnchor(sync_length_ms=50, sr=TARGET_SR, n_fft=n_fft, hop=hop)
        self.sync_detector = SyncDetector(self.sync_anchor, model)
        
    def encode_audio(self, audio_1s: torch.Tensor, payload_bits: torch.Tensor,
                    target_bits_per_window: int, base_symbol_amp: float = 0.1) -> torch.Tensor:
        """
        Encode payload into 1s audio segment using window-level processing.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 22050 samples
            payload_bits: Payload bits [1, total_bits]
            target_bits_per_window: Target bits per micro-chunk
            base_symbol_amp: Base symbol amplitude
            
        Returns:
            Watermarked audio [1, T]
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        # Add sync pattern
        audio_with_sync = self.sync_anchor.embed_sync_pattern(audio_1s, self.sync_strength)
        
        # Create micro-chunks
        windows = self.micro_chunker.chunk_1s_segment(audio_with_sync)
        n_windows = len(windows)
        
        # Calculate bits per window
        total_payload_bits = payload_bits.size(1)
        bits_per_window = total_payload_bits // n_windows
        if bits_per_window == 0:
            bits_per_window = 1
        
        # Process each window
        watermarked_windows = []
        cursor = 0
        
        for i, window in enumerate(windows):
            # Get bits for this window
            window_bits = payload_bits[:, cursor:cursor + min(bits_per_window, target_bits_per_window)]
            if window_bits.size(1) < target_bits_per_window:
                pad = torch.zeros(1, target_bits_per_window - window_bits.size(1), 
                                dtype=torch.long, device=window.device)
                window_bits = torch.cat([window_bits, pad], dim=1)
            
            # Map window to slots
            slots = self.mapper.map_window_to_slots(
                window_idx=i,
                n_windows=n_windows,
                target_bits=target_bits_per_window,
                audio_content=window
            )
            
            # Apply psychoacoustic gating
            from pipeline.micro_chunking import apply_psychoacoustic_gate
            masked_slots = apply_psychoacoustic_gate(window, slots, self.model, self.n_fft, self.hop)
            
            # Build message spectrogram
            M_spec = self._build_message_spec(window, masked_slots, window_bits, base_symbol_amp)
            
            # Encode
            window_wm, _ = self.model.encode(window, M_spec)
            window_wm = torch.clamp(window_wm, -1.0, 1.0)
            
            watermarked_windows.append(window_wm)
            cursor += bits_per_window
        
        # Fuse overlapping windows
        watermarked_audio = self._fuse_windows(watermarked_windows)
        
        return watermarked_audio
    
    def decode_audio(self, audio_1s: torch.Tensor, target_bits_per_window: int,
                    expected_payload_bytes: int = 64) -> Tuple[str, torch.Tensor, Dict]:
        """
        Decode payload from 1s audio segment using window-level processing.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 22050 samples
            target_bits_per_window: Target bits per micro-chunk
            expected_payload_bytes: Expected payload size in bytes
            
        Returns:
            (decoded_text, recovered_bits, metadata) tuple
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        # Detect sync pattern and align
        offset, confidence = self.sync_detector.detect_robust(audio_1s, threshold=0.05)
        aligned_audio = self.sync_anchor.align_audio(audio_1s, offset)
        
        # Create micro-chunks
        windows = self.micro_chunker.chunk_1s_segment(aligned_audio)
        n_windows = len(windows)
        
        # Process each window
        window_bits_list = []
        metadata = {
            'sync_offset': offset,
            'sync_confidence': confidence,
            'n_windows': n_windows,
            'window_ms': self.window_ms,
            'overlap_ratio': self.overlap_ratio
        }
        
        for i, window in enumerate(windows):
            # Map window to slots (same as encoding)
            slots = self.mapper.map_window_to_slots(
                window_idx=i,
                n_windows=n_windows,
                target_bits=target_bits_per_window,
                audio_content=window
            )
            
            # Apply psychoacoustic gating
            from pipeline.micro_chunking import apply_psychoacoustic_gate
            masked_slots = apply_psychoacoustic_gate(window, slots, self.model, self.n_fft, self.hop)
            
            if not masked_slots:
                window_bits_list.append(torch.tensor([], dtype=torch.long, device=window.device))
                continue
            
            # Decode
            M_rec = self.model.decode(window)
            
            # Extract bits from slots
            rec_vals = torch.zeros(1, len(masked_slots), device=window.device)
            for s, (f, t) in enumerate(masked_slots):
                if f < M_rec.size(2) and t < M_rec.size(3):
                    rec_vals[0, s] = M_rec[0, 0, f, t]
            
            # Convert to bits
            rec_bits = (rec_vals > 0).long()
            window_bits_list.append(rec_bits)
        
        # Fuse overlapping windows
        if window_bits_list:
            fused_bits = fuse_overlapping_windows(window_bits_list, self.overlap_ratio)
        else:
            fused_bits = torch.tensor([], dtype=torch.long, device=audio_1s.device)
        
        # Decode to text
        decoded_text = _decode_payload_bits_to_text(fused_bits, 4, expected_payload_bytes)
        
        return decoded_text, fused_bits, metadata
    
    def _build_message_spec(self, window: torch.Tensor, slots: List[Tuple[int, int]], 
                           bits: torch.Tensor, base_symbol_amp: float) -> torch.Tensor:
        """Build message spectrogram from window, slots, and bits."""
        if not slots or bits.numel() == 0:
            X = torch.stft(window, n_fft=self.n_fft, hop_length=self.hop, 
                          win_length=self.n_fft, return_complex=True)
            return torch.zeros_like(X)
        
        # Get STFT dimensions
        X = torch.stft(window, n_fft=self.n_fft, hop_length=self.hop, 
                      win_length=self.n_fft, return_complex=True)
        F_, T_ = X.shape[-2], X.shape[-1]
        
        # Create message spectrogram
        M_spec = torch.zeros(1, 2, F_, T_, device=window.device)
        
        # Place symbols at slots
        n_slots = min(len(slots), bits.size(1))
        for s in range(n_slots):
            f, t = slots[s]
            if f < F_ and t < T_:
                # Convert bit to symbol
                bit_val = bits[0, s].item()
                symbol = (bit_val * 2 - 1) * base_symbol_amp  # -amp or +amp
                
                # Place in real channel
                M_spec[0, 0, f, t] = symbol
        
        return M_spec
    
    def _fuse_windows(self, windows: List[torch.Tensor]) -> torch.Tensor:
        """Fuse overlapping windows back to 1s audio."""
        if not windows:
            return torch.zeros(1, CHUNK_SAMPLES, device=next(iter(windows)).device)
        
        # Simple overlap-add fusion
        # This is a simplified version - could be enhanced with proper windowing
        hop_samples = int(self.window_ms * TARGET_SR / 1000 * (1 - self.overlap_ratio))
        window_samples = windows[0].size(1)
        
        # Calculate total length
        total_length = (len(windows) - 1) * hop_samples + window_samples
        total_length = min(total_length, CHUNK_SAMPLES)
        
        # Initialize output
        output = torch.zeros(1, total_length, device=windows[0].device)
        
        # Overlap-add
        for i, window in enumerate(windows):
            start = i * hop_samples
            end = start + window_samples
            if start < total_length:
                actual_end = min(end, total_length)
                actual_window = window[:, :actual_end - start]
                output[:, start:actual_end] += actual_window
        
        # Normalize to prevent clipping
        output = torch.clamp(output, -1.0, 1.0)
        
        return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Window-level watermark inference")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output watermarked audio file")
    parser.add_argument("--payload", required=True, help="Payload text to embed")
    parser.add_argument("--model", default="window_level_checkpoints/window_level_best.pt", 
                       help="Model checkpoint path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--window_ms", type=int, default=15, help="Micro-chunk duration in ms")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="Overlap ratio")
    parser.add_argument("--target_bits_per_window", type=int, default=50, 
                       help="Target bits per micro-chunk")
    parser.add_argument("--base_symbol_amp", type=float, default=0.1, 
                       help="Base symbol amplitude")
    parser.add_argument("--sync_strength", type=float, default=0.1, 
                       help="Sync pattern strength")
    parser.add_argument("--decode_only", action="store_true", 
                       help="Only decode (don't encode)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    model = INNWatermarker(n_blocks=8, spec_channels=2, 
                          stft_cfg={"n_fft": 1024, "hop_length": 512, "win_length": 1024})
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()
    
    # Initialize inference
    inference = WindowLevelInference(
        model=model,
        window_ms=args.window_ms,
        overlap_ratio=args.overlap_ratio,
        mapper_seed=42,
        sync_strength=args.sync_strength
    )
    
    # Load audio
    print(f"Loading audio from {args.input}")
    wav, sr = _load_audio_mono(args.input)
    
    if sr != TARGET_SR:
        raise RuntimeError(f"Unexpected SR after resample: {sr}")
    
    # Split into 1s chunks
    chunks = _chunk_audio_1s(wav)
    print(f"Audio loaded: {len(chunks)} chunks")
    
    if not args.decode_only:
        # Encode
        print(f"Encoding payload: '{args.payload}'")
        
        # Encode payload
        payload_bits = _encode_payload_bits_from_text(
            args.payload, 64, 4, len(chunks) * args.target_bits_per_window, 
            device=torch.device(args.device)
        )
        
        # Process each chunk
        watermarked_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Get bits for this chunk
            start_bit = i * args.target_bits_per_window
            end_bit = min(start_bit + args.target_bits_per_window, payload_bits.size(1))
            chunk_bits = payload_bits[:, start_bit:end_bit]
            
            # Encode chunk
            chunk_wm = inference.encode_audio(
                chunk.to(args.device),
                chunk_bits,
                args.target_bits_per_window,
                args.base_symbol_amp
            )
            
            watermarked_chunks.append(chunk_wm.cpu())
        
        # Concatenate and save
        watermarked_audio = torch.cat(watermarked_chunks, dim=-1)
        torchaudio.save(args.output, watermarked_audio, TARGET_SR)
        print(f"Saved watermarked audio to {args.output}")
    
    else:
        # Decode
        print("Decoding watermarked audio...")
        
        decoded_texts = []
        all_metadata = []
        
        for i, chunk in enumerate(chunks):
            print(f"Decoding chunk {i+1}/{len(chunks)}")
            
            chunk_wm = chunk.to(args.device)
            decoded_text, recovered_bits, metadata = inference.decode_audio(
                chunk_wm, args.target_bits_per_window, 64
            )
            
            decoded_texts.append(decoded_text)
            all_metadata.append(metadata)
            
            print(f"  Sync offset: {metadata['sync_offset']}, confidence: {metadata['sync_confidence']:.3f}")
            print(f"  Decoded: '{decoded_text}'")
        
        # Combine all decoded text
        combined_text = "".join(decoded_texts)
        print(f"\nCombined decoded text: '{combined_text}'")
        
        # Print metadata summary
        avg_sync_conf = np.mean([m['sync_confidence'] for m in all_metadata])
        print(f"Average sync confidence: {avg_sync_conf:.3f}")


if __name__ == "__main__":
    main()
