# soundsafe/pipeline/micro_chunking.py
# Micro-chunking system with overlap for robust watermarking
# Implements 10-20ms windows with 50% overlap per 1-second segment

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np
import hashlib
from pipeline.error_correction import fuse_symbol_evidence_with_crc, compute_crc_agreement_rate


class MicroChunker:
    """
    Splits 1-second audio segments into overlapping micro-chunks.
    Each micro-chunk is 10-20ms with 50% overlap for redundancy.
    1 s at 44.1 kHz is 44,100 samples.
    """
    
    def __init__(self, window_ms: int = 15, overlap_ratio: float = 0.5, sr: int = 44100):
        """
        Args:
            window_ms: Micro-chunk duration in milliseconds (10-20ms recommended)
            overlap_ratio: Overlap between consecutive chunks (0.5 = 50% overlap)
            sr: Sample rate (default 44100 to match INN model)
        """
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.sr = sr
        
        # Calculate samples
        self.window_samples = int(window_ms * sr / 1000)  # ~662 samples for 15ms @ 44.1kHz
        self.hop_samples = int(self.window_samples * (1 - overlap_ratio))  # ~331 samples for 50% overlap
        
        # Ensure we have enough samples for at least one window
        self.min_samples = self.window_samples
        
    def chunk_1s_segment(self, audio_1s: torch.Tensor) -> List[torch.Tensor]:
        """
        Split 1-second audio into overlapping micro-chunks.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 44100 samples
            
        Returns:
            List of micro-chunk tensors, each [1, window_samples]
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
            
        T = audio_1s.size(1)
        if T < self.min_samples:
            # Pad if too short
            audio_1s = F.pad(audio_1s, (0, self.min_samples - T))
            T = self.min_samples
            
        chunks = []
        start = 0
        
        while start < T:
            end = min(start + self.window_samples, T)
            chunk = audio_1s[:, start:end]
            
            # Pad last chunk if needed
            if chunk.size(1) < self.window_samples:
                chunk = F.pad(chunk, (0, self.window_samples - chunk.size(1)))
                
            chunks.append(chunk)
            start += self.hop_samples
            
        return chunks
    
    def get_chunk_info(self, audio_1s: torch.Tensor) -> dict:
        """
        Get information about chunking without actually chunking.
        
        Returns:
            Dict with 'n_chunks', 'window_samples', 'hop_samples', 'total_coverage'
        """
        T = audio_1s.size(1)
        if T < self.min_samples:
            T = self.min_samples
            
        # Use exact formula: W = floor((segment_len - win_len)/hop) + 1
        n_chunks = max(1, int((T - self.window_samples) / self.hop_samples) + 1)
        total_coverage = (n_chunks - 1) * self.hop_samples + self.window_samples
        
        return {
            'n_chunks': n_chunks,
            'window_samples': self.window_samples,
            'hop_samples': self.hop_samples,
            'total_coverage': total_coverage,
            'overlap_ratio': self.overlap_ratio
        }


class DeterministicMapper:
    """
    Deterministic time-frequency mapper for coordinate-independent embedding.
    Maps window indices to (freq_bin, time_frame) slots using anchor-derived seeds.
    """
    
    def __init__(self, n_fft: int = 882, hop: int = 441, 
                 sr: int = 44100, window_ms: int = 15):
        """
        Args:
            n_fft: FFT size (must match INN model)
            hop: Hop length (must match INN model)
            sr: Sample rate (must match INN model)
            window_ms: Micro-chunk duration in ms
        """
        self.n_fft = n_fft
        self.hop = hop
        self.sr = sr
        self.window_ms = window_ms
        
        # Calculate frequency and time dimensions to match INN STFT exactly
        self.n_freq_bins = n_fft // 2 + 1  # 442 for n_fft=882
        self.window_frames = int(window_ms * sr / (hop * 1000))  # ~1.5 frames for 15ms @ 44.1kHz
        self.window_frames = max(1, self.window_frames)  # At least 1 frame
        
        # No global random state - use anchor-derived seeds per call
        
    def map_window_to_slots(self, window_idx: int, n_windows: int, 
                           target_bits: int, audio_content: torch.Tensor = None,
                           anchor_seed: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Map window index to deterministic (freq_bin, time_frame) slots.
        
        Args:
            window_idx: Index of current window (0 to n_windows-1)
            n_windows: Total number of windows in 1s segment
            target_bits: Target number of bits for this window
            audio_content: Optional audio content for content-aware mapping
            anchor_seed: Anchor-derived seed from audio content (required for determinism)
            
        Returns:
            List of (freq_bin, time_frame) tuples
        """
        if anchor_seed is None:
            raise ValueError("DeterministicMapper requires non-None anchor_seed in production paths")
        
        # Create deterministic seed based on window position and anchor
        window_seed = anchor_seed + window_idx * 1000 + n_windows * 100
        
        # Use content-aware mapping if audio provided
        if audio_content is not None:
            return self._content_aware_mapping(window_idx, n_windows, target_bits, 
                                             audio_content, window_seed)
        else:
            return self._deterministic_mapping(window_idx, n_windows, target_bits, window_seed)
    
    def _deterministic_mapping(self, window_idx: int, n_windows: int, 
                              target_bits: int, seed: int,
                              freq_ratio: float = 0.7) -> List[Tuple[int, int]]:
        """Pure deterministic mapping based on window position."""
        rng = np.random.RandomState(seed)
        
        # Distribute bits across frequency and time (configurable ratio)
        freq_bits = int(round(target_bits * freq_ratio))
        time_bits = max(0, target_bits - freq_bits)
        
        slots = []
        
        # Frequency slots (distributed across spectrum)
        if freq_bits > 0:
            freq_indices = rng.choice(self.n_freq_bins, size=min(freq_bits, self.n_freq_bins), 
                                    replace=False)
            for f in freq_indices:
                # Use middle frame of window
                t = 0 if self.window_frames == 1 else rng.choice(self.window_frames, size=1, replace=False)[0]
                slots.append((int(f), int(t)))
        
        # Time slots (distributed across window frames)
        if time_bits > 0:
            n_time_choices = max(2, self.window_frames)
            time_indices = rng.choice(n_time_choices, size=min(time_bits, n_time_choices), replace=False)
            for t in time_indices:
                # Select frequency from a small neighborhood around deterministic anchors
                band_center = self.n_freq_bins // 2
                spread = max(1, self.n_freq_bins // 16)
                f_candidates = np.arange(max(0, band_center - spread), min(self.n_freq_bins, band_center + spread))
                f = int(rng.choice(f_candidates)) if f_candidates.size > 0 else int(band_center)
                slots.append((int(f), int(t)))
        
        return slots[:target_bits]  # Ensure we don't exceed target
    
    def _content_aware_mapping(self, window_idx: int, n_windows: int, target_bits: int,
                              audio_content: torch.Tensor, seed: int,
                              freq_ratio: float = 0.7, thr_step: float = 0.10) -> List[Tuple[int, int]]:
        """Content-aware mapping using quantized audio characteristics."""
        rng = np.random.RandomState(seed)
        
        # Compute quantized audio features for deterministic mapping
        slots = []
        
        # Compute STFT with exact INN parameters
        X = self._compute_stft(audio_content)  # [1, 2, F, T]
        mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0]  # [F, T]
        
        # Quantize magnitude for deterministic ranking using shared bucketing intent
        mag_quantized = (torch.floor((mag / thr_step)) * thr_step * 1e6).long()
        
        # Compute per-band quantized thresholds
        band_energies = torch.median(mag_quantized, dim=1)[0]  # [F] - use median for robustness
        
        # Find top 4 frequency bands by quantized energy
        top_4_bands = torch.topk(band_energies, min(4, len(band_energies)))[1].cpu().numpy()
        
        # Create deterministic seed from quantized features
        feature_seed = self._compute_feature_seed(top_4_bands, mag_quantized.shape[1])
        local_rng = np.random.RandomState(seed + feature_seed)
        
        # Distribute bits across frequency and time deterministically
        freq_bits = int(round(target_bits * freq_ratio))
        time_bits = max(0, target_bits - freq_bits)
        
        # Frequency slots (prefer top energy bands)
        if freq_bits > 0:
            # Weight selection toward top energy bands
            weights = np.zeros(self.n_freq_bins)
            for i, band in enumerate(top_4_bands):
                if band < self.n_freq_bins:
                    weights[band] = 4 - i  # Higher weight for higher energy bands
            
            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(self.n_freq_bins) / self.n_freq_bins
            
            # Select frequency bins deterministically
            freq_indices = local_rng.choice(
                self.n_freq_bins, 
                size=min(freq_bits, self.n_freq_bins), 
                replace=False,
                p=weights
            )
            
            for f in freq_indices:
                # Spread time frames deterministically
                t = 0 if self.window_frames == 1 else local_rng.choice(self.window_frames, size=1, replace=False)[0]
                slots.append((int(f), int(t)))
        
        # Time slots (distributed across window frames)
        if time_bits > 0:
            n_time_choices = max(2, self.window_frames)
            time_indices = local_rng.choice(n_time_choices, size=min(time_bits, n_time_choices), replace=False)
            for t in time_indices:
                # Choose frequency near anchor bands deterministically
                band_center = self.n_freq_bins // 2
                spread = max(1, self.n_freq_bins // 16)
                f_candidates = np.arange(max(0, band_center - spread), min(self.n_freq_bins, band_center + spread))
                f = int(local_rng.choice(f_candidates)) if f_candidates.size > 0 else int(band_center)
                slots.append((int(f), int(t)))
        
        return slots[:target_bits]  # Ensure we don't exceed target
    
    def _compute_stft(self, audio_content: torch.Tensor) -> torch.Tensor:
        """Compute STFT with exact INN model parameters."""
        # Ensure correct shape [1, 1, T]
        if audio_content.dim() == 1:
            audio_content = audio_content.unsqueeze(0).unsqueeze(0)
        elif audio_content.dim() == 2:
            audio_content = audio_content.unsqueeze(0)
        
        # Use exact INN STFT parameters
        win_length = self.n_fft
        hop_length = self.hop
        
        # Build Hann window
        window = torch.hann_window(win_length, device=audio_content.device, dtype=audio_content.dtype)
        
        # Compute STFT with same parameters as INN
        X = torch.stft(
            audio_content.squeeze(1),
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # [1, F, T_frames]
        
        return torch.stack([X.real, X.imag], dim=1)  # [1, 2, F, T]
    
    def _compute_feature_seed(self, top_4_bands: np.ndarray, n_frames: int) -> int:
        """Compute deterministic seed from quantized audio features."""
        # Create deterministic hash from top bands and frame count
        data = f"{tuple(sorted(top_4_bands))}_{n_frames}".encode()
        return int(hashlib.sha256(data).hexdigest()[:8], 16)
    
    def get_mapping_info(self, n_windows: int, target_bits_per_window: int) -> dict:
        """Get information about the mapping without generating slots."""
        return {
            'n_freq_bins': self.n_freq_bins,
            'window_frames': self.window_frames,
            'total_slots_per_window': target_bits_per_window,
            'freq_slots_per_window': int(target_bits_per_window * 0.7),
            'time_slots_per_window': int(target_bits_per_window * 0.3)
        }


# EnhancedDeterministicMapper removed - using single redundancy approach with heavier RS codes


def _adaptive_stft(audio_window: torch.Tensor, overlap_ratio: float = 0.5, 
                   n_fft: int = 882, hop: int = 441) -> torch.Tensor:
    """
    Compute STFT with parameters that match the INN model exactly.
    
    Args:
        audio_window: Audio window [1, T] or [B, 1, T]
        overlap_ratio: Overlap ratio for hop length calculation (unused, kept for compatibility)
        n_fft: FFT size (must match INN model)
        hop: Hop length (must match INN model)
        
    Returns:
        STFT coefficients [B, 2, F, T] (real and imaginary channels)
    """
    # Normalize to [B, 1, T] format
    if audio_window.dim() == 1:  # [T]
        audio_window = audio_window.unsqueeze(0).unsqueeze(0)
    elif audio_window.dim() == 2:  # [B, T] or [1, T]
        if audio_window.size(0) == 1:  # [1, T]
            audio_window = audio_window.unsqueeze(0)  # [1, 1, T]
        else:  # [B, T]
            audio_window = audio_window.unsqueeze(1)  # [B, 1, T]
    elif audio_window.dim() == 3:  # [B, 1, T]
        pass
    else:
        raise ValueError(f"Unexpected audio_window ndim {audio_window.dim()}, expected 1/2/3")
    
    # Downmix to mono if needed
    if audio_window.size(1) > 1:
        audio_window = audio_window.mean(dim=1, keepdim=True)
    
    # x_wave: [B, 1, T]
    B, _, T = audio_window.shape
    
    # Use EXACT same parameters as INN model
    win_length = n_fft  # Use fixed window length to match INN
    hop_length = hop    # Use fixed hop length to match INN
    
    # Use same centering and padding rules as INN
    center = True
    pad_mode = "reflect"
    if (n_fft // 2) >= win_length:
        center = False
        pad_mode = "constant"
    
    # Build per-call Hann window on the correct device/dtype
    window = torch.hann_window(win_length, device=audio_window.device, dtype=audio_window.dtype)
    
    X = torch.stft(
        audio_window.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        return_complex=True,
    )  # [B, F, T_frames]
    
    return torch.stack([X.real, X.imag], dim=1)  # [B, 2, F, T]


def apply_psychoacoustic_gate(audio_window: torch.Tensor, slots: List[Tuple[int, int]], 
                            model, n_fft: int = 882, hop: int = 441) -> List[Tuple[int, int]]:
    """
    Apply psychoacoustic masking to filter slots based on audio content.
    Uses STFT parameters that match the INN model exactly.
    
    Args:
        audio_window: Audio window [1, T]
        slots: List of (freq_bin, time_frame) candidate slots
        model: INNWatermarker model (unused, kept for compatibility)
        n_fft: FFT size (must match INN model)
        hop: Hop length (must match INN model)
        
    Returns:
        Filtered list of slots that pass psychoacoustic masking
    """
    if not slots:
        return slots
        
    # Use STFT with exact INN parameters
    X = _adaptive_stft(audio_window, overlap_ratio=0.5, n_fft=n_fft, hop=hop)  # [1, 2, F, T]
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))  # [1, F, T]
    
    # Improved psychoacoustic gating with multiple criteria
    slot_scores = []
    for f, t in slots:
        if f < mag.size(1) and t < mag.size(2):
            # Get magnitude at this slot
            slot_mag = mag[0, f, t].item()
            
            # Get local context (3x3 neighborhood)
            f_start = max(0, f-1)
            f_end = min(mag.size(1), f+2)
            t_start = max(0, t-1)
            t_end = min(mag.size(2), t+2)
            
            local_mag = mag[0, f_start:f_end, t_start:t_end]
            local_mean = local_mag.mean().item()
            local_std = local_mag.std().item()
            
            # Score based on:
            # 1. Absolute magnitude (higher is better)
            # 2. Relative magnitude vs local mean (higher is better)
            # 3. Local variance (some variance is good for masking)
            relative_mag = slot_mag / (local_mean + 1e-12)
            variance_score = min(local_std / (local_mean + 1e-12), 2.0)  # Cap at 2.0
            
            # Combined score (weighted)
            score = (slot_mag * 0.4 + relative_mag * 0.4 + variance_score * 0.2)
            slot_scores.append((score, f, t))
        else:
            slot_scores.append((0.0, f, t))
    
    # Sort by score (descending) and take top-K deterministically
    slot_scores.sort(reverse=True)
    min_slots = max(1, len(slots) // 2)
    K = min_slots  # could be parameterized or tied to target bits per window
    filtered_slots = [(f, t) for _, f, t in slot_scores[:K]]
    
    return filtered_slots


def apply_psychoacoustic_gate_batched(audio_windows: torch.Tensor, all_slots: List[List[Tuple[int, int]]],
                                     overlap_ratio: float = 0.5, n_fft: int = 882, hop: int = 441) -> List[List[Tuple[int, int]]]:
    """Batched psychoacoustic gate for multiple windows at once.
    Args:
        audio_windows: [N, 1, T]
        all_slots: list of slot lists per window
        overlap_ratio: Overlap ratio (unused, kept for compatibility)
        n_fft: FFT size (must match INN model)
        hop: Hop length (must match INN model)
    Returns: filtered slot lists per window
    """
    if audio_windows.dim() == 2:
        audio_windows = audio_windows.unsqueeze(1)
    if audio_windows.numel() == 0:
        return all_slots
    # Compute batched STFT once with exact INN parameters
    X = _adaptive_stft(audio_windows, overlap_ratio=overlap_ratio, n_fft=n_fft, hop=hop)  # [N, 2, F, T]
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))  # [N, F, T]
    out: List[List[Tuple[int, int]]] = []
    N = audio_windows.size(0)
    for n in range(N):
        slots = all_slots[n]
        if not slots:
            out.append([])
            continue
        Fdim, Tdim = mag.size(1), mag.size(2)
        scored: List[Tuple[float, int, int]] = []
        for (f, t) in slots:
            if f < Fdim and t < Tdim:
                # 3x3 local region stats
                f0, f1 = max(0, f-1), min(Fdim, f+2)
                t0, t1 = max(0, t-1), min(Tdim, t+2)
                local = mag[n, f0:f1, t0:t1]
                slot_mag = mag[n, f, t]
                local_mean = local.mean()
                local_std = local.std()
                rel = slot_mag / (local_mean + 1e-12)
                var_score = torch.clamp(local_std / (local_mean + 1e-12), max=2.0)
                score = (slot_mag * 0.4 + rel * 0.4 + var_score * 0.2).item()
                scored.append((score, f, t))
        scored.sort(reverse=True)
        keep = max(1, len(slots) // 2)
        out.append([(f, t) for _, f, t in scored[:keep]])
    return out

def fuse_overlapping_windows(window_bits_list: List[torch.Tensor], 
                           overlap_ratio: float = 0.5) -> torch.Tensor:
    """
    Fuse bits from overlapping windows using majority voting.
    
    Args:
        window_bits_list: List of bit tensors from each window
        overlap_ratio: Overlap ratio for weighting
        
    Returns:
        Fused bit tensor
    """
    if not window_bits_list:
        return torch.tensor([], dtype=torch.long)
    
    if len(window_bits_list) == 1:
        return window_bits_list[0]
    
    # Stack all window bits
    all_bits = torch.stack(window_bits_list, dim=0)  # [n_windows, n_bits]
    
    # Simple majority voting
    # For overlapping regions, use weighted average based on overlap
    n_windows, n_bits = all_bits.shape
    
    # Calculate overlap weights
    weights = torch.ones(n_windows, n_bits)
    if overlap_ratio > 0:
        # Reduce weight for overlapping regions
        for i in range(1, n_windows):
            overlap_start = int(i * (1 - overlap_ratio) * n_bits)
            weights[i, :overlap_start] *= 0.5  # Reduce weight for overlapped bits
    
    # Weighted majority voting
    weighted_sum = (all_bits.float() * weights).sum(dim=0)
    total_weight = weights.sum(dim=0)
    
    # Avoid division by zero
    total_weight = torch.clamp(total_weight, min=1e-6)
    fused_probs = weighted_sum / total_weight
    
    # Convert back to bits
    fused_bits = (fused_probs > 0.5).long()
    
    return fused_bits.unsqueeze(0)  # [1, n_bits]


# Symbol fusion functions removed - using single redundancy approach with heavier RS codes
