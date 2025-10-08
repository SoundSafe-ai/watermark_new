# soundsafe/pipeline/micro_chunking.py
# Micro-chunking system with overlap for robust watermarking
# Implements 10-20ms windows with 50% overlap per 1-second segment

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np
from pipeline.error_correction import fuse_symbol_evidence_with_crc, compute_crc_agreement_rate


class MicroChunker:
    """
    Splits 1-second audio segments into overlapping micro-chunks.
    Each micro-chunk is 10-20ms with 50% overlap for redundancy.
    """
    
    def __init__(self, window_ms: int = 15, overlap_ratio: float = 0.5, sr: int = 22050):
        """
        Args:
            window_ms: Micro-chunk duration in milliseconds (10-20ms recommended)
            overlap_ratio: Overlap between consecutive chunks (0.5 = 50% overlap)
            sr: Sample rate
        """
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.sr = sr
        
        # Calculate samples
        self.window_samples = int(window_ms * sr / 1000)  # ~330 samples for 15ms @ 22.05kHz
        self.hop_samples = int(self.window_samples * (1 - overlap_ratio))  # ~165 samples for 50% overlap
        
        # Ensure we have enough samples for at least one window
        self.min_samples = self.window_samples
        
    def chunk_1s_segment(self, audio_1s: torch.Tensor) -> List[torch.Tensor]:
        """
        Split 1-second audio into overlapping micro-chunks.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 22050 samples
            
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
            
        n_chunks = max(1, (T - self.window_samples) // self.hop_samples + 1)
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
    Maps window indices to (freq_bin, time_frame) slots using seeded randomness.
    """
    
    def __init__(self, seed: int = 42, n_fft: int = 1024, hop: int = 512, 
                 sr: int = 22050, window_ms: int = 15):
        """
        Args:
            seed: Random seed for deterministic mapping
            n_fft: FFT size
            hop: Hop length
            sr: Sample rate
            window_ms: Micro-chunk duration in ms
        """
        self.seed = seed
        self.n_fft = n_fft
        self.hop = hop
        self.sr = sr
        self.window_ms = window_ms
        
        # Calculate frequency and time dimensions
        self.n_freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024
        self.window_frames = int(window_ms * sr / (hop * 1000))  # ~1.3 frames for 15ms
        self.window_frames = max(1, self.window_frames)  # At least 1 frame
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
    def map_window_to_slots(self, window_idx: int, n_windows: int, 
                           target_bits: int, audio_content: torch.Tensor = None) -> List[Tuple[int, int]]:
        """
        Map window index to deterministic (freq_bin, time_frame) slots.
        
        Args:
            window_idx: Index of current window (0 to n_windows-1)
            n_windows: Total number of windows in 1s segment
            target_bits: Target number of bits for this window
            audio_content: Optional audio content for content-aware mapping
            
        Returns:
            List of (freq_bin, time_frame) tuples
        """
        # Create deterministic seed based on window position
        window_seed = self.seed + window_idx * 1000 + n_windows * 100
        
        # Use content-aware mapping if audio provided
        if audio_content is not None:
            return self._content_aware_mapping(window_idx, n_windows, target_bits, 
                                             audio_content, window_seed)
        else:
            return self._deterministic_mapping(window_idx, n_windows, target_bits, window_seed)
    
    def _deterministic_mapping(self, window_idx: int, n_windows: int, 
                              target_bits: int, seed: int) -> List[Tuple[int, int]]:
        """Pure deterministic mapping based on window position."""
        rng = np.random.RandomState(seed)
        
        # Distribute bits across frequency and time
        freq_bits = int(target_bits * 0.7)  # 70% in frequency domain
        time_bits = target_bits - freq_bits  # 30% in time domain
        
        slots = []
        
        # Frequency slots (distributed across spectrum)
        if freq_bits > 0:
            freq_indices = rng.choice(self.n_freq_bins, size=min(freq_bits, self.n_freq_bins), 
                                    replace=False)
            for f in freq_indices:
                # Use middle frame of window
                t = self.window_frames // 2
                slots.append((int(f), int(t)))
        
        # Time slots (distributed across window frames)
        if time_bits > 0:
            time_indices = rng.choice(self.window_frames, size=min(time_bits, self.window_frames), 
                                    replace=False)
            for t in time_indices:
                # Use mid-frequency
                f = self.n_freq_bins // 2
                slots.append((int(f), int(t)))
        
        return slots[:target_bits]  # Ensure we don't exceed target
    
    def _content_aware_mapping(self, window_idx: int, n_windows: int, target_bits: int,
                              audio_content: torch.Tensor, seed: int) -> List[Tuple[int, int]]:
        """Content-aware mapping using audio characteristics."""
        rng = np.random.RandomState(seed)
        
        # Simple content-aware: prefer higher energy regions
        # This is a placeholder - can be enhanced with more sophisticated analysis
        slots = []
        
        # For now, use deterministic mapping but could be enhanced
        # to analyze audio content and prefer perceptually masked regions
        return self._deterministic_mapping(window_idx, n_windows, target_bits, seed)
    
    def get_mapping_info(self, n_windows: int, target_bits_per_window: int) -> dict:
        """Get information about the mapping without generating slots."""
        return {
            'n_freq_bins': self.n_freq_bins,
            'window_frames': self.window_frames,
            'total_slots_per_window': target_bits_per_window,
            'freq_slots_per_window': int(target_bits_per_window * 0.7),
            'time_slots_per_window': int(target_bits_per_window * 0.3)
        }


class EnhancedDeterministicMapper:
    """
    Enhanced mapper that implements symbol-level redundancy and time-frequency scattering.
    Maps each symbol to r=2-4 different windows for robustness.
    """
    
    def __init__(self, seed: int = 42, n_fft: int = 1024, hop: int = 512, 
                 max_freq_bins: int = 513, max_time_frames: int = 44,
                 redundancy: int = 3, bias_lower_mid: bool = True):
        """
        Args:
            seed: Random seed for deterministic mapping
            n_fft: FFT size
            hop: Hop length
            max_freq_bins: Maximum frequency bins
            max_time_frames: Maximum time frames
            redundancy: Number of windows each symbol should be embedded in (r=2-4)
            bias_lower_mid: Whether to bias toward lower-mid frequency bands
        """
        self.rng = np.random.RandomState(seed)
        self.n_fft = n_fft
        self.hop = hop
        self.max_freq_bins = max_freq_bins
        self.max_time_frames = max_time_frames
        self.redundancy = redundancy
        self.bias_lower_mid = bias_lower_mid
        
        # Define lower-mid frequency range for robustness bias
        self.lower_mid_freq_start = int(0.1 * max_freq_bins)  # ~10% of spectrum
        self.lower_mid_freq_end = int(0.4 * max_freq_bins)    # ~40% of spectrum
    
    def map_symbol_to_windows(self, symbol_idx: int, n_windows: int, 
                             n_symbols: int) -> List[Tuple[int, int]]:
        """
        Map a symbol to r=2-4 different windows with time-frequency scattering.
        
        Args:
            symbol_idx: Index of the symbol (0 to n_symbols-1)
            n_windows: Total number of windows in the 1-second segment
            n_symbols: Total number of symbols to embed
            
        Returns:
            List of (window_idx, band_idx) tuples where this symbol should be embedded
        """
        # Ensure reproducibility for a given symbol_idx
        local_rng = np.random.RandomState(self.rng.randint(0, 2**32 - 1) + symbol_idx)
        
        # Select r=2-4 windows for this symbol
        redundancy = min(self.redundancy, n_windows)
        selected_windows = local_rng.choice(n_windows, size=redundancy, replace=False)
        
        # For each selected window, choose a frequency band
        symbol_placements = []
        for window_idx in selected_windows:
            # Choose frequency band with optional lower-mid bias
            if self.bias_lower_mid and local_rng.random() < 0.7:  # 70% chance for lower-mid
                band_idx = local_rng.randint(self.lower_mid_freq_start, self.lower_mid_freq_end)
            else:
                band_idx = local_rng.randint(0, self.max_freq_bins)
            
            symbol_placements.append((int(window_idx), int(band_idx)))
        
        return symbol_placements
    
    def build_symbol_mapping(self, n_windows: int, n_symbols: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Build complete symbol-to-windows mapping for all symbols.
        
        Args:
            n_windows: Total number of windows in the 1-second segment
            n_symbols: Total number of symbols to embed
            
        Returns:
            Dict mapping symbol_idx -> List of (window_idx, band_idx) tuples
        """
        symbol_mappings = {}
        for symbol_idx in range(n_symbols):
            symbol_mappings[symbol_idx] = self.map_symbol_to_windows(
                symbol_idx, n_windows, n_symbols
            )
        return symbol_mappings
    
    def get_window_symbols(self, window_idx: int, symbol_mappings: Dict[int, List[Tuple[int, int]]]) -> List[int]:
        """
        Get all symbols that should be embedded in a specific window.
        
        Args:
            window_idx: Index of the window
            symbol_mappings: Complete symbol mapping
            
        Returns:
            List of symbol indices that should be embedded in this window
        """
        window_symbols = []
        for symbol_idx, placements in symbol_mappings.items():
            for w_idx, _ in placements:
                if w_idx == window_idx:
                    window_symbols.append(symbol_idx)
        return window_symbols


def _adaptive_stft(audio_window: torch.Tensor) -> torch.Tensor:
    """
    Compute STFT with parameters adapted to the window length.
    
    Args:
        audio_window: Audio window [1, T] or [B, 1, T]
        
    Returns:
        STFT coefficients [1, 2, F, T] (real and imaginary channels)
    """
    # Get the time dimension (last dimension)
    T = audio_window.shape[-1]
    
    # Adaptive STFT parameters
    win_length = T
    # Next power of 2 >= win_length
    n_fft = 1 << (win_length - 1).bit_length()  # 330 -> 512
    
    # Ensure reflect padding is valid: n_fft//2 < T
    if (n_fft // 2) >= T:
        n_fft = max(256, n_fft // 2)  # fallback safeguard
    
    hop_length = max(1, win_length // 2)  # ~50% overlap
    
    # Create window on same device and dtype as input
    window = torch.hann_window(win_length, device=audio_window.device, dtype=audio_window.dtype)
    
    # Ensure input is 2D for STFT: [B, T]
    if audio_window.dim() == 3:
        audio_2d = audio_window.squeeze(1)  # [B, T]
    else:
        audio_2d = audio_window  # [1, T]
    
    # Compute STFT
    X = torch.stft(
        audio_2d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",  # Now valid since n_fft//2 < T
        return_complex=True,
    )  # [B, F, T]
    
    # Convert to real/imaginary channels: [B, 2, F, T]
    X_ri = torch.stack([X.real, X.imag], dim=1)
    
    return X_ri


def apply_psychoacoustic_gate(audio_window: torch.Tensor, slots: List[Tuple[int, int]], 
                            model, n_fft: int = 1024, hop: int = 512) -> List[Tuple[int, int]]:
    """
    Apply psychoacoustic masking to filter slots based on audio content.
    Uses adaptive STFT parameters that match the micro-window size.
    
    Args:
        audio_window: Audio window [1, T]
        slots: List of (freq_bin, time_frame) candidate slots
        model: INNWatermarker model (unused, kept for compatibility)
        n_fft: FFT size (unused, kept for compatibility)
        hop: Hop length (unused, kept for compatibility)
        
    Returns:
        Filtered list of slots that pass psychoacoustic masking
    """
    if not slots:
        return slots
        
    # Use adaptive STFT instead of model.stft
    X = _adaptive_stft(audio_window)  # [1, 2, F, T]
    mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))  # [1, F, T]
    
    # Simple psychoacoustic gating: prefer slots with higher magnitude
    # (indicating more masking headroom)
    slot_scores = []
    for f, t in slots:
        if f < mag.size(1) and t < mag.size(2):
            score = mag[0, f, t].item()
            slot_scores.append((score, f, t))
        else:
            slot_scores.append((0.0, f, t))
    
    # Sort by score and take top slots
    slot_scores.sort(reverse=True)
    filtered_slots = [(f, t) for _, f, t in slot_scores]
    
    return filtered_slots


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


def fuse_symbol_evidence(window_predictions: List[torch.Tensor], 
                        symbol_mappings: Dict[int, List[Tuple[int, int]]],
                        n_symbols: int) -> torch.Tensor:
    """
    Fuse evidence from multiple windows for each symbol.
    
    Args:
        window_predictions: List of [1, S_window] tensors for each window
        symbol_mappings: Dict mapping symbol_idx -> List of (window_idx, band_idx) tuples
        n_symbols: Total number of symbols
        
    Returns:
        [1, n_symbols] tensor with fused symbol predictions
    """
    device = window_predictions[0].device
    fused_symbols = torch.zeros(1, n_symbols, device=device)
    
    for symbol_idx, placements in symbol_mappings.items():
        if symbol_idx >= n_symbols:
            continue
            
        # Collect predictions from all windows containing this symbol
        symbol_evidence = []
        for window_idx, band_idx in placements:
            if window_idx < len(window_predictions):
                window_pred = window_predictions[window_idx]
                if band_idx < window_pred.size(1):
                    symbol_evidence.append(window_pred[0, band_idx].item())
        
        if symbol_evidence:
            # Simple fusion: average the evidence
            # In practice, you might want weighted average, majority voting, etc.
            fused_value = sum(symbol_evidence) / len(symbol_evidence)
            fused_symbols[0, symbol_idx] = fused_value
    
    return fused_symbols


def fuse_symbol_evidence_enhanced(window_predictions: List[torch.Tensor], 
                                 symbol_mappings: Dict[int, List[Tuple[int, int]]],
                                 n_symbols: int, symbol_length: int = 8,
                                 use_crc: bool = True, min_agreement_rate: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced fusion with CRC-based weighting and outlier rejection.
    
    Args:
        window_predictions: List of [1, S_window] tensors for each window
        symbol_mappings: Dict mapping symbol_idx -> List of (window_idx, band_idx) tuples
        n_symbols: Total number of symbols
        symbol_length: Length of each symbol in bits
        use_crc: Whether to use CRC validation
        min_agreement_rate: Minimum CRC agreement rate to accept a symbol
        
    Returns:
        (fused_symbols, confidence_scores) tuple
    """
    if use_crc:
        return fuse_symbol_evidence_with_crc(
            window_predictions, symbol_mappings, n_symbols, 
            symbol_length, min_agreement_rate
        )
    else:
        # Fallback to simple fusion
        fused_symbols = fuse_symbol_evidence(window_predictions, symbol_mappings, n_symbols)
        confidence_scores = torch.ones_like(fused_symbols)
        return fused_symbols, confidence_scores
