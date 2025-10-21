# soundsafe/pipeline/sync_anchors.py
# Per-second sync anchor mechanism for coordinate-independent decoding
# Provides alignment without requiring stored slot coordinates

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import hashlib
from typing import Tuple, Optional
from models.inn_encoder_decoder import INNWatermarker
from pipeline.moore_galsberg import MooreGlasbergAnalyzer


class SyncAnchor:
    """
    Per-second sync anchor system for robust alignment.
    Embeds distinctive patterns at the start of each 1s segment.
    """
    
    def __init__(self, sync_length_ms: int = 50, sr: int = 44100, 
                 n_fft: int = 882, hop: int = 441, pattern_type: str = "chirp",
                 n_anchor_bands: int = 4, ranking_method: str = "median",
                 hashing_salt: str = "soundsafe_anchor"):
        """
        Args:
            sync_length_ms: Length of sync pattern in milliseconds
            sr: Sample rate (must match INN model)
            n_fft: FFT size (must match INN model)
            hop: Hop length (must match INN model)
            pattern_type: Type of sync pattern ("chirp", "tone", "noise")
            n_anchor_bands: Number of anchor bands to use for seeding
            ranking_method: Method for ranking bands ("mean", "median")
            hashing_salt: Salt for deterministic hashing
        """
        self.sync_length_ms = sync_length_ms
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.pattern_type = pattern_type
        self.n_anchor_bands = n_anchor_bands
        self.ranking_method = ranking_method
        self.hashing_salt = hashing_salt
        
        # Calculate sync pattern length in samples
        self.sync_samples = int(sync_length_ms * sr / 1000)  # ~2205 samples for 50ms @ 44.1kHz
        self.sync_frames = int(sync_length_ms * sr / (hop * 1000))  # ~2.5 frames
        
        # Initialize Moore-Glasberg analyzer for anchor band detection
        self.mg_analyzer = MooreGlasbergAnalyzer(
            sample_rate=sr, 
            n_fft=n_fft, 
            hop_length=hop, 
            n_critical_bands=24
        )
        
        # Generate sync pattern
        self.sync_pattern = self._generate_sync_pattern()
        
    def _generate_sync_pattern(self) -> torch.Tensor:
        """Generate distinctive sync pattern."""
        if self.pattern_type == "chirp":
            return self._generate_chirp_pattern()
        elif self.pattern_type == "tone":
            return self._generate_tone_pattern()
        elif self.pattern_type == "noise":
            return self._generate_noise_pattern()
        else:
            raise ValueError(f"Unknown pattern_type: {self.pattern_type}")
    
    def _generate_chirp_pattern(self) -> torch.Tensor:
        """Generate linear chirp pattern (frequency sweep)."""
        t = torch.linspace(0, self.sync_length_ms / 1000, self.sync_samples)
        
        # Linear chirp from 1000 Hz to 2000 Hz
        f0, f1 = 1000, 2000
        freq = f0 + (f1 - f0) * t / t[-1]
        phase = 2 * np.pi * torch.cumsum(freq, dim=0) / self.sr
        
        # Add envelope to avoid clicks
        envelope = torch.exp(-t * 10)  # Exponential decay
        chirp = 0.3 * torch.sin(phase) * envelope
        
        return chirp.unsqueeze(0)  # [1, sync_samples]
    
    def _generate_tone_pattern(self) -> torch.Tensor:
        """Generate pure tone pattern."""
        t = torch.linspace(0, self.sync_length_ms / 1000, self.sync_samples)
        freq = 1500  # 1.5 kHz tone
        tone = 0.3 * torch.sin(2 * np.pi * freq * t)
        
        # Add envelope
        envelope = torch.exp(-t * 5)
        tone = tone * envelope
        
        return tone.unsqueeze(0)  # [1, sync_samples]
    
    def _generate_noise_pattern(self) -> torch.Tensor:
        """Generate filtered noise pattern."""
        # Generate white noise
        noise = torch.randn(self.sync_samples) * 0.1
        
        # Apply bandpass filter (1000-2000 Hz)
        # Simple IIR filter approximation
        b = torch.tensor([0.1, 0, -0.1])  # High-pass component
        a = torch.tensor([1.0, -1.8, 0.8])  # Low-pass component
        
        # Apply filter (simplified)
        filtered_noise = noise
        for i in range(2, len(noise)):
            filtered_noise[i] = (b[0] * noise[i] + b[1] * noise[i-1] + b[2] * noise[i-2] -
                               a[1] * filtered_noise[i-1] - a[2] * filtered_noise[i-2])
        
        # Add envelope
        t = torch.linspace(0, self.sync_length_ms / 1000, self.sync_samples)
        envelope = torch.exp(-t * 8)
        filtered_noise = filtered_noise * envelope
        
        return filtered_noise.unsqueeze(0)  # [1, sync_samples]
    
    def embed_sync_pattern(self, audio_1s: torch.Tensor, 
                          sync_strength: float = 0.1) -> torch.Tensor:
        """
        Embed sync pattern at the start of 1s audio segment.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 22050 samples
            sync_strength: Strength of sync pattern (0.0 to 1.0)
            
        Returns:
            Audio with embedded sync pattern
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        T = audio_1s.size(1)
        if T < self.sync_samples:
            # Pad if too short
            audio_1s = F.pad(audio_1s, (0, self.sync_samples - T))
            T = self.sync_samples
        
        # Create copy and embed sync pattern
        audio_with_sync = audio_1s.clone()
        
        # Add sync pattern to beginning
        sync_region = audio_1s[:, :self.sync_samples]
        # Move sync pattern to same device as audio
        sync_pattern_scaled = (self.sync_pattern * sync_strength).to(audio_1s.device)
        
        # Mix with original audio (not replace)
        audio_with_sync[:, :self.sync_samples] = sync_region + sync_pattern_scaled
        
        # Clamp to prevent clipping
        audio_with_sync = torch.clamp(audio_with_sync, -1.0, 1.0)
        
        return audio_with_sync
    
    def detect_sync_pattern(self, audio_1s: torch.Tensor, 
                           threshold: float = 0.05) -> Tuple[int, float]:
        """
        Detect sync pattern in audio and return alignment offset.
        
        Args:
            audio_1s: Audio tensor [1, T]
            threshold: Detection threshold for correlation
            
        Returns:
            (offset_samples, confidence) tuple
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        T = audio_1s.size(1)
        if T < self.sync_samples:
            return 0, 0.0
        
        # Cross-correlation between audio and sync pattern
        audio_flat = audio_1s.squeeze(0)  # [T]
        # Move sync pattern to same device as audio
        sync_flat = self.sync_pattern.squeeze(0).to(audio_1s.device)  # [sync_samples]
        
        # Normalize both signals
        audio_norm = (audio_flat - audio_flat.mean()) / (audio_flat.std() + 1e-8)
        sync_norm = (sync_flat - sync_flat.mean()) / (sync_flat.std() + 1e-8)
        
        # Cross-correlation
        correlation = F.conv1d(
            audio_norm.unsqueeze(0).unsqueeze(0),  # [1, 1, T]
            sync_norm.flip(0).unsqueeze(0).unsqueeze(0),  # [1, 1, sync_samples]
            padding=self.sync_samples - 1
        ).squeeze()  # [T + sync_samples - 1]
        
        # Find peak
        max_corr, max_idx = correlation.max(dim=0)
        confidence = max_corr.item()
        
        # Calculate offset (convert to samples)
        offset = max_idx.item() - (self.sync_samples - 1)
        
        # Check if confidence meets threshold
        if confidence < threshold:
            return 0, confidence
        
        return int(offset), confidence
    
    def align_audio(self, audio_1s: torch.Tensor, offset: int) -> torch.Tensor:
        """
        Align audio based on detected sync pattern offset.
        
        Args:
            audio_1s: Audio tensor [1, T]
            offset: Offset in samples (can be negative)
            
        Returns:
            Aligned audio tensor
        """
        if offset == 0:
            return audio_1s
        
        T = audio_1s.size(1)
        
        if offset > 0:
            # Shift right (remove samples from beginning)
            if offset >= T:
                # Offset too large, return zeros
                return torch.zeros_like(audio_1s)
            return audio_1s[:, offset:]
        else:
            # Shift left (pad beginning with zeros)
            pad_samples = -offset
            return F.pad(audio_1s, (pad_samples, 0))[:, :T]
    
    def detect_anchor_and_seed(self, audio_1s: torch.Tensor, 
                              confidence_threshold: float = 0.3) -> Tuple[Optional[int], float]:
        """
        Detect anchor bands and compute deterministic seed.
        
        Args:
            audio_1s: Audio tensor [1, T] where T ≈ 44100 samples
            confidence_threshold: Minimum confidence for anchor-based seeding
            
        Returns:
            (seed, confidence) tuple. Returns (None, 0.0) if confidence too low.
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        try:
            # Compute STFT with exact INN parameters
            X = self._compute_stft(audio_1s)  # [1, 2, F, T]
            mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))[0]  # [F, T]
            
            # Compute quantized band thresholds using Moore-Glasberg
            band_thr_bt = self.mg_analyzer.band_thresholds(mag.cpu().numpy())  # [BANDS, T]
            
            # Quantize thresholds for robustness
            quantized_thr_bt = np.floor(band_thr_bt / 1e-6) * 1e-6  # Same quantization as allocator
            
            # Compute band energies using specified ranking method
            if self.ranking_method == "median":
                band_energies = np.median(quantized_thr_bt, axis=1)  # [BANDS]
            else:  # mean
                band_energies = np.mean(quantized_thr_bt, axis=1)  # [BANDS]
            
            # Find top N anchor bands by energy
            top_bands = np.argsort(band_energies)[-self.n_anchor_bands:]
            
            # Sort by center frequency to create canonical ordering
            band_center_freqs = self.mg_analyzer.band_indices  # [F] -> band_id
            # Get center frequencies for top bands
            top_band_freqs = []
            for band in top_bands:
                # Find center frequency bin for this band
                band_bins = np.where(band_center_freqs == band)[0]
                if len(band_bins) > 0:
                    center_freq = np.mean(band_bins)
                    top_band_freqs.append((center_freq, band))
            
            # Sort by center frequency
            top_band_freqs.sort(key=lambda x: x[0])
            canonical_bands = tuple(band for _, band in top_band_freqs)
            
            # Compute confidence based on energy separation
            if len(band_energies) > 0:
                sorted_energies = np.sort(band_energies)
                if len(sorted_energies) >= 2:
                    # Confidence based on separation between top bands and others
                    top_energy = np.mean(sorted_energies[-self.n_anchor_bands:])
                    other_energy = np.mean(sorted_energies[:-self.n_anchor_bands])
                    confidence = min(1.0, (top_energy - other_energy) / (top_energy + 1e-12))
                else:
                    confidence = 0.5
            else:
                confidence = 0.0
            
            if confidence < confidence_threshold:
                return None, confidence
            
            # Create deterministic seed from anchor bands and segment length
            segment_length = audio_1s.size(1)
            seed_data = f"{canonical_bands}_{segment_length}_{self.hashing_salt}".encode()
            seed = int(hashlib.sha256(seed_data).hexdigest()[:8], 16)
            
            return seed, confidence
            
        except Exception as e:
            # If anchor detection fails, return low confidence
            return None, 0.0
    
    def _compute_stft(self, audio_1s: torch.Tensor) -> torch.Tensor:
        """Compute STFT with exact INN model parameters."""
        # Ensure correct shape [1, 1, T]
        if audio_1s.dim() == 2:
            audio_1s = audio_1s.unsqueeze(0)
        
        # Use exact INN STFT parameters
        win_length = self.n_fft
        hop_length = self.hop
        
        # Build Hann window
        window = torch.hann_window(win_length, device=audio_1s.device, dtype=audio_1s.dtype)
        
        # Compute STFT with same parameters as INN
        X = torch.stft(
            audio_1s.squeeze(1),
            n_fft=self.n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # [1, F, T_frames]
        
        return torch.stack([X.real, X.imag], dim=1)  # [1, 2, F, T]
    
    def get_sync_info(self) -> dict:
        """Get information about sync pattern and anchor configuration."""
        return {
            'sync_length_ms': self.sync_length_ms,
            'sync_samples': self.sync_samples,
            'sync_frames': self.sync_frames,
            'pattern_type': self.pattern_type,
            'n_anchor_bands': self.n_anchor_bands,
            'ranking_method': self.ranking_method,
            'hashing_salt': self.hashing_salt
        }


class SyncDetector:
    """
    Advanced sync pattern detector with anchor-based seeding and fallback strategies.
    """
    
    def __init__(self, sync_anchor: SyncAnchor, model: INNWatermarker):
        """
        Args:
            sync_anchor: SyncAnchor instance
            model: INNWatermarker model for STFT analysis
        """
        self.sync_anchor = sync_anchor
        self.model = model
        
    def detect_with_stft(self, audio_1s: torch.Tensor, 
                        threshold: float = 0.05) -> Tuple[int, float]:
        """
        Detect sync pattern using STFT analysis for better robustness.
        
        Args:
            audio_1s: Audio tensor [1, T]
            threshold: Detection threshold
            
        Returns:
            (offset_samples, confidence) tuple
        """
        if audio_1s.dim() != 2 or audio_1s.size(0) != 1:
            raise ValueError(f"Expected audio_1s shape [1, T], got {audio_1s.shape}")
        
        T = audio_1s.size(1)
        if T < self.sync_anchor.sync_samples:
            return 0, 0.0
        
        # Get STFT of both audio and sync pattern
        X_audio = self.model.stft(audio_1s)  # [1, 2, F, T]
        X_sync = self.model.stft(self.sync_anchor.sync_pattern)  # [1, 2, F, T_sync]
        
        # Use magnitude for correlation
        mag_audio = torch.sqrt(torch.clamp(X_audio[:, 0]**2 + X_audio[:, 1]**2, min=1e-12))
        mag_sync = torch.sqrt(torch.clamp(X_sync[:, 0]**2 + X_sync[:, 1]**2, min=1e-12))
        
        # Cross-correlation in frequency domain
        correlation = F.conv2d(
            mag_audio.unsqueeze(0),  # [1, 1, F, T]
            mag_sync.flip(-1).unsqueeze(0),  # [1, 1, F, T_sync]
            padding=(0, mag_sync.size(-1) - 1)
        ).squeeze()  # [F, T + T_sync - 1]
        
        # Sum across frequency bins
        correlation_sum = correlation.sum(dim=0)  # [T + T_sync - 1]
        
        # Find peak
        max_corr, max_idx = correlation_sum.max(dim=0)
        confidence = max_corr.item() / (mag_audio.numel() + 1e-8)
        
        # Calculate offset
        offset = max_idx.item() - (mag_sync.size(-1) - 1)
        
        if confidence < threshold:
            return 0, confidence
        
        return int(offset), confidence
    
    def detect_robust(self, audio_1s: torch.Tensor, 
                     threshold: float = 0.05, anchor_threshold: float = 0.3) -> Tuple[int, float, Optional[int]]:
        """
        Robust detection using anchor-based seeding with fallback to sync patterns.
        
        Args:
            audio_1s: Audio tensor [1, T]
            threshold: Detection threshold for sync patterns
            anchor_threshold: Confidence threshold for anchor-based seeding
            
        Returns:
            (offset_samples, confidence, anchor_seed) tuple
        """
        # First try anchor-based seeding
        anchor_seed, anchor_confidence = self.sync_anchor.detect_anchor_and_seed(
            audio_1s, anchor_threshold
        )
        
        if anchor_seed is not None and anchor_confidence >= anchor_threshold:
            # Anchor-based seeding successful, no offset needed
            return 0, anchor_confidence, anchor_seed
        
        # Fallback to sync pattern detection for severe cases
        # Try time-domain detection first
        offset_td, conf_td = self.sync_anchor.detect_sync_pattern(audio_1s, threshold)
        
        # Try frequency-domain detection
        offset_fd, conf_fd = self.detect_with_stft(audio_1s, threshold)
        
        # Choose best sync pattern result
        if conf_td > conf_fd:
            return offset_td, conf_td, None
        else:
            return offset_fd, conf_fd, None
    
    def detect_anchor_and_seed(self, audio_1s: torch.Tensor, 
                              confidence_threshold: float = 0.3) -> Tuple[Optional[int], float]:
        """
        Detect anchor bands and compute deterministic seed.
        
        Args:
            audio_1s: Audio tensor [1, T]
            confidence_threshold: Minimum confidence for anchor-based seeding
            
        Returns:
            (seed, confidence) tuple
        """
        return self.sync_anchor.detect_anchor_and_seed(audio_1s, confidence_threshold)
