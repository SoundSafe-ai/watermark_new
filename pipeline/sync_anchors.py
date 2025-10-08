# soundsafe/pipeline/sync_anchors.py
# Per-second sync anchor mechanism for coordinate-independent decoding
# Provides alignment without requiring stored slot coordinates

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from models.inn_encoder_decoder import INNWatermarker


class SyncAnchor:
    """
    Per-second sync anchor system for robust alignment.
    Embeds distinctive patterns at the start of each 1s segment.
    """
    
    def __init__(self, sync_length_ms: int = 50, sr: int = 22050, 
                 n_fft: int = 1024, hop: int = 512, pattern_type: str = "chirp"):
        """
        Args:
            sync_length_ms: Length of sync pattern in milliseconds
            sr: Sample rate
            n_fft: FFT size
            hop: Hop length
            pattern_type: Type of sync pattern ("chirp", "tone", "noise")
        """
        self.sync_length_ms = sync_length_ms
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.pattern_type = pattern_type
        
        # Calculate sync pattern length in samples
        self.sync_samples = int(sync_length_ms * sr / 1000)  # ~1102 samples for 50ms @ 22.05kHz
        self.sync_frames = int(sync_length_ms * sr / (hop * 1000))  # ~2.4 frames
        
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
    
    def get_sync_info(self) -> dict:
        """Get information about sync pattern."""
        return {
            'sync_length_ms': self.sync_length_ms,
            'sync_samples': self.sync_samples,
            'sync_frames': self.sync_frames,
            'pattern_type': self.pattern_type
        }


class SyncDetector:
    """
    Advanced sync pattern detector with multiple detection strategies.
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
                     threshold: float = 0.05) -> Tuple[int, float]:
        """
        Robust detection using multiple strategies.
        
        Args:
            audio_1s: Audio tensor [1, T]
            threshold: Detection threshold
            
        Returns:
            (offset_samples, confidence) tuple
        """
        # Try time-domain detection first
        offset_td, conf_td = self.sync_anchor.detect_sync_pattern(audio_1s, threshold)
        
        # Try frequency-domain detection
        offset_fd, conf_fd = self.detect_with_stft(audio_1s, threshold)
        
        # Choose best result
        if conf_td > conf_fd:
            return offset_td, conf_td
        else:
            return offset_fd, conf_fd
