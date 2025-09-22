#!/usr/bin/env python3
"""
Comprehensive loss functions for SoundSafe watermarking training
All losses are GPU-optimized for maximum performance
Focus on imperceptibility training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Tuple
import math

class ImperceptibilityLoss(nn.Module):
    """
    Comprehensive imperceptibility loss combining multiple perceptual metrics
    All computations happen on GPU for maximum performance
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
        n_mfcc: int = 20,
        device: str = 'cuda'
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        
        # Pre-compute transforms on GPU
        self._setup_transforms(n_fft, hop_length, n_mels, n_mfcc)
        
        # Loss weights (tunable)
        self.weights = {
            'l1': 1.0,
            'l2': 0.5,
            'mrstft_mag': 2.0,
            'mrstft_sc': 1.0,
            'mfcc': 1.5,
            'snr': 0.3,
            'spectral_centroid': 0.5,
            'spectral_rolloff': 0.5,
            'zero_crossing_rate': 0.2
        }
        
    def _setup_transforms(self, n_fft: int, hop_length: int, n_mels: int, n_mfcc: int):
        """Setup GPU-optimized transforms"""
        # Multi-resolution STFT
        self.stft_configs = [
            {'n_fft': 512, 'hop_length': 128, 'win_length': 512},
            {'n_fft': 1024, 'hop_length': 256, 'win_length': 1024},
            {'n_fft': 2048, 'hop_length': 512, 'win_length': 2048}
        ]
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(self.device)
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        ).to(self.device)
        
        # DCT matrix for MFCC (pre-computed)
        self.dct_matrix = self._create_dct_matrix(n_mels, n_mfcc).to(self.device)
        
    def _create_dct_matrix(self, n_mels: int, n_mfcc: int) -> torch.Tensor:
        """Create DCT matrix for MFCC computation"""
        n = torch.arange(n_mels, device=self.device).float()
        k = torch.arange(n_mfcc, device=self.device).float().unsqueeze(1)
        dct = torch.cos((torch.pi / n_mels) * (n + 0.5) * k)
        dct[0] *= 1.0 / math.sqrt(2.0)
        return dct
    
    def _stft_magnitude(self, x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        """Compute STFT magnitude on GPU"""
        window = torch.hann_window(win_length, device=x.device)
        stft = torch.stft(
            x.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=window, return_complex=True
        )
        return torch.abs(stft)
    
    def _multi_res_stft_loss(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-resolution STFT loss"""
        x_ = x.squeeze(1)  # [B, T]
        y_ = y.squeeze(1)  # [B, T]
        
        mag_loss = 0.0
        sc_loss = 0.0
        
        for config in self.stft_configs:
            X = self._stft_magnitude(x_, **config)
            Y = self._stft_magnitude(y_, **config)
            
            # Magnitude loss
            mag_loss += F.l1_loss(X, Y)
            
            # Spectral convergence
            sc = torch.norm(X - Y, p='fro') / (torch.norm(X, p='fro') + 1e-9)
            sc_loss += sc
        
        # Average over resolutions
        mag_loss = mag_loss / len(self.stft_configs)
        sc_loss = sc_loss / len(self.stft_configs)
        
        return {
            'mrstft_mag': mag_loss,
            'mrstft_sc': sc_loss,
            'mrstft_total': mag_loss + sc_loss
        }
    
    def _mfcc_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """MFCC cosine similarity loss"""
        x_ = x.squeeze(1)  # [B, T]
        y_ = y.squeeze(1)  # [B, T]
        
        # Compute MFCCs
        X_mfcc = self.mfcc_transform(x_)
        Y_mfcc = self.mfcc_transform(y_)
        
        # Flatten for cosine similarity
        X_flat = X_mfcc.view(X_mfcc.size(0), -1)
        Y_flat = Y_mfcc.view(Y_mfcc.size(0), -1)
        
        # Cosine similarity
        X_norm = F.normalize(X_flat, dim=1)
        Y_norm = F.normalize(Y_flat, dim=1)
        cosine_sim = (X_norm * Y_norm).sum(dim=1).mean()
        
        return 1.0 - cosine_sim
    
    def _snr_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio loss"""
        signal_power = torch.sum(x ** 2, dim=(-1, -2)) + 1e-9
        noise_power = torch.sum((x - y) ** 2, dim=(-1, -2)) + 1e-9
        snr = 10.0 * torch.log10(signal_power / noise_power)
        return -snr.mean() / 10.0  # Negative SNR (we want high SNR)
    
    def _spectral_features_loss(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Spectral features loss (centroid, rolloff, ZCR)"""
        x_ = x.squeeze(1)  # [B, T]
        y_ = y.squeeze(1)  # [B, T]
        
        # Compute STFT
        X_stft = torch.stft(x_, n_fft=1024, hop_length=512, return_complex=True)
        Y_stft = torch.stft(y_, n_fft=1024, hop_length=512, return_complex=True)
        
        X_mag = torch.abs(X_stft)
        Y_mag = torch.abs(Y_stft)
        
        # Spectral centroid
        freqs = torch.linspace(0, self.sample_rate/2, X_mag.size(1), device=x.device)
        X_centroid = torch.sum(freqs.unsqueeze(0).unsqueeze(-1) * X_mag, dim=1) / (torch.sum(X_mag, dim=1) + 1e-9)
        Y_centroid = torch.sum(freqs.unsqueeze(0).unsqueeze(-1) * Y_mag, dim=1) / (torch.sum(Y_mag, dim=1) + 1e-9)
        centroid_loss = F.mse_loss(X_centroid, Y_centroid)
        
        # Spectral rolloff (95th percentile)
        X_rolloff = torch.quantile(X_mag, 0.95, dim=1)
        Y_rolloff = torch.quantile(Y_mag, 0.95, dim=1)
        rolloff_loss = F.mse_loss(X_rolloff, Y_rolloff)
        
        # Zero crossing rate
        X_zcr = torch.mean(torch.diff(torch.sign(x_), dim=1) != 0, dim=1, dtype=torch.float32)
        Y_zcr = torch.mean(torch.diff(torch.sign(y_), dim=1) != 0, dim=1, dtype=torch.float32)
        zcr_loss = F.mse_loss(X_zcr, Y_zcr)
        
        return {
            'spectral_centroid': centroid_loss,
            'spectral_rolloff': rolloff_loss,
            'zero_crossing_rate': zcr_loss
        }
    
    def forward(self, original: torch.Tensor, watermarked: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive imperceptibility loss
        
        Args:
            original: Original audio [B, 1, T]
            watermarked: Watermarked audio [B, 1, T]
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Basic reconstruction losses
        losses['l1'] = F.l1_loss(original, watermarked)
        losses['l2'] = F.mse_loss(original, watermarked)
        
        # Multi-resolution STFT loss
        mrstft_losses = self._multi_res_stft_loss(original, watermarked)
        losses.update(mrstft_losses)
        
        # MFCC loss
        losses['mfcc'] = self._mfcc_loss(original, watermarked)
        
        # SNR loss
        losses['snr'] = self._snr_loss(original, watermarked)
        
        # Spectral features loss
        spectral_losses = self._spectral_features_loss(original, watermarked)
        losses.update(spectral_losses)
        
        # Weighted total loss
        total_loss = sum(
            self.weights.get(key, 1.0) * loss 
            for key, loss in losses.items() 
            if key in self.weights
        )
        losses['total'] = total_loss
        
        return losses

class MessageLoss(nn.Module):
    """
    Message recovery loss for payload accuracy
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predicted_bits: torch.Tensor, target_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute message recovery loss
        
        Args:
            predicted_bits: Predicted bits [B, N_bits]
            target_bits: Target bits [B, N_bits]
        """
        # Binary cross-entropy loss
        bce_loss = self.bce_loss(predicted_bits, target_bits)
        
        # Bit error rate
        predicted_binary = (torch.sigmoid(predicted_bits) > 0.5).float()
        bit_error_rate = (predicted_binary != target_bits).float().mean()
        
        return {
            'message_bce': bce_loss,
            'bit_error_rate': bit_error_rate,
            'message_total': bce_loss
        }

class AdversarialLoss(nn.Module):
    """
    Adversarial loss for training against detection
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, discriminator_output: torch.Tensor, is_real: bool = True) -> torch.Tensor:
        """
        Compute adversarial loss
        
        Args:
            discriminator_output: Discriminator output [B, 1]
            is_real: Whether the input is real or generated
        """
        if is_real:
            # Real samples should be classified as real (1)
            target = torch.ones_like(discriminator_output)
        else:
            # Generated samples should be classified as real (1) for generator
            target = torch.ones_like(discriminator_output)
        
        return F.binary_cross_entropy_with_logits(discriminator_output, target)

# Test the losses
if __name__ == "__main__":
    print("Testing loss functions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 4
    audio_length = 22050  # 1 second
    
    original = torch.randn(batch_size, 1, audio_length, device=device)
    watermarked = original + torch.randn_like(original) * 0.1
    
    # Test imperceptibility loss
    impercept_loss = ImperceptibilityLoss(device=device)
    losses = impercept_loss(original, watermarked)
    
    print("Imperceptibility losses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test message loss
    message_loss = MessageLoss(device=device)
    predicted_bits = torch.randn(batch_size, 1000, device=device)
    target_bits = torch.randint(0, 2, (batch_size, 1000), device=device).float()
    
    msg_losses = message_loss(predicted_bits, target_bits)
    print("\nMessage losses:")
    for key, value in msg_losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("✓ Loss functions test passed!")
