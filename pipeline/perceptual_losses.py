# soundsafe/pipeline/perceptual_losses.py
# Perceptual loss suite for watermark imperceptibility: multi-res STFT, spectral convergence,
# simple MFCC proxy (mel filterbank + DCT), and time-domain SNR.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helpers ----------

def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    # x: [B,T]
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True)  # [B,F,T]
    mag = torch.abs(X)
    return mag

def build_mel_filter(sr: int, n_fft: int, n_mels: int, device) -> torch.Tensor:
    # Simple triangular mel filterbank (HTK-ish)
    def hz_to_mel(f): return 2595 * torch.log10(torch.tensor(1.0, device=device) + f / 700.0)
    def mel_to_hz(m): return 700.0 * (10**(m / 2595.0) - 1.0)
    f_max = sr / 2
    m_min, m_max = hz_to_mel(torch.tensor(0.0, device=device)), hz_to_mel(torch.tensor(f_max, device=device))
    m_points = torch.linspace(m_min, m_max, n_mels + 2, device=device)
    f_points = mel_to_hz(m_points)
    bins = torch.floor((n_fft // 2) * f_points / f_max).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1, device=device)
    for m in range(1, n_mels + 1):
        f_m0, f_m1, f_m2 = bins[m - 1], bins[m], bins[m + 1]
        if f_m1 == f_m0: f_m1 += 1
        if f_m2 == f_m1: f_m2 += 1
        fb[m - 1, f_m0:f_m1] = torch.linspace(0, 1, f_m1 - f_m0, device=device)
        fb[m - 1, f_m1:f_m2] = torch.linspace(1, 0, f_m2 - f_m1, device=device)
    return fb  # [M, F]

def dct_mat(n_mels: int, n_ceps: int, device) -> torch.Tensor:
    # Type-II DCT matrix for MFCC
    n = torch.arange(n_mels, device=device).float()
    k = torch.arange(n_ceps, device=device).float().unsqueeze(1)
    D = torch.cos((torch.pi / n_mels) * (n + 0.5) * k)
    D[0] *= 1.0 / torch.sqrt(torch.tensor(2.0, device=device))
    return D

# ---------- Losses ----------

@dataclass
class MultiResSTFTLoss:
    fft_sizes: List[int] = None
    hops: List[int] = None
    wins: List[int] = None
    mag_weight: float = 1.0
    sc_weight: float = 1.0

    def __post_init__(self):
        if self.fft_sizes is None:
            self.fft_sizes = [512, 1024, 2048]
        if self.hops is None:
            self.hops = [128, 256, 512]
        if self.wins is None:
            self.wins = self.fft_sizes

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x, y: [B,1,T]
        Returns dict with 'mrstft_mag', 'mrstft_sc', 'mrstft_total'
        """
        x_ = x.squeeze(1); y_ = y.squeeze(1)
        mag_loss = 0.0; sc_loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hops, self.wins):
            X = stft_mag(x_, n_fft, hop, win)
            Y = stft_mag(y_, n_fft, hop, win)
            # Spectral magnitude L1
            mag_loss = mag_loss + F.l1_loss(X, Y)
            # Spectral convergence
            sc = torch.norm(X - Y, p='fro') / (torch.norm(X, p='fro') + 1e-9)
            sc_loss = sc_loss + sc
        mag_loss = mag_loss * (self.mag_weight / len(self.fft_sizes))
        sc_loss = sc_loss * (self.sc_weight / len(self.fft_sizes))
        return {
            "mrstft_mag": mag_loss,
            "mrstft_sc": sc_loss,
            "mrstft_total": mag_loss + sc_loss
        }

@dataclass
class MFCCCosineLoss:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    n_ceps: int = 20
    weight: float = 1.0

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_ = x.squeeze(1); y_ = y.squeeze(1)
        X = stft_mag(x_, self.n_fft, self.hop_length, self.n_fft)
        Y = stft_mag(y_, self.n_fft, self.hop_length, self.n_fft)
        fb = build_mel_filter(self.sample_rate, self.n_fft, self.n_mels, device=x.device)  # [M,F]
        Xmel = torch.matmul(fb, X) + 1e-9
        Ymel = torch.matmul(fb, Y) + 1e-9
        Xlog = torch.log(Xmel)
        Ylog = torch.log(Ymel)
        D = dct_mat(self.n_mels, self.n_ceps, x.device)  # [C, M]
        Xmfcc = torch.matmul(D, Xlog)  # [C, T]
        Ymfcc = torch.matmul(D, Ylog)
        # Cosine distance averaged
        x_norm = F.normalize(Xmfcc, dim=0)
        y_norm = F.normalize(Ymfcc, dim=0)
        cos = (x_norm * y_norm).sum(dim=0).mean()
        loss = (1.0 - cos) * self.weight
        return {"mfcc_cos": loss}

@dataclass
class TimeDomainSNRLoss:
    weight: float = 1.0
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        num = torch.sum(x**2, dim=(-1,-2)) + 1e-9
        den = torch.sum((x - y)**2, dim=(-1,-2)) + 1e-9
        snr = 10.0 * torch.log10(num / den)
        # Want HIGH SNR -> minimize negative SNR
        loss = (-snr.mean()) * self.weight / 10.0
        return {"snr": loss}

@dataclass
class CombinedPerceptualLoss:
    mrstft: MultiResSTFTLoss = None
    mfcc: MFCCCosineLoss = None
    snr: TimeDomainSNRLoss = None
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.mrstft is None:
            self.mrstft = MultiResSTFTLoss()
        if self.mfcc is None:
            self.mfcc = MFCCCosineLoss()
        if self.snr is None:
            self.snr = TimeDomainSNRLoss()
        if self.weights is None:
            self.weights = {"mrstft": 1.0, "mfcc": 0.5, "snr": 0.2}

    def __call__(self, original_audio: torch.Tensor, watermarked_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        Lm = self.mrstft(original_audio, watermarked_audio)
        Lc = self.mfcc(original_audio, watermarked_audio)
        Ls = self.snr(original_audio, watermarked_audio)
        total = (self.weights["mrstft"] * Lm["mrstft_total"] +
                 self.weights["mfcc"]  * Lc["mfcc_cos"] +
                 self.weights["snr"]   * Ls["snr"])
        out = {"total_perceptual_loss": total}
        out.update(Lm); out.update(Lc); out.update(Ls)
        return out
