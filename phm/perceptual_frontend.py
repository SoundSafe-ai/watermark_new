# soundsafe/phm/perceptual_frontend.py
# Mobile-friendly perceptual frontend (MobileNetV3-Small style) producing:
#   - perc_vec: fixed-dim embedding
#   - perc_scores: {'presence_p', 'artifact_risk'}
#
# Notes:
# - No embedding/decoding/ECC here. This is an assessor-only module.
# - Front-end matches encoder STFT config (sr=22.05kHz, n_fft=1024, hop=512).
# - Uses log-mel or |STFT| magnitude; default is log-mel for perceptual alignment.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Signal frontend (log-mel) ---------

def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    # x: [B, 1, T] -> mag: [B, F, T']
    window = torch.hann_window(win, device=x.device)
    X = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win,
                   window=window, return_complex=True)  # [B,F,T]
    return torch.abs(X)

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

# --------- MobileNetV3-Small style backbone (very compact) ---------

class SqueezeExcite(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        s = x.mean(dim=(2,3), keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, exp=4, se=True, stride=1):
        super().__init__()
        mid = in_ch * exp
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.exp = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, k, stride=stride, padding=k//2, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.se = SqueezeExcite(mid) if se else nn.Identity()
        self.pw = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        h = F.hardswish(self.bn1(self.exp(x)))
        h = F.hardswish(self.bn2(self.dw(h)))
        h = self.se(h)
        h = self.bn3(self.pw(h))
        return x + h if self.use_res else h

class PerceptualFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop: int = 512,
        n_mels: int = 64,
        emb_dim: int = 192,
        use_logmel: bool = True,
    ):
        super().__init__()
        self.sr, self.n_fft, self.hop = sample_rate, n_fft, hop
        self.n_mels = n_mels
        self.use_logmel = use_logmel
        # Conv stem expects [B,1,H,W]; we will format spec as [B,1,M,T]
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        self.blocks = nn.Sequential(
            MBConv(16, 16, k=3, exp=2, se=True, stride=1),
            MBConv(16, 24, k=3, exp=3, se=False, stride=2),
            MBConv(24, 24, k=3, exp=3, se=False, stride=1),
            MBConv(24, 40, k=5, exp=3, se=True, stride=2),
            MBConv(40, 40, k=5, exp=3, se=True, stride=1),
            MBConv(40, 64, k=5, exp=4, se=True, stride=2),
            MBConv(64, 96, k=5, exp=4, se=True, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.emb = nn.Linear(96, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Hardswish(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, 2)  # [presence_logit, artifact_logit]
        )

    def _spec(self, x_wave: torch.Tensor) -> torch.Tensor:
        mag = stft_mag(x_wave, self.n_fft, self.hop, self.n_fft)  # [B,F,T]
        if self.use_logmel:
            fb = build_mel_filter(self.sr, self.n_fft, self.n_mels, device=x_wave.device)  # [M,F]
            mel = torch.matmul(fb, mag) + 1e-9  # [B,M,T]
            spec = torch.log(mel)
            spec = spec.unsqueeze(1)  # [B,1,M,T]
        else:
            spec = torch.log(mag + 1e-9).unsqueeze(1)  # [B,1,F,T]
        return spec

    @torch.no_grad()
    def infer_features(self, x_wave: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x_wave: [B,1,T] ~1s EUL
        Returns:
          perc_vec: [B, emb_dim]
          perc_scores: {'presence_p':[B,1], 'artifact_risk':[B,1]}
        """
        spec = self._spec(x_wave)
        h = self.stem(spec)
        h = self.blocks(h)
        g = self.pool(h).squeeze(-1).squeeze(-1)  # [B,C]
        emb = self.emb(g)                         # [B,emb_dim]
        logits = self.head(emb)                   # [B,2]
        presence_p = torch.sigmoid(logits[:, :1])
        artifact_risk = torch.sigmoid(logits[:, 1:2])
        return emb, {"presence_p": presence_p, "artifact_risk": artifact_risk}
