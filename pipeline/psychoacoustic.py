# soundsafe/pipeline/psychoacoustic.py
# Simplified MPEG-1 Model 1 masking (music-biased), with MEL fallback

from __future__ import annotations
import torch
import torch.nn.functional as F

def mag_from_ri(X_ri: torch.Tensor) -> torch.Tensor:
    # X_ri: [B, 2, F, T]
    return torch.sqrt(torch.clamp(X_ri[:,0]**2 + X_ri[:,1]**2, min=1e-12))

def bark_scale(f_hz: torch.Tensor) -> torch.Tensor:
    # Zwicker approximation
    return 13.0 * torch.atan(0.00076 * f_hz) + 3.5 * torch.atan((f_hz/7500.0)**2)

def spreading_function(db_vals: torch.Tensor, bark: torch.Tensor) -> torch.Tensor:
    # Very lightweight spreading across Bark bands
    # db_vals: [B, F, T] in dB
    B, F, T = db_vals.shape
    bark_norm = (bark - bark.min()) / (bark.max() - bark.min() + 1e-9)
    # build a small (F x F) kernel in Bark distance
    D = (bark.view(1,F,1) - bark.view(F,1,1)).abs()  # [F,F,1]
    # 10 dB/Bark drop approx
    K = torch.exp(-2.302585 * D)  # e^{-ln(10)*D}
    K = K / (K.sum(dim=1, keepdim=True) + 1e-9)
    out = torch.einsum('bft,ff1->bft', db_vals, K[...,0])
    return out

def masking_threshold(X_ri: torch.Tensor, sr: int, n_fft: int) -> torch.Tensor:
    """
    Simplified MPEG-1 Model 1:
    - estimate PSD
    - detect maskers (peak pick)
    - spread in Bark domain
    Returns: threshold linear magnitude [B, F, T]
    """
    B, _, F, T = X_ri.shape
    mag = mag_from_ri(X_ri)  # [B, F, T]
    psd_db = 20.0 * torch.log10(torch.clamp(mag, min=1e-9))
    # crude peak emphasis for tonal maskers
    peak_db = F.max_pool2d(psd_db.unsqueeze(1), kernel_size=(5,3), stride=1, padding=(2,1)).squeeze(1)
    tonal = torch.maximum(psd_db, peak_db - 3.0)  # tonal enhancement
    # Bark mapping
    freqs = torch.linspace(0, sr/2, F, device=X_ri.device)
    bark = bark_scale(freqs)
    spread_db = spreading_function(tonal, bark) - 24.0  # global offset
    thr_lin = torch.pow(10.0, spread_db / 20.0)  # back to linear
    return thr_lin

def mel_proxy_threshold(X_ri: torch.Tensor, n_mels: int = 64) -> torch.Tensor:
    # Fast proxy: gate by mel energy; produce per-bin threshold by upsampling
    B, _, F, T = X_ri.shape
    mag = mag_from_ri(X_ri)  # [B,F,T]
    mel = F.avg_pool2d(mag.unsqueeze(1), kernel_size=(F//n_mels,1), stride=(F//n_mels,1)).squeeze(1)  # [B, n_mels, T]
    up = F.interpolate(mel, size=(F,T), mode='nearest')
    return 0.5 * up  # allow ~50% of mel energy per bin by default
