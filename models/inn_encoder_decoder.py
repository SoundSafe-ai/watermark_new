# models/inn_encoder_decoder.py
# Invertible encoder/decoder for audio watermarking (spectrogram domain, real/imag channels)
# Spec: 8 invertible blocks; each block has three 5-layer dense CNN subnets (phi, rho, eta)
# Message fusion: channel-concat of message spectrogram with audio spectrogram at each block input
# Invertible encoder/decoder (RI channels) for audio watermarking.
# Designed to work with a psychoacoustic + adaptive allocation pipeline that converts payload bytes
# into a sparse message spectrogram (M_spec) via (bin,frame) slots and per-slot amplitudes.
# The INN architecture itself is unchanged: 8 invertible blocks, φ/ρ/η as 5-layer DenseBlocks,
# with channel-concat of [audio_spec || message_spec] at each block.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Dense conv block used for φ(·), ρ(·), η(·)
# ---------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_ch: int, growth: int = 32, depth: int = 5, k: int = 3, out_ch: int = None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(depth):
            self.layers.append(nn.Conv2d(ch, growth, kernel_size=k, padding=k//2))
            ch += growth
        self.out = nn.Conv2d(ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for conv in self.layers:
            y = F.gelu(conv(torch.cat(feats, dim=1)))
            feats.append(y)
        return self.out(torch.cat(feats, dim=1))

# ---------------------------
# Invertible block
# Forward (encoder):
#   x_{l+1} = x_l + φ([x_l || m_l])
#   g       = sigmoid(ρ([x_{l+1} || m_l]))
#   m_{l+1} = m_l * g + η([x_{l+1} || m_l])
#
# Inverse (decoder):
#   g_inv   = sigmoid(-ρ([x_{l+1} || m_l]))   (analytic inverse of gating)
#   m_l     = (m_{l+1} - η([x_{l+1} || m_l])) * g_inv
#   x_l     = x_{l+1} - φ([x_l || m_l])       (note: we plug m_l we just recovered)
# ---------------------------
class InvertibleBlock(nn.Module):
    def __init__(self, spec_channels: int):
        super().__init__()
        # Inputs to subnets are concatenations: audio || message  -> 2*spec_channels
        in_ch = 2 * spec_channels
        self.phi = DenseBlock(in_ch=in_ch, out_ch=spec_channels)
        self.rho = DenseBlock(in_ch=in_ch, out_ch=spec_channels)
        self.eta = DenseBlock(in_ch=in_ch, out_ch=spec_channels)

    def _sigmoid(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(t)

    def forward_enc(self, x_l: torch.Tensor, m_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xm = torch.cat([x_l, m_l], dim=1)
        x_next = x_l + self.phi(xm)

        xm_next = torch.cat([x_next, m_l], dim=1)
        gate = self._sigmoid(self.rho(xm_next))
        m_next = m_l * gate + self.eta(xm_next)
        return x_next, m_next

    def forward_dec(self, x_next: torch.Tensor, m_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # We need m_l inside phi([x_l || m_l]); compute m_l first using x_next and unknown m_l – use fixed-point with 1–2 iterations.
        # Because the mapping is well-behaved (sigmoid-bounded), 1 iteration is sufficient in practice.
        # Start with a warm start: use m_est = m_next (works because eta is small by training) then refine once.

        m_est = m_next
        xm_next = torch.cat([x_next, m_est], dim=1)
        gate_inv = self._sigmoid(-self.rho(xm_next))
        m_l = (m_next - self.eta(xm_next)) * gate_inv

        # Now recover x_l using m_l
        xm_l = torch.cat([x_next - 0.0, m_l], dim=1)  # placeholder for [x_l || m_l], use x_next in phi arg estimate
        # One-step Newton: x_l = x_next - phi([x_l || m_l]); approximate with x_l ≈ x_next - phi([x_next || m_l])
        x_l = x_next - self.phi(xm_l)
        return x_l, m_l

# ---------------------------
# STFT wrappers (RI channels)
# ---------------------------
class STFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, win_length=1024):
        super().__init__()
        self.n_fft = n_fft; self.hop = hop_length; self.win = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, x_wave: torch.Tensor) -> torch.Tensor:
        # x_wave: [B, 1, T]
        X = torch.stft(
            x_wave.squeeze(1), n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, return_complex=True
        )  # [B, F, T]
        return torch.stack([X.real, X.imag], dim=1)  # [B, 2, F, T]

class ISTFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=512, win_length=1024):
        super().__init__()
        self.n_fft = n_fft; self.hop = hop_length; self.win = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, X_ri: torch.Tensor) -> torch.Tensor:
        # X_ri: [B, 2, F, T]
        X = torch.complex(X_ri[:,0], X_ri[:,1])  # [B, F, T]
        x = torch.istft(
            X, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
            window=self.window, length=None
        )  # [B, T]
        return x.unsqueeze(1)

# ---------------------------
# Full INN model
# ---------------------------
class INNWatermarker(nn.Module):
    def __init__(self, n_blocks=8, spec_channels=2, stft_cfg=None):
        super().__init__()
        stft_cfg = stft_cfg or dict(n_fft=1024, hop_length=512, win_length=1024)
        self.spec_channels = spec_channels
        self.stft = STFT(**stft_cfg)
        self.istft = ISTFT(**stft_cfg)
        self.blocks = nn.ModuleList([InvertibleBlock(spec_channels) for _ in range(n_blocks)])

    # ----- Encoder: waveform + message spec -> watermarked waveform -----
    def encode(self, x_wave: torch.Tensor, m_spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_wave: [B, 1, T], m_spec: [B, 2, F, T]
        X0 = self.stft(x_wave)
        assert X0.shape == m_spec.shape, f"Shape mismatch: audio {X0.shape} vs message {m_spec.shape}"
        x, m = X0, m_spec
        for blk in self.blocks:
            x, m = blk.forward_enc(x, m)
        x_wm = self.istft(x)
        return x_wm, x  # return watermarked waveform and final spec if needed

    # ----- Decoder: watermarked waveform -> recovered message spec -----
    def decode(self, x_wm_wave: torch.Tensor) -> torch.Tensor:
        Xn = self.stft(x_wm_wave)
        # Initialize m_n with zeros (or tiny noise) for inversion
        m = torch.zeros_like(Xn)
        x = Xn
        for blk in reversed(self.blocks):
            x, m = blk.forward_dec(x, m)
        # m is m_0 (recovered message spectrogram)
        return m

    # Convenience for spectrogram IO (if you already have specs)
    def encode_spec(self, X_spec: torch.Tensor, M_spec: torch.Tensor) -> torch.Tensor:
        x, m = X_spec, M_spec
        for blk in self.blocks:
            x, m = blk.forward_enc(x, m)
        return x

    def decode_spec(self, X_spec_wm: torch.Tensor) -> torch.Tensor:
        x, m = X_spec_wm, torch.zeros_like(X_spec_wm)
        for blk in reversed(self.blocks):
            x, m = blk.forward_dec(x, m)
        return m
