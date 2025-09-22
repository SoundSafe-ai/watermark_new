# soundsafe/phm/telemetry.py
# Canonical PHM I/O dataclasses (shapes documented for clarity).
# These classes do not depend on the INN model directly—only on decoder outputs.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch

@dataclass
class PerceptualInput:
    """
    Raw waveform for the current 1s EUL (or precomputed spec if you choose to pass it).
    """
    audio_1s: torch.Tensor  # [B,1,T] at 22.05 kHz, ~1 sec

@dataclass
class TechnicalTelemetry:
    """
    Decoder/INGEST telemetry for the same EUL (all tensors torch on same device).
    - softbit_conf: [B,T,1] average soft confidence per frame (0..1)
    - slot_snr:     [B,T,1] proxy for per-frame embedding SNR (normalized)
    - slot_fill:    [B,T,1] used slots / available slots
    - sync_drift:   [B,T,1] small resample/tempo deviation estimate
    - rs_errata:    [B,T,1] corrected bytes (or syndrome weight) per frame
    - rs_success:   [B,1]   1 if RS decode succeeded for EUL else 0
    """
    softbit_conf: torch.Tensor
    slot_snr: torch.Tensor
    slot_fill: torch.Tensor
    sync_drift: torch.Tensor
    rs_errata: torch.Tensor
    rs_success: torch.Tensor

    def as_sequence(self) -> torch.Tensor:
        """
        Concatenate frame-wise features -> [B,T,F_t]
        """
        seq = torch.cat([self.softbit_conf, self.slot_snr, self.slot_fill, self.sync_drift, self.rs_errata], dim=-1)  # [B,T,5]
        return seq

@dataclass
class PHMOutput:
    presence_p: torch.Tensor        # [B,1]
    decode_reliability: torch.Tensor# [B,1]
    artifact_risk: torch.Tensor     # [B,1]
    fingerprint: torch.Tensor       # [B,F_fp]
    # Optional: attach raw branch scores if you want for debugging
    perc_scores: Optional[Dict[str, torch.Tensor]] = None
    tech_scores: Optional[Dict[str, torch.Tensor]] = None

