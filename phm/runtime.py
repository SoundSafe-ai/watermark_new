# soundsafe/phm/runtime.py
# Orchestration to run PHM on a single 1s EUL segment:
#   run_phm_eul(perceptual_model, technical_model, fusion_head, perceptual_input, technical_telemetry, mode='scan'|'full')
#
# - 'scan' mode can downscale mel bins or skip some features to run faster (kept simple here).
# - No embedding/decoding here; assumes INN pipeline produced telemetry already.

from __future__ import annotations
from typing import Literal
import torch
from phm.perceptual_frontend import PerceptualFrontend
from phm.technical_frontend import TechnicalFrontend
from phm.fusion_head import FusionHead
from phm.telemetry import PerceptualInput, TechnicalTelemetry, PHMOutput

@torch.no_grad()
def run_phm_eul(
    perceptual_model: PerceptualFrontend,
    technical_model: TechnicalFrontend,
    fusion_head: FusionHead,
    perceptual_input: PerceptualInput,
    technical_telemetry: TechnicalTelemetry,
    mode: Literal['scan','full'] = 'full',
) -> PHMOutput:
    """
    Returns fused PHMOutput for one EUL. Assumes all tensors are on the same device/dtype.
    """
    x = perceptual_input.audio_1s  # [B,1,T]
    # Fast path options (you can add more shortcuts for 'scan' mode if needed)
    if mode == 'scan':
        # For scan, you might set model.use_logmel=True with fewer mel bins when instantiating,
        # or downsample audio beforehand. We keep invocation identical here.
        pass

    perc_vec, perc_scores = perceptual_model.infer_features(x)            # [B,Dp], dict
    tech_seq = technical_telemetry.as_sequence()                           # [B,T,F_t]
    tech_vec, tech_scores = technical_model.infer_features(tech_seq)       # [B,Dt], dict

    fused = fusion_head(perc_vec, tech_vec)                                 # dict

    return PHMOutput(
        presence_p=fused["presence_p"],
        decode_reliability=fused["decode_reliability"],
        artifact_risk=fused["artifact_risk"],
        fingerprint=fused["fingerprint"],
        perc_scores=perc_scores,
        tech_scores=tech_scores,
    )
