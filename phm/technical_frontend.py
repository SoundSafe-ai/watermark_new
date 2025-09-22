# soundsafe/phm/technical_frontend.py
# Technical frontend (bi-GRU) over decoder telemetry sequences producing:
#   - tech_vec: fixed-dim embedding
#   - tech_scores: {'presence_p_t', 'reliability_t'}
#
# Inputs are NOT raw audio. They are sequences (T≈43) of telemetry features:
#   softbit confidence, slot SNR proxy, slot fill ratio, sync drift,
#   RS errata per frame (if available), etc.

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class TechnicalFrontend(nn.Module):
    def __init__(self, feat_dim: int = 8, hidden: int = 128, layers: int = 2, emb_dim: int = 192, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if layers > 1 else 0.0,
        )
        d = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(d, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim, 2)  # [presence_logit_t, reliability_logit_t]
        )

    @torch.no_grad()
    def infer_features(self, tech_seq: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        tech_seq: [B, T, F_t] telemetry sequence per EUL
        Returns:
          tech_vec: [B, emb_dim]
          tech_scores: {'presence_p_t':[B,1], 'reliability_t':[B,1]}
        """
        out, h = self.gru(tech_seq)            # out: [B,T,d]
        # Temporal pooling (mean + max) for stability
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        pooled = 0.5 * (mean_pool + max_pool)  # [B,d]
        emb = self.proj(pooled)                # [B,emb_dim]
        logits = self.head(emb)                # [B,2]
        presence_p_t = torch.sigmoid(logits[:, :1])
        reliability_t = torch.sigmoid(logits[:, 1:2])
        return emb, {"presence_p_t": presence_p_t, "reliability_t": reliability_t}
