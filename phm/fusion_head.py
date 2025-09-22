# soundsafe/phm/fusion_head.py
# Fusion of perceptual + technical embeddings using attention, producing:
#   - presence_p (primary)
#   - decode_reliability
#   - artifact_risk
#
# You can extend with attack classification or fingerprint heads as needed.

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionHead(nn.Module):
    def __init__(self, d_perc: int = 192, d_tech: int = 192, d_model: int = 192, n_heads: int = 3, fingerprint_dim: int = 128):
        super().__init__()
        self.proj_p = nn.Linear(d_perc, d_model)
        self.proj_t = nn.Linear(d_tech, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Heads
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)  # [presence_logit, reliability_logit, artifact_logit]
        )
        self.fp = nn.Linear(d_model, fingerprint_dim)

    @torch.no_grad()
    def forward(self, perc_vec: torch.Tensor, tech_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        perc_vec: [B, Dp], tech_vec: [B, Dt]
        Returns dict with probabilities and fingerprint.
        """
        B = perc_vec.size(0)
        p = self.proj_p(perc_vec).unsqueeze(1)  # [B,1,D]
        t = self.proj_t(tech_vec).unsqueeze(1)  # [B,1,D]
        seq = torch.cat([p, t], dim=1)          # [B,2,D]
        # Self-attention over the 2-token sequence
        attn_out, _ = self.attn(seq, seq, seq)  # [B,2,D]
        x = self.norm1(seq + attn_out)
        x = self.norm2(x + self.ffn(x))         # [B,2,D]
        # Pool tokens (mean)
        z = x.mean(dim=1)                        # [B,D]
        logits = self.cls(z)                     # [B,3]
        presence_p = torch.sigmoid(logits[:, 0:1])
        decode_rel = torch.sigmoid(logits[:, 1:2])
        artifact_risk = torch.sigmoid(logits[:, 2:3])
        fingerprint = F.normalize(self.fp(z), dim=-1)
        return {"presence_p": presence_p, "decode_reliability": decode_rel, "artifact_risk": artifact_risk, "fingerprint": fingerprint}
