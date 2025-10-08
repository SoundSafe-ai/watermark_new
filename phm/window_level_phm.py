# soundsafe/phm/window_level_phm.py
# PHM integration for window-level bit scoring in micro-chunking system
# Provides window-level presence, reliability, and artifact detection

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from phm.perceptual_frontend import PerceptualFrontend
from phm.technical_frontend import TechnicalFrontend
from phm.fusion_head import FusionHead
from phm.telemetry import TechnicalTelemetry, PerceptualInput, PHMOutput


class WindowLevelPHM:
    """
    PHM system for window-level bit scoring in micro-chunking system.
    Processes each micro-chunk individually and provides detailed metrics.
    """
    
    def __init__(self, device: str = "cuda", 
                 perceptual_emb_dim: int = 192,
                 technical_emb_dim: int = 192,
                 fusion_d_model: int = 192):
        """
        Args:
            device: Device to run on
            perceptual_emb_dim: Perceptual frontend embedding dimension
            technical_emb_dim: Technical frontend embedding dimension
            fusion_d_model: Fusion head model dimension
        """
        self.device = device
        
        # Initialize PHM components
        self.perceptual_model = PerceptualFrontend(emb_dim=perceptual_emb_dim).to(device)
        self.technical_model = TechnicalFrontend(emb_dim=technical_emb_dim).to(device)
        self.fusion_head = FusionHead(
            d_perc=perceptual_emb_dim,
            d_tech=technical_emb_dim,
            d_model=fusion_d_model
        ).to(device)
        
        # Set to eval mode (inference only)
        self.perceptual_model.eval()
        self.technical_model.eval()
        self.fusion_head.eval()
        
    def score_window_bits(self, window_audio: torch.Tensor, 
                         window_telemetry: Optional[Dict] = None,
                         recovered_bits: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Score bits for a single micro-window.
        
        Args:
            window_audio: Audio window [1, T] ~15ms
            window_telemetry: Optional technical telemetry dict
            recovered_bits: Optional recovered bits [1, n_bits]
            
        Returns:
            Dict with window-level scores
        """
        if window_audio.dim() != 2 or window_audio.size(0) != 1:
            raise ValueError(f"Expected window_audio shape [1, T], got {window_audio.shape}")
        
        with torch.no_grad():
            # Perceptual analysis
            perc_vec, perc_scores = self.perceptual_model.infer_features(window_audio)
            
            # Technical analysis
            if window_telemetry is not None:
                tech_seq = self._build_technical_sequence(window_telemetry, window_audio.device)
                tech_vec, tech_scores = self.technical_model.infer_features(tech_seq)
            else:
                # Use default technical features
                tech_vec, tech_scores = self._default_technical_features(window_audio)
            
            # Fusion
            fused = self.fusion_head(perc_vec, tech_vec)
            
            # Bit-level analysis if bits provided
            bit_scores = {}
            if recovered_bits is not None and recovered_bits.numel() > 0:
                bit_scores = self._analyze_bits(recovered_bits, window_audio)
            
            # Combine all scores
            window_scores = {
                'presence_p': fused['presence_p'],
                'decode_reliability': fused['decode_reliability'],
                'artifact_risk': fused['artifact_risk'],
                'fingerprint': fused['fingerprint'],
                'perc_scores': perc_scores,
                'tech_scores': tech_scores,
                **bit_scores
            }
            
            return window_scores
    
    def score_multiple_windows(self, windows: List[torch.Tensor],
                              window_telemetries: Optional[List[Dict]] = None,
                              recovered_bits_list: Optional[List[torch.Tensor]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Score bits for multiple micro-windows.
        
        Args:
            windows: List of audio windows [1, T]
            window_telemetries: Optional list of technical telemetry dicts
            recovered_bits_list: Optional list of recovered bits
            
        Returns:
            List of window-level score dicts
        """
        if window_telemetries is None:
            window_telemetries = [None] * len(windows)
        if recovered_bits_list is None:
            recovered_bits_list = [None] * len(windows)
        
        all_scores = []
        for i, window in enumerate(windows):
            scores = self.score_window_bits(
                window,
                window_telemetries[i],
                recovered_bits_list[i]
            )
            all_scores.append(scores)
        
        return all_scores
    
    def aggregate_window_scores(self, window_scores_list: List[Dict[str, torch.Tensor]],
                               overlap_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Aggregate scores from multiple overlapping windows.
        
        Args:
            window_scores_list: List of window-level score dicts
            overlap_ratio: Overlap ratio for weighting
            
        Returns:
            Aggregated scores dict
        """
        if not window_scores_list:
            return {}
        
        # Extract common keys
        keys = window_scores_list[0].keys()
        aggregated = {}
        
        for key in keys:
            if key in ['perc_scores', 'tech_scores']:
                # Keep as lists for detailed analysis
                aggregated[key] = [scores[key] for scores in window_scores_list]
                continue
            
            # Stack tensors
            tensors = []
            for scores in window_scores_list:
                if key in scores and isinstance(scores[key], torch.Tensor):
                    tensors.append(scores[key])
            
            if not tensors:
                continue
            
            stacked = torch.stack(tensors, dim=0)  # [n_windows, ...]
            
            # Apply overlap weighting
            if overlap_ratio > 0:
                weights = torch.ones(len(tensors), device=tensors[0].device)
                for i in range(1, len(weights)):
                    # Reduce weight for overlapping regions
                    weights[i] *= (1 - overlap_ratio)
                
                # Weighted average
                weighted_sum = (stacked * weights.unsqueeze(-1)).sum(dim=0)
                total_weight = weights.sum()
                aggregated[key] = weighted_sum / total_weight
            else:
                # Simple average
                aggregated[key] = stacked.mean(dim=0)
        
        return aggregated
    
    def _build_technical_sequence(self, telemetry: Dict, device: torch.device) -> torch.Tensor:
        """Build technical telemetry sequence from dict."""
        # Extract features from telemetry dict
        features = []
        
        # Common technical features
        if 'softbit_confidence' in telemetry:
            features.append(telemetry['softbit_confidence'])
        else:
            features.append(torch.tensor(0.5, device=device))  # Default confidence
        
        if 'snr_proxy' in telemetry:
            features.append(telemetry['snr_proxy'])
        else:
            features.append(torch.tensor(1.0, device=device))  # Default SNR
        
        if 'slot_fill_ratio' in telemetry:
            features.append(telemetry['slot_fill_ratio'])
        else:
            features.append(torch.tensor(1.0, device=device))  # Default fill ratio
        
        if 'sync_drift' in telemetry:
            features.append(telemetry['sync_drift'])
        else:
            features.append(torch.tensor(0.0, device=device))  # Default drift
        
        if 'rs_errata' in telemetry:
            features.append(telemetry['rs_errata'])
        else:
            features.append(torch.tensor(0.0, device=device))  # Default errata
        
        if 'confidence_max' in telemetry:
            features.append(telemetry['confidence_max'])
        else:
            features.append(torch.tensor(0.5, device=device))  # Default max conf
        
        if 'confidence_std' in telemetry:
            features.append(telemetry['confidence_std'])
        else:
            features.append(torch.tensor(0.1, device=device))  # Default std
        
        if 'magnitude_mean' in telemetry:
            features.append(telemetry['magnitude_mean'])
        else:
            features.append(torch.tensor(0.1, device=device))  # Default magnitude
        
        # Stack features [1, T, F] where T is time frames, F is feature dim
        feature_tensor = torch.stack(features, dim=-1).unsqueeze(0)  # [1, 1, F]
        
        # Expand to multiple time frames if needed
        n_frames = 10  # Default number of frames
        feature_tensor = feature_tensor.expand(1, n_frames, -1)  # [1, T, F]
        
        return feature_tensor
    
    def _default_technical_features(self, window_audio: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate default technical features from audio."""
        # Simple technical features based on audio characteristics
        device = window_audio.device
        
        # Energy-based features
        energy = window_audio.pow(2).mean()
        energy_db = 10 * torch.log10(energy + 1e-9)
        
        # Spectral features
        fft = torch.fft.fft(window_audio, dim=-1)
        magnitude = torch.abs(fft)
        spectral_centroid = (magnitude * torch.arange(magnitude.size(-1), device=device)).sum() / magnitude.sum()
        
        # Create technical sequence
        tech_seq = torch.tensor([
            [0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 0.1, 0.1]  # Default features
        ], device=device).unsqueeze(0)  # [1, 1, 8]
        
        # Expand to multiple frames
        tech_seq = tech_seq.expand(1, 10, -1)  # [1, 10, 8]
        
        # Generate technical scores
        tech_scores = {
            'presence_p_t': torch.sigmoid(torch.tensor(0.0, device=device)),
            'reliability_t': torch.sigmoid(torch.tensor(0.0, device=device))
        }
        
        return tech_seq, tech_scores
    
    def _analyze_bits(self, bits: torch.Tensor, window_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Analyze recovered bits for quality metrics."""
        if bits.numel() == 0:
            return {}
        
        # Bit-level statistics
        bit_confidence = torch.sigmoid(torch.abs(bits.float() - 0.5) * 10)  # Higher for more confident bits
        bit_entropy = -bits.float() * torch.log(bits.float() + 1e-9) - (1 - bits.float()) * torch.log(1 - bits.float() + 1e-9)
        
        # Consistency metrics
        bit_consistency = 1.0 - bit_entropy.mean()  # Higher is more consistent
        
        return {
            'bit_confidence': bit_confidence.mean(),
            'bit_consistency': bit_consistency,
            'bit_entropy': bit_entropy.mean(),
            'n_bits': torch.tensor(bits.numel(), dtype=torch.float32)
        }
    
    def get_window_quality_summary(self, window_scores: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Get human-readable quality summary for a window."""
        summary = {}
        
        # Extract scalar values
        for key, value in window_scores.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    summary[key] = value.item()
                elif value.dim() == 0:
                    summary[key] = value.item()
                else:
                    summary[key] = value.mean().item()
            else:
                summary[key] = value
        
        return summary
    
    def detect_watermark_presence(self, window_scores: Dict[str, torch.Tensor], 
                                 threshold: float = 0.5) -> bool:
        """Detect if watermark is present in window based on scores."""
        presence_p = window_scores.get('presence_p', torch.tensor(0.0))
        if isinstance(presence_p, torch.Tensor):
            presence_p = presence_p.mean().item()
        
        return presence_p > threshold
    
    def assess_reliability(self, window_scores: Dict[str, torch.Tensor], 
                          threshold: float = 0.5) -> bool:
        """Assess if window decoding is reliable."""
        reliability = window_scores.get('decode_reliability', torch.tensor(0.0))
        if isinstance(reliability, torch.Tensor):
            reliability = reliability.mean().item()
        
        return reliability > threshold
    
    def detect_artifacts(self, window_scores: Dict[str, torch.Tensor], 
                        threshold: float = 0.5) -> bool:
        """Detect if window contains artifacts."""
        artifact_risk = window_scores.get('artifact_risk', torch.tensor(0.0))
        if isinstance(artifact_risk, torch.Tensor):
            artifact_risk = artifact_risk.mean().item()
        
        return artifact_risk > threshold


def create_window_telemetry_from_bits(recovered_bits: torch.Tensor, 
                                    window_audio: torch.Tensor,
                                    slots: List[Tuple[int, int]]) -> Dict:
    """
    Create technical telemetry from recovered bits and audio.
    
    Args:
        recovered_bits: Recovered bits [1, n_bits]
        window_audio: Audio window [1, T]
        slots: List of (freq_bin, time_frame) slots
        
    Returns:
        Technical telemetry dict
    """
    device = window_audio.device
    
    # Calculate bit confidence
    bit_confidence = torch.sigmoid(torch.abs(recovered_bits.float() - 0.5) * 10)
    
    # Calculate SNR proxy (simplified)
    audio_energy = window_audio.pow(2).mean()
    snr_proxy = torch.clamp(audio_energy * 10, 0.1, 10.0)
    
    # Calculate slot fill ratio
    slot_fill_ratio = len(slots) / max(1, recovered_bits.size(1))
    
    # Calculate sync drift (simplified)
    sync_drift = torch.tensor(0.0, device=device)  # Would need actual sync analysis
    
    # Calculate RS errata (simplified)
    rs_errata = torch.tensor(0.0, device=device)  # Would need actual RS analysis
    
    # Calculate confidence statistics
    confidence_max = bit_confidence.max()
    confidence_std = bit_confidence.std()
    
    # Calculate magnitude mean
    magnitude_mean = audio_energy.sqrt()
    
    return {
        'softbit_confidence': bit_confidence.mean(),
        'snr_proxy': snr_proxy,
        'slot_fill_ratio': torch.tensor(slot_fill_ratio, device=device),
        'sync_drift': sync_drift,
        'rs_errata': rs_errata,
        'confidence_max': confidence_max,
        'confidence_std': confidence_std,
        'magnitude_mean': magnitude_mean
    }
