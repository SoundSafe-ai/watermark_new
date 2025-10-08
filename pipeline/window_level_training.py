# soundsafe/pipeline/window_level_training.py
# Window-level training pipeline for micro-chunking system
# Replaces 1s targets with window-level labels and overlap duplication

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

from models.inn_encoder_decoder import INNWatermarker
from pipeline.micro_chunking import MicroChunker, DeterministicMapper, EnhancedDeterministicMapper, apply_psychoacoustic_gate, fuse_symbol_evidence, fuse_symbol_evidence_enhanced
from pipeline.error_correction import compute_crc_penalty, add_crc_to_symbol
from pipeline.sync_anchors import SyncAnchor
from pipeline.psychoacoustic import mel_proxy_threshold


class WindowLevelTrainer:
    """
    Training pipeline for micro-chunking system with window-level labels.
    Handles overlap duplication and psychoacoustic gating.
    """
    
    def __init__(self, window_ms: int = 15, overlap_ratio: float = 0.5, 
                 sr: int = 22050, n_fft: int = 1024, hop: int = 512,
                 mapper_seed: int = 42, sync_strength: float = 0.1):
        """
        Args:
            window_ms: Micro-chunk duration in milliseconds
            overlap_ratio: Overlap between consecutive chunks
            sr: Sample rate
            n_fft: FFT size
            hop: Hop length
            mapper_seed: Seed for deterministic mapping
            sync_strength: Strength of sync pattern
        """
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.mapper_seed = mapper_seed
        self.sync_strength = sync_strength
        
        # Initialize components
        self.micro_chunker = MicroChunker(window_ms, overlap_ratio, sr)
        self.mapper = DeterministicMapper(mapper_seed, n_fft, hop, sr, window_ms)
        self.sync_anchor = SyncAnchor(sync_length_ms=50, sr=sr, n_fft=n_fft, hop=hop)
        
    def build_window_level_plan(self, model: INNWatermarker, x: torch.Tensor, 
                               target_bits_per_window: int, 
                               base_symbol_amp: float = 0.1) -> Dict:
        """
        Build window-level plan for 1s audio segment.
        
        Args:
            model: INNWatermarker model
            x: Audio tensor [1, T] where T ≈ 22050 samples
            target_bits_per_window: Target bits per micro-chunk
            base_symbol_amp: Base symbol amplitude
            
        Returns:
            Dict with window plans and metadata
        """
        if x.dim() != 2 or x.size(0) != 1:
            raise ValueError(f"Expected x shape [1, T], got {x.shape}")
        
        # Add sync pattern
        x_with_sync = self.sync_anchor.embed_sync_pattern(x, self.sync_strength)
        
        # Create micro-chunks
        windows = self.micro_chunker.chunk_1s_segment(x_with_sync)
        n_windows = len(windows)
        
        # Build plan for each window
        window_plans = []
        total_slots = 0
        
        for i, window in enumerate(windows):
            # Map window to slots
            slots = self.mapper.map_window_to_slots(
                window_idx=i,
                n_windows=n_windows,
                target_bits=target_bits_per_window,
                audio_content=window
            )
            
            # Apply psychoacoustic gating
            masked_slots = apply_psychoacoustic_gate(window, slots, model, self.n_fft, self.hop)
            
            # Calculate per-slot amplitudes
            amp_per_slot = self._calculate_slot_amplitudes(window, masked_slots, model)
            
            window_plan = {
                'window': window,
                'slots': masked_slots,
                'amp_per_slot': amp_per_slot,
                'n_slots': len(masked_slots),
                'window_idx': i
            }
            
            window_plans.append(window_plan)
            total_slots += len(masked_slots)
        
        return {
            'window_plans': window_plans,
            'n_windows': n_windows,
            'total_slots': total_slots,
            'overlap_ratio': self.overlap_ratio,
            'window_ms': self.window_ms
        }
    
    def _calculate_slot_amplitudes(self, window: torch.Tensor, slots: List[Tuple[int, int]], 
                                  model: INNWatermarker) -> torch.Tensor:
        """Calculate per-slot amplitude scaling based on psychoacoustic analysis."""
        if not slots:
            return torch.tensor([], dtype=torch.float32)
        
        # Get STFT of window
        X = model.stft(window)  # [1, 2, F, T]
        mag = torch.sqrt(torch.clamp(X[:, 0]**2 + X[:, 1]**2, min=1e-12))  # [1, F, T]
        
        # Get psychoacoustic threshold
        thr = mel_proxy_threshold(X, n_mels=64)  # [1, F, T]
        
        # Calculate amplitude scaling for each slot
        amp_per_slot = []
        for f, t in slots:
            if f < mag.size(1) and t < mag.size(2):
                # Higher threshold -> more headroom -> higher amplitude
                headroom = thr[0, f, t] / (mag[0, f, t] + 1e-9)
                amp = torch.clamp(headroom, 0.1, 2.0)  # Reasonable range
            else:
                amp = torch.tensor(1.0)
            amp_per_slot.append(amp)
        
        return torch.stack(amp_per_slot)
    
    def duplicate_labels_across_overlaps(self, window_plans: List[Dict], 
                                       payload_bits: torch.Tensor) -> List[Dict]:
        """
        Duplicate payload bits across overlapping windows for redundancy.
        
        Args:
            window_plans: List of window plans
            payload_bits: Payload bits [1, total_bits]
            
        Returns:
            Updated window plans with duplicated bits
        """
        if not window_plans:
            return window_plans
        
        n_windows = len(window_plans)
        total_payload_bits = payload_bits.size(1)
        
        # Calculate bits per window (with overlap consideration)
        bits_per_window = total_payload_bits // n_windows
        if bits_per_window == 0:
            bits_per_window = 1
        
        # Distribute payload bits across windows
        cursor = 0
        for i, plan in enumerate(window_plans):
            n_slots = plan['n_slots']
            if n_slots == 0:
                plan['bits'] = torch.tensor([], dtype=torch.long)
                continue
            
            # Get bits for this window
            window_bits = payload_bits[:, cursor:cursor + min(bits_per_window, n_slots)]
            if window_bits.size(1) < n_slots:
                # Pad if needed
                pad_bits = torch.zeros(1, n_slots - window_bits.size(1), 
                                     dtype=torch.long, device=window_bits.device)
                window_bits = torch.cat([window_bits, pad_bits], dim=1)
            
            plan['bits'] = window_bits
            cursor += bits_per_window
            
            # Handle overlap: duplicate bits in overlapping regions
            if i > 0 and self.overlap_ratio > 0:
                # Get previous window's bits for overlap
                prev_plan = window_plans[i-1]
                if 'bits' in prev_plan and prev_plan['bits'].numel() > 0:
                    # Calculate overlap region
                    overlap_bits = int(n_slots * self.overlap_ratio)
                    if overlap_bits > 0:
                        # Mix bits in overlap region (majority voting)
                        prev_bits = prev_plan['bits'][:, :overlap_bits]
                        curr_bits = window_bits[:, :overlap_bits]
                        
                        # Simple majority voting
                        mixed_bits = ((prev_bits + curr_bits) > 0).long()
                        window_bits[:, :overlap_bits] = mixed_bits
                        plan['bits'] = window_bits
        
        return window_plans
    
    def build_message_spec_from_window_plan(self, window_plan: Dict, 
                                          base_symbol_amp: float) -> torch.Tensor:
        """
        Build message spectrogram from window plan.
        
        Args:
            window_plan: Window plan dict
            base_symbol_amp: Base symbol amplitude
            
        Returns:
            Message spectrogram [1, 2, F, T]
        """
        window = window_plan['window']
        slots = window_plan['slots']
        amp_per_slot = window_plan['amp_per_slot']
        bits = window_plan['bits']
        
        if not slots or bits.numel() == 0:
            # Return empty spectrogram using adaptive STFT
            X = self.stft_micro(window)
            return torch.zeros(1, 2, X.shape[-2], X.shape[-1], device=window.device)
        
        # Get STFT dimensions using adaptive STFT
        X = self.stft_micro(window)
        F_, T_ = X.shape[-2], X.shape[-1]
        
        # Create message spectrogram
        M_spec = torch.zeros(1, 2, F_, T_, device=window.device)
        
        # Place symbols at slots
        n_slots = min(len(slots), bits.size(1))
        for s in range(n_slots):
            f, t = slots[s]
            if f < F_ and t < T_:
                # Convert bit to symbol
                bit_val = bits[0, s].item()
                symbol = (bit_val * 2 - 1) * base_symbol_amp  # -amp or +amp
                
                # Apply amplitude scaling
                if s < amp_per_slot.size(0):
                    symbol *= amp_per_slot[s].item()
                
                # Place in real channel
                M_spec[0, 0, f, t] = symbol
        
        return M_spec
    
    def compute_window_level_loss(self, model: INNWatermarker, window_plans: List[Dict],
                                 base_symbol_amp: float, loss_weights: Dict) -> Dict:
        """
        Compute window-level loss for training with batched model calls.
        
        Args:
            model: INNWatermarker model
            window_plans: List of window plans
            base_symbol_amp: Base symbol amplitude
            loss_weights: Loss weight dictionary
            
        Returns:
            Loss dictionary
        """
        total_loss = 0.0
        total_bits = 0
        total_errors = 0
        total_perceptual = 0.0
        
        # Filter out empty plans and prepare for batching
        valid_plans = [plan for plan in window_plans if plan['n_slots'] > 0]
        if not valid_plans:
            return {
                'total_loss': 0.0,
                'ber': 0.0,
                'perceptual_loss': 0.0,
                'total_bits': 0,
                'total_errors': 0,
                'loss_tensor': torch.tensor(0.0, device=next(iter(window_plans))['window'].device if window_plans else torch.device('cpu'))
            }
        
        # Batch windows and message spectrograms
        windows = torch.stack([plan['window'] for plan in valid_plans], dim=0)  # [N, 1, T]
        M_specs = torch.stack([
            self.build_message_spec_from_window_plan(plan, base_symbol_amp) 
            for plan in valid_plans
        ], dim=0)  # [N, 1, 2, F, T]
        M_specs = M_specs.squeeze(1)  # [N, 2, F, T] - Remove extra dimension
        
        # Single batched encode/decode call
        windows_wm, _ = model.encode(windows, M_specs)  # [N, 1, T]
        windows_wm = torch.clamp(windows_wm, -1.0, 1.0)
        M_recs = model.decode(windows_wm)  # [N, 1, F, T]
        
        # Process each window individually for loss computation
        for i, plan in enumerate(valid_plans):
            window = plan['window']
            slots = plan['slots']
            bits = plan['bits']
            amp_per_slot = plan['amp_per_slot']
            
            # Extract decoded spectrogram for this window
            M_rec = M_recs[i:i+1]  # [1, 1, F, T]
            
            # Extract bits from slots
            rec_vals = torch.zeros(1, len(slots), device=window.device)
            for s, (f, t) in enumerate(slots):
                if f < M_rec.size(2) and t < M_rec.size(3):
                    rec_vals[0, s] = M_rec[0, 0, f, t]
            
            # Compute losses with bit mask
            target_bits = bits[:, :len(slots)]
            bit_mask = plan.get('bit_mask', torch.ones_like(target_bits, dtype=torch.bool))
            bit_mask = bit_mask[:, :len(slots)]  # Ensure mask matches slots length
            
            # Only compute loss for supervised bits
            if bit_mask.any():
                # Bit error loss (only for supervised bits)
                logits = rec_vals.clamp(-6.0, 6.0)
                ce_loss = F.binary_cross_entropy_with_logits(
                    logits[bit_mask], target_bits[bit_mask].float()
                )
                
                # MSE loss (only for supervised bits)
                target_symbols = (target_bits * 2 - 1).float() * base_symbol_amp
                if len(amp_per_slot) >= len(slots):
                    target_symbols = target_symbols * amp_per_slot[:len(slots)].unsqueeze(0)
                mse_loss = F.mse_loss(rec_vals[bit_mask], target_symbols[bit_mask])
            else:
                # No supervised bits in this window
                ce_loss = torch.tensor(0.0, device=window.device)
                mse_loss = torch.tensor(0.0, device=window.device)
            
            # Perceptual loss (simplified)
            perc_loss = F.mse_loss(window, window_wm)
            
            # Accumulate losses
            window_loss = (loss_weights.get('ce', 1.0) * ce_loss + 
                          loss_weights.get('mse', 0.25) * mse_loss + 
                          loss_weights.get('perceptual', 0.01) * perc_loss)
            
            supervised_slots = int(bit_mask.sum().item())
            if supervised_slots > 0:
                loss_terms.append(window_loss * supervised_slots)
                total_loss += float((window_loss * supervised_slots).detach().item())
                total_bits += supervised_slots
                total_errors += int((rec_vals[bit_mask] > 0).long().ne(target_bits[bit_mask]).sum().item())
            total_perceptual += perc_loss.item()
        
        # Average losses
        if total_bits > 0 and len(loss_terms) > 0:
            loss_tensor = torch.stack(loss_terms).sum() / total_bits
            avg_loss = float(loss_tensor.detach().item())
            ber = total_errors / total_bits
            avg_perceptual = total_perceptual / max(1, len(window_plans))
        else:
            loss_tensor = torch.tensor(0.0, requires_grad=True)
            avg_loss = 0.0
            ber = 0.0
            avg_perceptual = 0.0
        
        return {
            'loss_tensor': loss_tensor,
            'total_loss': avg_loss,
            'ber': ber,
            'perceptual_loss': avg_perceptual,
            'total_bits': total_bits,
            'total_errors': total_errors
        }
    
    def stft_micro(self, audio_window: torch.Tensor, overlap_ratio: float = None) -> torch.Tensor:
        """
        Compute STFT with parameters adapted to micro-window size.
        
        Args:
            audio_window: Audio window [1, T] or [B, 1, T]
            overlap_ratio: Overlap ratio (defaults to self.overlap_ratio)
            
        Returns:
            STFT coefficients [B, F, T] (complex)
        """
        if overlap_ratio is None:
            overlap_ratio = self.overlap_ratio
            
        T = audio_window.shape[-1]
        win_len = T
        n_fft = 1 << (win_len - 1).bit_length()  # next pow2, e.g., 330 -> 512
        hop = max(1, int(round(win_len * (1.0 - overlap_ratio))))  # 50% overlap -> win_len//2
        window = torch.hann_window(win_len, device=audio_window.device, dtype=audio_window.dtype)

        # collapse to [B,T]
        x = audio_window.squeeze(1) if audio_window.dim() == 3 else audio_window
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win_len,
                       window=window, center=True, pad_mode="reflect", return_complex=True)
        return X

    def get_training_info(self) -> Dict:
        """Get information about training configuration."""
        return {
            'window_ms': self.window_ms,
            'overlap_ratio': self.overlap_ratio,
            'sr': self.sr,
            'n_fft': self.n_fft,
            'hop': self.hop,
            'mapper_seed': self.mapper_seed,
            'sync_strength': self.sync_strength
        }


class EnhancedWindowLevelTrainer(WindowLevelTrainer):
    """
    Enhanced window-level trainer with symbol-level redundancy and fusion.
    Implements the new architecture with r=2-4 redundancy per symbol.
    """
    
    def __init__(self, sr: int = 22050, window_ms: int = 15, overlap_ratio: float = 0.5, 
                 mapper_seed: int = 42, n_fft: int = 1024, hop: int = 512,
                 sync_strength: float = 0.1, redundancy: int = 3, bias_lower_mid: bool = True,
                 use_crc: bool = True, symbol_length: int = 8, min_agreement_rate: float = 0.3):
        super().__init__(window_ms, overlap_ratio, sr, n_fft, hop, mapper_seed, sync_strength)
        
        # Replace simple mapper with enhanced mapper
        self.enhanced_mapper = EnhancedDeterministicMapper(
            seed=mapper_seed, n_fft=n_fft, hop=hop,
            max_freq_bins=513, max_time_frames=44,
            redundancy=redundancy, bias_lower_mid=bias_lower_mid
        )
        
        # Keep the old mapper for backward compatibility
        self.deterministic_mapper = DeterministicMapper(
            seed=mapper_seed, n_fft=n_fft, hop=hop, sr=sr, window_ms=window_ms
        )
        
        # CRC settings
        self.use_crc = use_crc
        self.symbol_length = symbol_length
        self.min_agreement_rate = min_agreement_rate
    
    def _choose_indices(self, num_bits: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
        """
        Fairly select k indices from num_bits using deterministic sampling.
        
        Args:
            num_bits: Total number of bits available
            k: Number of bits to select
            seed: Deterministic seed for reproducible selection
            device: Device for the output tensor
            
        Returns:
            Tensor of selected indices [k]
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randperm(num_bits, generator=generator, device=device)[:k]
    
    def build_symbol_level_plan(self, model: INNWatermarker, x_1s: torch.Tensor, 
                               n_symbols: int, base_symbol_amp: float) -> Dict:
        """
        Build symbol-level plan with redundancy across windows.
        
        Args:
            model: INNWatermarker model
            x_1s: 1-second audio segment [1, T]
            n_symbols: Number of symbols to embed
            base_symbol_amp: Base symbol amplitude
            
        Returns:
            Dict with symbol mappings and window plans
        """
        if x_1s.dim() != 2 or x_1s.size(0) != 1:
            raise ValueError(f"Expected x_1s shape [1, T], got {x_1s.shape}")
        
        # Add sync pattern
        x_1s_with_sync = self.sync_anchor.embed_sync_pattern(x_1s)
        
        # Create micro-chunks
        windows = self.micro_chunker.chunk_1s_segment(x_1s_with_sync)
        n_windows = len(windows)
        
        # Build symbol-to-windows mapping
        symbol_mappings = self.enhanced_mapper.build_symbol_mapping(n_windows, n_symbols)
        
        # Build window plans with symbol assignments
        window_plans = []
        for i, window in enumerate(windows):
            # Get symbols assigned to this window
            window_symbols = self.enhanced_mapper.get_window_symbols(i, symbol_mappings)
            
            # Create slots for this window (one per assigned symbol)
            slots = []
            for symbol_idx in window_symbols:
                # Get the band index for this symbol in this window
                for w_idx, band_idx in symbol_mappings[symbol_idx]:
                    if w_idx == i:
                        # Convert band_idx to (freq, time) slot
                        # For simplicity, use band_idx as freq and middle time frame
                        time_frame = self.enhanced_mapper.max_time_frames // 2
                        slots.append((band_idx, time_frame))
                        break
            
            # Apply psychoacoustic gating
            masked_slots = apply_psychoacoustic_gate(window, slots, model, self.n_fft, self.hop)
            
            # Create amplitude per slot (simple equal allocation)
            amp_per_slot = torch.ones(len(masked_slots), device=window.device)
            
            window_plan = {
                'window': window,
                'slots': masked_slots,
                'amp_per_slot': amp_per_slot,
                'n_slots': len(masked_slots),
                'window_idx': i,
                'window_symbols': window_symbols  # New: symbols assigned to this window
            }
            
            window_plans.append(window_plan)
        
        return {
            'window_plans': window_plans,
            'symbol_mappings': symbol_mappings,
            'n_windows': n_windows,
            'n_symbols': n_symbols,
            'original_1s_audio': x_1s
        }
    
    def build_symbol_level_labels(self, window_plans: List[Dict], 
                                 payload_bits: torch.Tensor) -> List[Dict]:
        """
        Build symbol-level labels that span multiple windows.
        
        Args:
            window_plans: List of window plans with symbol assignments
            payload_bits: [1, n_symbols] tensor with payload bits
            
        Returns:
            Updated window_plans with symbol-level labels
        """
        if payload_bits.dim() == 1:
            payload_bits = payload_bits.unsqueeze(0)
        
        n_symbols = payload_bits.size(1)
        
        for plan in window_plans:
            window_symbols = plan['window_symbols']
            n_slots = plan['n_slots']
            
            if n_slots == 0:
                plan['bits'] = torch.tensor([], dtype=torch.long, device=payload_bits.device).unsqueeze(0)
                plan['symbol_bits'] = torch.tensor([], dtype=torch.long, device=payload_bits.device).unsqueeze(0)
                continue
            
            # Create window-level bits with proper supervision masking
            num_bits = payload_bits.size(1)
            if n_slots >= num_bits:
                # Pad if we have more slots than payload bits
                pad = torch.zeros(1, n_slots - num_bits, dtype=payload_bits.dtype, device=payload_bits.device)
                bits = torch.cat([payload_bits, pad], dim=1)
                # Create supervision mask: True for real bits, False for padded zeros
                mask = torch.cat([
                    torch.ones(1, num_bits, dtype=torch.bool, device=payload_bits.device),
                    torch.zeros(1, n_slots - num_bits, dtype=torch.bool, device=payload_bits.device)
                ], dim=1)
            else:
                # Fairly select which bits to keep using deterministic sampling
                # Use window index and second index for deterministic seed
                window_idx = plan.get('window_idx', 0)
                second_idx = plan.get('second_idx', 0)
                seed = (second_idx * 1000 + window_idx) % (2**32)  # Deterministic seed
                keep_idx = self._choose_indices(num_bits, n_slots, seed, payload_bits.device)
                bits = payload_bits[:, keep_idx]
                # All selected bits should be supervised
                mask = torch.ones(1, n_slots, dtype=torch.bool, device=payload_bits.device)
            
            plan['bits'] = bits
            plan['bit_mask'] = mask
            
            # Create symbol-level bits (only for symbols assigned to this window)
            symbol_bits = []
            for symbol_idx in window_symbols:
                if symbol_idx < n_symbols:
                    symbol_bits.append(payload_bits[0, symbol_idx].item())
            
            plan['symbol_bits'] = torch.tensor(symbol_bits, dtype=torch.long, device=payload_bits.device).unsqueeze(0)
        
        return window_plans
    
    def compute_symbol_level_loss(self, model: INNWatermarker, window_plans: List[Dict],
                                 symbol_mappings: Dict, base_symbol_amp: float, 
                                 loss_weights: Dict) -> Dict:
        """
        Compute symbol-level loss with fusion across windows using batched model calls.
        
        Args:
            model: INNWatermarker model
            window_plans: List of window plans
            symbol_mappings: Symbol-to-windows mapping
            base_symbol_amp: Base symbol amplitude
            loss_weights: Loss weight dictionary
            
        Returns:
            Dict with symbol-level loss metrics
        """
        loss_terms: List[torch.Tensor] = []
        total_loss = 0.0
        total_symbols = 0
        total_errors = 0
        total_perceptual = 0.0
        
        # Filter out empty plans and prepare for batching
        valid_plans = [plan for plan in window_plans if plan['n_slots'] > 0]
        empty_plan_indices = [i for i, plan in enumerate(window_plans) if plan['n_slots'] == 0]
        
        # Collect window predictions for symbol fusion
        window_predictions = []
        
        # Handle empty plans first
        for i in empty_plan_indices:
            plan = window_plans[i]
            dev = plan.get('window').device if isinstance(plan.get('window'), torch.Tensor) else None
            window_predictions.append(torch.zeros(1, 0, device=dev) if dev is not None else torch.zeros(1, 0))
        
        if valid_plans:
            # Batch windows and message spectrograms
            windows = torch.stack([plan['window'] for plan in valid_plans], dim=0)  # [N, 1, T]
            M_specs = torch.stack([
                self.build_message_spec_from_window_plan(plan, base_symbol_amp) 
                for plan in valid_plans
            ], dim=0)  # [N, 1, 2, F, T]
            M_specs = M_specs.squeeze(1)  # [N, 2, F, T] - Remove extra dimension
            
            # Single batched encode/decode call
            windows_wm, _ = model.encode(windows, M_specs)  # [N, 1, T]
            windows_wm = torch.clamp(windows_wm, -1.0, 1.0)
            M_recs = model.decode(windows_wm)  # [N, 1, F, T]
            
            # Process each valid window
            valid_idx = 0
            for i, plan in enumerate(window_plans):
                if plan['n_slots'] == 0:
                    continue  # Already handled above
                
                window = plan['window']
                slots = plan['slots']
                symbol_bits = plan['symbol_bits']
                amp_per_slot = plan['amp_per_slot']
                
                # Extract decoded spectrogram and watermarked window for this window
                M_rec = M_recs[valid_idx:valid_idx+1]  # [1, 1, F, T]
                window_wm = windows_wm[valid_idx:valid_idx+1]  # [1, 1, T]
                valid_idx += 1
                
                # Extract predictions from slots
                rec_vals = torch.zeros(1, len(slots), device=window.device)
                for s, (f, t) in enumerate(slots):
                    if f < M_rec.size(2) and t < M_rec.size(3):
                        rec_vals[0, s] = M_rec[0, 0, f, t]
                
                window_predictions.append(rec_vals)
                
                # Compute window-level losses with bit mask
                if len(symbol_bits) > 0:
                    target_bits = symbol_bits[:, :len(slots)]
                    bit_mask = plan.get('bit_mask', torch.ones_like(target_bits, dtype=torch.bool))
                    bit_mask = bit_mask[:, :len(slots)]  # Ensure mask matches slots length
                
                # Only compute loss for supervised bits
                if bit_mask.any():
                    # Bit error loss (only for supervised bits)
                    logits = rec_vals.clamp(-6.0, 6.0)
                    ce_loss = F.binary_cross_entropy_with_logits(
                        logits[bit_mask], target_bits[bit_mask].float()
                    )
                    
                    # MSE loss (only for supervised bits)
                    target_symbols = (target_bits * 2 - 1).float() * base_symbol_amp
                    if len(amp_per_slot) >= len(slots):
                        target_symbols = target_symbols * amp_per_slot[:len(slots)].unsqueeze(0)
                    mse_loss = F.mse_loss(rec_vals[bit_mask], target_symbols[bit_mask])
                else:
                    # No supervised bits in this window
                    ce_loss = torch.tensor(0.0, device=window.device)
                    mse_loss = torch.tensor(0.0, device=window.device)
                
                # Perceptual loss
                perceptual_loss_output = self.perceptual_loss_fn(window.float(), window_wm.float())
                perceptual_loss = perceptual_loss_output["total_perceptual_loss"]
                
                # Window-level loss
                window_loss = (loss_weights.get('ce', 0.0) * ce_loss +
                             loss_weights.get('mse', 0.0) * mse_loss +
                             loss_weights.get('perceptual', 0.0) * perceptual_loss)
                
                total_loss += float(window_loss.detach().item())
                loss_terms.append(window_loss)
                total_perceptual += perceptual_loss.item()
                
                # BER calculation (only for supervised bits)
                if bit_mask.any():
                    pred_bits = (rec_vals > 0).long()
                    errors = (pred_bits[bit_mask] != target_bits[bit_mask]).float().sum().item()
                    total_errors += errors
                    total_symbols += bit_mask.sum().item()
        
        # Symbol-level fusion and loss
        if window_predictions and symbol_mappings:
            n_symbols = len(symbol_mappings)
            
            if self.use_crc:
                # Use CRC-enhanced fusion
                fused_symbols, confidence_scores = fuse_symbol_evidence_enhanced(
                    window_predictions, symbol_mappings, n_symbols,
                    self.symbol_length, self.use_crc, self.min_agreement_rate
                )
            else:
                # Use simple fusion
                fused_symbols = fuse_symbol_evidence(window_predictions, symbol_mappings, n_symbols)
                confidence_scores = torch.ones_like(fused_symbols)
            
            # Create target symbols from payload
            # This is a simplified version - in practice, you'd need to map payload bits to symbols
            target_symbols = torch.zeros_like(fused_symbols)
            for i in range(min(n_symbols, fused_symbols.size(1))):
                target_symbols[0, i] = 1.0  # Placeholder - should be actual payload bits
            
            # Symbol-level loss
            symbol_ce_loss = F.binary_cross_entropy_with_logits(fused_symbols, target_symbols)
            symbol_mse_loss = F.mse_loss(fused_symbols, target_symbols)
            
            # CRC penalty
            crc_penalty = 0.0
            if self.use_crc:
                crc_penalty = compute_crc_penalty(fused_symbols, target_symbols, self.symbol_length)
            
            # Add symbol-level loss to total
            symbol_loss = (loss_weights.get('ce', 0.0) * symbol_ce_loss +
                          loss_weights.get('mse', 0.0) * symbol_mse_loss +
                          loss_weights.get('crc', 0.1) * crc_penalty)
            total_loss += float(symbol_loss.detach().item())
            loss_terms.append(symbol_loss)
        
        ber = total_errors / total_symbols if total_symbols > 0 else 0.0
        
        # Produce a grad-capable scalar for the caller to backprop through
        if len(loss_terms) > 0:
            denom = max(1, total_symbols)
            loss_tensor = torch.stack([t for t in loss_terms]).sum() / denom
        else:
            first_param = next(iter(model.parameters()), None)
            device = first_param.device if first_param is not None else torch.device('cpu')
            loss_tensor = torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'loss_tensor': loss_tensor,
            'total_loss': float(total_loss),
            'ber': float(ber),
            'total_bits': int(total_symbols),  # match window-level API expected by callers
            'total_symbols': int(total_symbols),
            'total_errors': int(total_errors),
            'perceptual_loss': float(total_perceptual)
        }
