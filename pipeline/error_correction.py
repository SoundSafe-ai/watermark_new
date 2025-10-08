# soundsafe/pipeline/error_correction.py
# Enhanced error correction with inner CRC per symbol
# Implements CRC-8 validation and weighted soft-combining

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


def compute_crc8(data: bytes) -> int:
    """
    Compute CRC-8 checksum using polynomial 0x07 (CRC-8-CCITT).
    
    Args:
        data: Input bytes
        
    Returns:
        CRC-8 checksum (0-255)
    """
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return crc


def bits_to_bytes(bits: torch.Tensor) -> bytes:
    """
    Convert bit tensor to bytes.
    
    Args:
        bits: [1, N] tensor with bits (0 or 1)
        
    Returns:
        Bytes representation
    """
    if bits.numel() == 0:
        return b''
    
    bits_flat = bits.flatten()
    # Pad to multiple of 8 if needed
    if len(bits_flat) % 8 != 0:
        padding = 8 - (len(bits_flat) % 8)
        bits_flat = F.pad(bits_flat, (0, padding))
    
    # Convert to bytes
    bytes_list = []
    for i in range(0, len(bits_flat), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits_flat):
                byte_val |= (int(bits_flat[i + j]) & 1) << j
        bytes_list.append(byte_val)
    
    return bytes(bytes_list)


def bytes_to_bits(data: bytes, target_length: int) -> torch.Tensor:
    """
    Convert bytes to bit tensor.
    
    Args:
        data: Input bytes
        target_length: Target number of bits
        
    Returns:
        [1, target_length] tensor with bits
    """
    bits = []
    for byte_val in data:
        for j in range(8):
            bits.append((byte_val >> j) & 1)
    
    # Pad or truncate to target length
    while len(bits) < target_length:
        bits.append(0)
    
    return torch.tensor(bits[:target_length], dtype=torch.long).unsqueeze(0)


def add_crc_to_symbol(symbol_bits: torch.Tensor, symbol_length: int = 8) -> torch.Tensor:
    """
    Add CRC-8 to a symbol (8 bits + 8 CRC = 16 bits total).
    
    Args:
        symbol_bits: [1, symbol_length] tensor with symbol bits
        symbol_length: Length of symbol in bits (default 8)
        
    Returns:
        [1, symbol_length + 8] tensor with symbol + CRC bits
    """
    if symbol_bits.numel() == 0:
        return torch.zeros(1, symbol_length + 8, dtype=torch.long)
    
    # Ensure we have the right length
    if symbol_bits.size(1) > symbol_length:
        symbol_bits = symbol_bits[:, :symbol_length]
    elif symbol_bits.size(1) < symbol_length:
        pad = torch.zeros(1, symbol_length - symbol_bits.size(1), dtype=torch.long, device=symbol_bits.device)
        symbol_bits = torch.cat([symbol_bits, pad], dim=1)
    
    # Convert to bytes and compute CRC
    symbol_bytes = bits_to_bytes(symbol_bits)
    crc_val = compute_crc8(symbol_bytes)
    
    # Convert CRC to bits
    crc_bits = []
    for j in range(8):
        crc_bits.append((crc_val >> j) & 1)
    
    crc_tensor = torch.tensor(crc_bits, dtype=torch.long, device=symbol_bits.device).unsqueeze(0)
    
    # Combine symbol + CRC
    return torch.cat([symbol_bits, crc_tensor], dim=1)


def validate_symbol_crc(symbol_with_crc: torch.Tensor, symbol_length: int = 8) -> Tuple[bool, torch.Tensor]:
    """
    Validate CRC for a symbol with CRC.
    
    Args:
        symbol_with_crc: [1, symbol_length + 8] tensor with symbol + CRC bits
        symbol_length: Length of symbol in bits (default 8)
        
    Returns:
        (is_valid, symbol_bits) tuple
    """
    if symbol_with_crc.size(1) < symbol_length + 8:
        return False, torch.zeros(1, symbol_length, dtype=torch.long, device=symbol_with_crc.device)
    
    # Split symbol and CRC
    symbol_bits = symbol_with_crc[:, :symbol_length]
    crc_bits = symbol_with_crc[:, symbol_length:symbol_length + 8]
    
    # Compute expected CRC
    symbol_bytes = bits_to_bytes(symbol_bits)
    expected_crc = compute_crc8(symbol_bytes)
    
    # Convert received CRC to int
    received_crc = 0
    for j in range(8):
        received_crc |= (int(crc_bits[0, j]) & 1) << j
    
    is_valid = (expected_crc == received_crc)
    return is_valid, symbol_bits


def compute_crc_agreement_rate(window_predictions: List[torch.Tensor],
                              symbol_mappings: Dict[int, List[Tuple[int, int]]],
                              symbol_length: int = 8) -> torch.Tensor:
    """
    Calculate CRC agreement rate across windows for each symbol.
    
    Args:
        window_predictions: List of [1, S_window] tensors for each window
        symbol_mappings: Dict mapping symbol_idx -> List of (window_idx, band_idx) tuples
        symbol_length: Length of each symbol in bits
        
    Returns:
        [1, n_symbols] tensor with CRC agreement rates (0.0 to 1.0)
    """
    n_symbols = len(symbol_mappings)
    device = window_predictions[0].device
    agreement_rates = torch.zeros(1, n_symbols, device=device)
    
    for symbol_idx, placements in symbol_mappings.items():
        if symbol_idx >= n_symbols:
            continue
        
        valid_crcs = 0
        total_windows = 0
        
        for window_idx, band_idx in placements:
            if window_idx < len(window_predictions):
                window_pred = window_predictions[window_idx]
                if band_idx < window_pred.size(1):
                    # Extract symbol + CRC bits (assuming 16 bits total: 8 symbol + 8 CRC)
                    symbol_with_crc = window_pred[:, band_idx:band_idx + symbol_length + 8]
                    
                    # Validate CRC
                    is_valid, _ = validate_symbol_crc(symbol_with_crc, symbol_length)
                    if is_valid:
                        valid_crcs += 1
                    total_windows += 1
        
        if total_windows > 0:
            agreement_rates[0, symbol_idx] = valid_crcs / total_windows
    
    return agreement_rates


def fuse_symbol_evidence_with_crc(window_predictions: List[torch.Tensor], 
                                 symbol_mappings: Dict[int, List[Tuple[int, int]]],
                                 n_symbols: int, symbol_length: int = 8,
                                 min_agreement_rate: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced fusion with CRC-based weighting and outlier rejection.
    
    Args:
        window_predictions: List of [1, S_window] tensors for each window
        symbol_mappings: Dict mapping symbol_idx -> List of (window_idx, band_idx) tuples
        n_symbols: Total number of symbols
        symbol_length: Length of each symbol in bits
        min_agreement_rate: Minimum CRC agreement rate to accept a symbol
        
    Returns:
        (fused_symbols, confidence_scores) tuple
    """
    device = window_predictions[0].device
    fused_symbols = torch.zeros(1, n_symbols, device=device)
    confidence_scores = torch.zeros(1, n_symbols, device=device)
    
    # Compute CRC agreement rates
    agreement_rates = compute_crc_agreement_rate(window_predictions, symbol_mappings, symbol_length)
    
    for symbol_idx, placements in symbol_mappings.items():
        if symbol_idx >= n_symbols:
            continue
        
        # Check if symbol meets minimum agreement rate
        if agreement_rates[0, symbol_idx] < min_agreement_rate:
            # Reject symbol as outlier
            fused_symbols[0, symbol_idx] = 0.0
            confidence_scores[0, symbol_idx] = 0.0
            continue
        
        # Collect valid predictions from windows containing this symbol
        symbol_evidence = []
        crc_weights = []
        
        for window_idx, band_idx in placements:
            if window_idx < len(window_predictions):
                window_pred = window_predictions[window_idx]
                if band_idx < window_pred.size(1):
                    # Extract symbol + CRC bits
                    symbol_with_crc = window_pred[:, band_idx:band_idx + symbol_length + 8]
                    
                    # Validate CRC and extract symbol
                    is_valid, symbol_bits = validate_symbol_crc(symbol_with_crc, symbol_length)
                    
                    if is_valid:
                        # Use symbol bits for fusion
                        symbol_value = symbol_bits.float().mean().item()  # Convert to scalar
                        symbol_evidence.append(symbol_value)
                        crc_weights.append(1.0)  # Full weight for valid CRC
                    else:
                        # Reduced weight for invalid CRC
                        symbol_value = symbol_with_crc[:, :symbol_length].float().mean().item()
                        symbol_evidence.append(symbol_value)
                        crc_weights.append(0.1)  # Reduced weight
        
        if symbol_evidence:
            # Weighted average based on CRC validity
            weights = torch.tensor(crc_weights, device=device)
            evidence = torch.tensor(symbol_evidence, device=device)
            
            # Normalize weights
            weights = weights / (weights.sum() + 1e-6)
            
            # Weighted fusion
            fused_value = (evidence * weights).sum().item()
            confidence = agreement_rates[0, symbol_idx].item()
            
            fused_symbols[0, symbol_idx] = fused_value
            confidence_scores[0, symbol_idx] = confidence
        else:
            # No valid evidence
            fused_symbols[0, symbol_idx] = 0.0
            confidence_scores[0, symbol_idx] = 0.0
    
    return fused_symbols, confidence_scores


def compute_crc_penalty(symbol_predictions: torch.Tensor, 
                        target_symbols: torch.Tensor,
                        symbol_length: int = 8) -> torch.Tensor:
    """
    Compute CRC penalty for training loss.
    
    Args:
        symbol_predictions: [1, n_symbols] tensor with predicted symbol values
        target_symbols: [1, n_symbols] tensor with target symbol values
        symbol_length: Length of each symbol in bits
        
    Returns:
        CRC penalty tensor
    """
    if symbol_predictions.numel() == 0:
        return torch.tensor(0.0, device=symbol_predictions.device)
    
    # Convert predictions to bits (threshold at 0.5)
    pred_bits = (symbol_predictions > 0.5).long()
    target_bits = (target_symbols > 0.5).long()
    
    # Add CRC to both predicted and target symbols
    pred_with_crc = add_crc_to_symbol(pred_bits, symbol_length)
    target_with_crc = add_crc_to_symbol(target_bits, symbol_length)
    
    # Compute CRC validation for predictions
    crc_penalty = 0.0
    for i in range(symbol_predictions.size(1)):
        pred_symbol = pred_with_crc[:, i * (symbol_length + 8):(i + 1) * (symbol_length + 8)]
        target_symbol = target_with_crc[:, i * (symbol_length + 8):(i + 1) * (symbol_length + 8)]
        
        # Check if predicted symbol has valid CRC
        pred_valid, _ = validate_symbol_crc(pred_symbol, symbol_length)
        target_valid, _ = validate_symbol_crc(target_symbol, symbol_length)
        
        # Add penalty if prediction CRC is invalid but target is valid
        if not pred_valid and target_valid:
            crc_penalty += 1.0
    
    return torch.tensor(crc_penalty, device=symbol_predictions.device)
