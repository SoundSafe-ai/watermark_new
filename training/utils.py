#!/usr/bin/env python3
"""
Training utilities for SoundSafe watermarking
GPU-optimized helper functions and metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path
import json

class TrainingMetrics:
    """Track training metrics and losses"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.losses = {}
        self.counts = {}
        self.start_time = time.time()
    
    def update(self, losses: Dict[str, torch.Tensor]):
        """Update metrics with new loss values"""
        for key, value in losses.items():
            if key not in self.losses:
                self.losses[key] = 0.0
                self.counts[key] = 0
            
            self.losses[key] += value.item()
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics"""
        averages = {}
        for key in self.losses:
            if self.counts[key] > 0:
                averages[key] = self.losses[key] / self.counts[key]
        return averages
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time

class ModelCheckpointer:
    """Handle model checkpointing and loading"""
    
    def __init__(self, checkpoint_dir: str, device: str = 'cuda'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.best_loss = float('inf')
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: TrainingMetrics,
        is_best: bool = False
    ) -> str:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics.get_averages(),
            'best_loss': self.best_loss
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_loss = metrics.get_averages().get('total', float('inf'))
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[str] = None
    ) -> int:
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Load best model
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            if not checkpoint_path.exists():
                return 0
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Update best loss
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        return checkpoint.get('epoch', 0)

class AudioMetrics:
    """Compute audio quality metrics"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def psnr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Peak Signal-to-Noise Ratio"""
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'), device=self.device)
        max_val = torch.max(original)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr
    
    def ssim(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index (simplified)"""
        # Simplified SSIM computation
        mu1 = torch.mean(original)
        mu2 = torch.mean(reconstructed)
        sigma1 = torch.var(original)
        sigma2 = torch.var(reconstructed)
        sigma12 = torch.mean((original - mu1) * (reconstructed - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive audio metrics"""
        with torch.no_grad():
            psnr_val = self.psnr(original, reconstructed).item()
            ssim_val = self.ssim(original, reconstructed).item()
            
            # L1 and L2 distances
            l1_dist = torch.mean(torch.abs(original - reconstructed)).item()
            l2_dist = torch.sqrt(torch.mean((original - reconstructed) ** 2)).item()
            
            return {
                'psnr': psnr_val,
                'ssim': ssim_val,
                'l1_distance': l1_dist,
                'l2_distance': l2_dist
            }

class LearningRateScheduler:
    """Custom learning rate scheduler"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 10,
        decay_epochs: int = 50,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch: int):
        """Update learning rate"""
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        elif epoch < self.decay_epochs:
            # Decay phase
            progress = (epoch - self.warmup_epochs) / (self.decay_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 - progress) + self.min_lr * progress
        else:
            # Minimum learning rate
            lr = self.min_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def setup_device() -> Tuple[torch.device, bool]:
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device, True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
        return device, True
    else:
        device = torch.device('cpu')
        print("Using CPU")
        return device, False

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model: nn.Module, name: str = "Model"):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"\n{name} Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB (float32)")

def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "samples").mkdir(exist_ok=True)
    
    return exp_dir

def save_training_config(config: Dict, save_path: Path):
    """Save training configuration to JSON"""
    config_path = save_path / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to: {config_path}")

# Test utilities
if __name__ == "__main__":
    print("Testing training utilities...")
    
    # Test device setup
    device, is_gpu = setup_device()
    print(f"Device: {device}, GPU: {is_gpu}")
    
    # Test metrics
    metrics = TrainingMetrics()
    test_losses = {'loss1': torch.tensor(1.0), 'loss2': torch.tensor(2.0)}
    metrics.update(test_losses)
    print(f"Metrics: {metrics.get_averages()}")
    
    # Test audio metrics
    audio_metrics = AudioMetrics(device=device)
    original = torch.randn(1, 1, 1000, device=device)
    reconstructed = original + torch.randn_like(original) * 0.1
    audio_scores = audio_metrics.compute_metrics(original, reconstructed)
    print(f"Audio metrics: {audio_scores}")
    
    print("✓ Utilities test passed!")
