#!/usr/bin/env python3
"""
Main training script for SoundSafe watermarking - Imperceptibility Focus
GPU-optimized training with comprehensive loss functions
Standalone script with built-in setup and validation
"""

import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
import json
import random
import multiprocessing as mp

# Set multiprocessing start method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchaudio")


# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("⚠️  TensorBoard not available - logging will be limited")
    TENSORBOARD_AVAILABLE = False
    # Create a dummy SummaryWriter class
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def add_audio(self, *args, **kwargs):
            pass
        def close(self):
            pass

# Add current directory to path
sys.path.insert(0, '.')

from models.inn_encoder_decoder import INNWatermarker
from pipeline.ingest_and_chunk import EULDriver
from phm.perceptual_frontend import PerceptualFrontend
from phm.technical_frontend import TechnicalFrontend
from phm.fusion_head import FusionHead
from phm.runtime import run_phm_eul
from phm.telemetry import PerceptualInput, TechnicalTelemetry

from training.dataset import create_dataloaders
from training.losses import ImperceptibilityLoss, MessageLoss, AdversarialLoss
from training.utils import (
    TrainingMetrics, ModelCheckpointer, AudioMetrics, 
    LearningRateScheduler, setup_device, print_model_info,
    create_experiment_dir, save_training_config
)

def setup_training_environment(data_dir: str, create_samples: bool = False, num_samples: int = 20):
    """
    Setup training environment with directory creation and validation
    """
    print("🚀 Setting up training environment...")
    
    # Create necessary directories
    base_path = Path('.')
    dirs_to_create = [
        'data/train',
        'data/val', 
        'experiments',
        'checkpoints',
        'logs',
        'samples'
    ]
    
    for dir_path in dirs_to_create:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {full_path}")
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available - training will be slower on CPU")
    
    # Check required packages
    required_packages = ['torch', 'torchaudio', 'numpy', 'reedsolo']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not installed")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Validate data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory does not exist: {data_path}")
        if create_samples:
            print("Creating sample data...")
            create_sample_audio_data(data_dir, num_samples)
        else:
            print("Use --create_samples to generate sample data")
            return False
    
    # Check for audio files
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Missing train/ or val/ subdirectories in {data_path}")
        return False
    
    # Count audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    train_files = []
    val_files = []
    
    for ext in audio_extensions:
        train_files.extend(train_dir.glob(f"**/*{ext}"))
        val_files.extend(val_dir.glob(f"**/*{ext}"))
    
    print(f"\n📊 Dataset validation:")
    print(f"   Training files: {len(train_files)}")
    print(f"   Validation files: {len(val_files)}")
    
    if len(train_files) == 0:
        print("❌ No training audio files found!")
        if create_samples:
            print("Creating sample data...")
            create_sample_audio_data(data_dir, num_samples)
        else:
            return False
    
    if len(val_files) == 0:
        print("❌ No validation audio files found!")
        if create_samples:
            print("Creating sample data...")
            create_sample_audio_data(data_dir, num_samples)
        else:
            return False
    
    print("✅ Environment setup completed successfully!")
    return True

def create_sample_audio_data(data_dir: str, num_samples: int = 20):
    """
    Create sample audio data for testing
    """
    print(f"Creating {num_samples} sample audio files...")
    
    try:
        import torchaudio
    except ImportError:
        print("❌ torchaudio not available for creating sample data")
        return False
    
    data_path = Path(data_dir)
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample audio files
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    samples = int(sample_rate * duration)
    
    for split, split_dir in [('train', train_dir), ('val', val_dir)]:
        for i in range(num_samples):
            # Generate random audio with some structure
            audio = torch.randn(samples) * 0.1  # Low amplitude
            
            # Add sine wave for structure
            t = torch.linspace(0, duration, samples)
            freq = 440 + i * 50  # Different frequency for each sample
            audio += 0.05 * torch.sin(2 * torch.pi * freq * t)
            
            # Add some noise
            audio += torch.randn(samples) * 0.01
            
            # Save as WAV
            filename = split_dir / f"sample_{i:03d}.wav"
            torchaudio.save(str(filename), audio.unsqueeze(0), sample_rate)
        
        print(f"✓ Created {num_samples} samples in {split_dir}")
    
    return True

class WatermarkingTrainer:
    """Main trainer class for watermarking system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device, self.is_gpu = setup_device()
        
        # Create experiment directory
        self.exp_dir = create_experiment_dir(
            config['experiment_dir'], 
            config['experiment_name']
        )
        
        # Save configuration
        save_training_config(config, self.exp_dir)
        
        # Setup logging
        self.writer = SummaryWriter(self.exp_dir / "logs")
        
        # Initialize models
        self._setup_models()
        
        # Initialize losses
        self._setup_losses()
        
        # Initialize optimizers
        self._setup_optimizers()
        
        # Initialize metrics and checkpointer
        self.metrics = TrainingMetrics()
        self.audio_metrics = AudioMetrics(self.device)
        self.checkpointer = ModelCheckpointer(
            str(self.exp_dir / "checkpoints"), 
            self.device
        )
        
        print(f"Experiment directory: {self.exp_dir}")
        print_model_info(self.inn_model, "INN Watermarker")
        print_model_info(self.phm_perc, "Perceptual Frontend")
        print_model_info(self.phm_tech, "Technical Frontend")
        print_model_info(self.phm_fusion, "Fusion Head")
    
    def _setup_models(self):
        """Initialize all models"""
        # INN Watermarker
        self.inn_model = INNWatermarker(
            n_blocks=self.config['inn_blocks'],
            spec_channels=2
        ).to(self.device)
        
        # EUL Driver
        self.eul_driver = EULDriver(
            sr=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop=self.config['hop_length']
        )
        
        # PHM Components
        self.phm_perc = PerceptualFrontend(
            sample_rate=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            emb_dim=self.config['phm_emb_dim']
        ).to(self.device)
        
        self.phm_tech = TechnicalFrontend(
            feat_dim=5,  # From telemetry
            emb_dim=self.config['phm_emb_dim']
        ).to(self.device)
        
        self.phm_fusion = FusionHead(
            d_perc=self.config['phm_emb_dim'],
            d_tech=self.config['phm_emb_dim'],
            d_model=self.config['phm_emb_dim']
        ).to(self.device)
        
        # Set models to training mode
        self.inn_model.train()
        self.phm_perc.train()
        self.phm_tech.train()
        self.phm_fusion.train()
    
    def _setup_losses(self):
        """Initialize loss functions"""
        self.impercept_loss = ImperceptibilityLoss(
            sample_rate=self.config['sample_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            device=self.device
        )
        
        self.message_loss = MessageLoss(device=self.device)
        self.adv_loss = AdversarialLoss(device=self.device)
    
    def _setup_optimizers(self):
        """Initialize optimizers"""
        # INN optimizer
        self.inn_optimizer = optim.AdamW(
            self.inn_model.parameters(),
            lr=self.config['inn_lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # PHM optimizer
        phm_params = list(self.phm_perc.parameters()) + \
                    list(self.phm_tech.parameters()) + \
                    list(self.phm_fusion.parameters())
        
        self.phm_optimizer = optim.Adam(
            phm_params,
            lr=self.config['phm_lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate schedulers
        self.inn_scheduler = LearningRateScheduler(
            self.inn_optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            decay_epochs=self.config['decay_epochs']
        )
        
        self.phm_scheduler = LearningRateScheduler(
            self.phm_optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            decay_epochs=self.config['decay_epochs']
        )
    
    def _generate_message_spec(self, payload_bits: torch.Tensor, audio_shape: torch.Tensor) -> torch.Tensor:
        """Generate message spectrogram from payload bits"""
        batch_size, _, freq_bins, time_frames = audio_shape
        
        # Create message spectrogram
        message_spec = torch.zeros(batch_size, 2, freq_bins, time_frames, device=self.device)
        
        # Simple BPSK encoding on real channel
        for b in range(batch_size):
            for i, bit in enumerate(payload_bits[b]):
                if i >= freq_bins * time_frames:
                    break
                f = i % freq_bins
                t = i // freq_bins
                if t < time_frames:
                    # Convert bit to amplitude
                    amplitude = (bit * 2 - 1) * 0.1  # -0.1 or +0.1
                    message_spec[b, 0, f, t] = amplitude
        
        return message_spec
    
    def _compute_phm_loss(self, watermarked_audio: torch.Tensor) -> dict:
        """Compute PHM assessment loss"""
        # Generate dummy telemetry for training
        batch_size = watermarked_audio.size(0)
        tech_telemetry = TechnicalTelemetry(
            softbit_conf=torch.randn(batch_size, 43, 1, device=self.device),
            slot_snr=torch.randn(batch_size, 43, 1, device=self.device),
            slot_fill=torch.randn(batch_size, 43, 1, device=self.device),
            sync_drift=torch.randn(batch_size, 43, 1, device=self.device),
            rs_errata=torch.randn(batch_size, 43, 1, device=self.device),
            rs_success=torch.ones(batch_size, 1, device=self.device)
        )
        
        # Run PHM
        perc_input = PerceptualInput(audio_1s=watermarked_audio)
        phm_output = run_phm_eul(
            self.phm_perc, self.phm_tech, self.phm_fusion,
            perc_input, tech_telemetry
        )
        
        # PHM should detect watermarks (presence_p should be high)
        target_presence = torch.ones_like(phm_output.presence_p)
        presence_loss = nn.BCELoss()(phm_output.presence_p, target_presence)
        
        return {
            'phm_presence': presence_loss,
            'phm_reliability': phm_output.decode_reliability.mean(),
            'phm_artifact_risk': phm_output.artifact_risk.mean()
        }
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch"""
        self.inn_model.train()
        self.phm_perc.train()
        self.phm_tech.train()
        self.phm_fusion.train()
        
        self.metrics.reset()
        
        for batch_idx, (audio, payload_bits) in enumerate(train_loader):
            # Move data to device
            audio = audio.to(self.device, non_blocking=True)
            payload_bits = payload_bits.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.inn_optimizer.zero_grad()
            self.phm_optimizer.zero_grad()
            
            # Forward pass through INN
            # Get STFT of audio
            audio_spec = self.inn_model.stft(audio)
            
            # Generate message spectrogram
            message_spec = self._generate_message_spec(payload_bits, audio_spec.shape)
            
            # Encode watermark
            watermarked_spec, _ = self.inn_model.encode_spec(audio_spec, message_spec)
            
            # Convert back to audio
            watermarked_audio = self.inn_model.istft(watermarked_spec)
            
            # Compute losses
            losses = {}
            
            # Imperceptibility loss
            impercept_losses = self.impercept_loss(audio, watermarked_audio)
            losses.update(impercept_losses)
            
            # Message recovery loss (simplified)
            # For now, we'll use a placeholder - in full training, you'd decode and compare
            message_losses = self.message_loss(
                torch.randn_like(payload_bits),  # Placeholder prediction
                payload_bits
            )
            losses.update(message_losses)
            
            # PHM loss
            phm_losses = self._compute_phm_loss(watermarked_audio)
            losses.update(phm_losses)
            
            # Total loss
            total_loss = (
                losses['total'] * self.config['impercept_weight'] +
                losses['message_total'] * self.config['message_weight'] +
                losses['phm_presence'] * self.config['phm_weight']
            )
            losses['total_combined'] = total_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.inn_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(
                list(self.phm_perc.parameters()) + 
                list(self.phm_tech.parameters()) + 
                list(self.phm_fusion.parameters()), 
                max_norm=1.0
            )
            
            # Update parameters
            self.inn_optimizer.step()
            self.phm_optimizer.step()
            
            # Update metrics
            self.metrics.update(losses)
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                elapsed = self.metrics.get_elapsed_time()
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {total_loss.item():.4f}, Time: {elapsed:.2f}s')
        
        # Update learning rates
        self.inn_scheduler.step(epoch)
        self.phm_scheduler.step(epoch)
        
        # Log epoch metrics
        epoch_metrics = self.metrics.get_averages()
        for key, value in epoch_metrics.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        return epoch_metrics
    
    def validate(self, val_loader, epoch: int):
        """Validate the model"""
        self.inn_model.eval()
        self.phm_perc.eval()
        self.phm_tech.eval()
        self.phm_fusion.eval()
        
        val_metrics = TrainingMetrics()
        
        with torch.no_grad():
            for audio, payload_bits in val_loader:
                audio = audio.to(self.device, non_blocking=True)
                payload_bits = payload_bits.to(self.device, non_blocking=True)
                
                # Forward pass
                audio_spec = self.inn_model.stft(audio)
                message_spec = self._generate_message_spec(payload_bits, audio_spec.shape)
                watermarked_spec, _ = self.inn_model.encode_spec(audio_spec, message_spec)
                watermarked_audio = self.inn_model.istft(watermarked_spec)
                
                # Compute losses
                impercept_losses = self.impercept_loss(audio, watermarked_audio)
                val_metrics.update(impercept_losses)
                
                # Compute audio quality metrics
                audio_scores = self.audio_metrics.compute_metrics(audio, watermarked_audio)
                val_metrics.update({f'val_{k}': torch.tensor(v) for k, v in audio_scores.items()})
        
        # Log validation metrics
        val_avg = val_metrics.get_averages()
        for key, value in val_avg.items():
            self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        return val_avg
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        start_epoch = 0
        
        # Load checkpoint if resuming
        if self.config['resume']:
            start_epoch = self.checkpointer.load_checkpoint(
                self.inn_model, self.inn_optimizer, self.inn_scheduler
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            print('-' * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader, epoch)
            
            # Print epoch summary
            print(f'Train Loss: {train_metrics.get("total_combined", 0):.4f}')
            print(f'Val Loss: {val_metrics.get("total", 0):.4f}')
            print(f'Val PSNR: {val_metrics.get("val_psnr", 0):.2f}')
            print(f'Val SSIM: {val_metrics.get("val_ssim", 0):.4f}')
            
            # Save checkpoint
            is_best = val_metrics.get('total', float('inf')) < best_val_loss
            if is_best:
                best_val_loss = val_metrics.get('total', float('inf'))
            
            checkpoint_path = self.checkpointer.save_checkpoint(
                epoch, self.inn_model, self.inn_optimizer, 
                self.inn_scheduler, self.metrics, is_best
            )
            
            if is_best:
                print(f'New best model saved: {checkpoint_path}')
        
        print('\nTraining completed!')
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train SoundSafe Watermarking - Imperceptibility')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to dataset directory (should contain train/ and val/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--create_samples', action='store_true',
                       help='Create sample audio files if data directory is empty')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of sample files to create (if --create_samples)')
    
    # Model arguments
    parser.add_argument('--inn_blocks', type=int, default=8,
                       help='Number of INN blocks')
    parser.add_argument('--phm_emb_dim', type=int, default=192,
                       help='PHM embedding dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--inn_lr', type=float, default=1e-4,
                       help='Learning rate for INN model')
    parser.add_argument('--phm_lr', type=float, default=5e-4,
                       help='Learning rate for PHM components')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Warmup epochs for learning rate')
    parser.add_argument('--decay_epochs', type=int, default=50,
                       help='Decay epochs for learning rate')
    
    # Loss weights
    parser.add_argument('--impercept_weight', type=float, default=1.0,
                       help='Weight for imperceptibility loss')
    parser.add_argument('--message_weight', type=float, default=0.5,
                       help='Weight for message loss')
    parser.add_argument('--phm_weight', type=float, default=0.3,
                       help='Weight for PHM loss')
    
    # Other arguments
    parser.add_argument('--experiment_dir', type=str, default='./experiments',
                       help='Directory to save experiments')
    parser.add_argument('--experiment_name', type=str, default='imperceptibility_training',
                       help='Name of the experiment')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Log interval for training progress')
    
    args = parser.parse_args()
    
    # Setup training environment
    print("=" * 60)
    print("SoundSafe Watermarking Training - Imperceptibility Focus")
    print("=" * 60)
    
    if not setup_training_environment(args.data_dir, args.create_samples, args.num_samples):
        print("\n❌ Setup failed. Please check the errors above and try again.")
        return
    
    # Create config
    config = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'inn_blocks': args.inn_blocks,
        'phm_emb_dim': args.phm_emb_dim,
        'epochs': args.epochs,
        'inn_lr': args.inn_lr,
        'phm_lr': args.phm_lr,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'decay_epochs': args.decay_epochs,
        'impercept_weight': args.impercept_weight,
        'message_weight': args.message_weight,
        'phm_weight': args.phm_weight,
        'experiment_dir': args.experiment_dir,
        'experiment_name': args.experiment_name,
        'resume': args.resume,
        'log_interval': args.log_interval,
        'sample_rate': 22050,
        'n_fft': 1024,
        'hop_length': 512,
        'duration': 1.0,
        'payload_size': 125
    }
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    
    # Adjust num_workers based on CUDA availability
    num_workers = config['num_workers']
    if torch.cuda.is_available() and num_workers > 0:
        # Use multiprocessing with CUDA
        print(f"Using {num_workers} workers with CUDA multiprocessing")
    else:
        # Fallback to single-threaded loading
        num_workers = 0
        print("Using single-threaded data loading")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=num_workers,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        sample_rate=config['sample_rate'],
        duration=config['duration'],
        payload_size=config['payload_size']
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create trainer
    print("\n🏗️  Initializing trainer...")
    trainer = WatermarkingTrainer(config)
    
    # Start training
    print("\n🚀 Starting training...")
    print("=" * 60)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
