#!/usr/bin/env python3
"""
GPU-optimized dataset loader for SoundSafe watermarking training
Focuses on imperceptibility training with audio augmentation
"""

import os
import random
import torch
import warnings
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import numpy as np
from pathlib import Path

# Suppress TorchAudio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchaudio")


class AudioWatermarkDataset(Dataset):
    """
    GPU-optimized dataset for audio watermarking training
    All audio processing happens on GPU for maximum performance
    """
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 22050,
        duration: float = 1.0,  # 1 second segments
        payload_size: int = 125,  # bytes for RS(167,125)
        device: str = 'cuda',
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.payload_size = payload_size
        self.device = device
        self.augment = augment
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        print(f"Found {len(self.audio_files)} audio files")
        
        # Pre-compute segment count per file
        self.segments_per_file = int(duration * sample_rate)
        self.total_segments = len(self.audio_files) * max(1, int(10 / duration))  # ~10 seconds per file
        
        # GPU-optimized transforms
        self._setup_transforms()
        
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in the dataset directory"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.data_dir.glob(f"**/*{ext}"))
        
        return sorted(audio_files)
    
    def _setup_transforms(self):
        """Setup GPU-optimized audio transforms"""
        # All transforms will be applied on GPU
        self.resample = T.Resample(orig_freq=22050, new_freq=self.sample_rate).to(self.device)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        ).to(self.device)
        
    def _load_audio_cpu(self, file_path: Path) -> torch.Tensor:
        """Load audio file on CPU (multiprocessing safe)"""
        try:
            # Load audio (this happens on CPU)
            waveform, orig_sr = torchaudio.load(str(file_path))
            
            # Resample if needed (on CPU)
            if orig_sr != self.sample_rate:
                resampler = T.Resample(orig_sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0)  # [T]
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(int(self.sample_rate * self.duration))
    
    def _augment_audio_gpu(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation on GPU"""
        if not self.augment:
            return audio
        
        # Time stretching (±5%)
        if random.random() < 0.3:
            stretch_factor = random.uniform(0.95, 1.05)
            new_length = int(len(audio) * stretch_factor)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Pitch shifting (±2 semitones) - simplified version
        if random.random() < 0.3:
            pitch_shift = random.uniform(-2, 2)
            # Simple pitch shift using resampling (approximation)
            shift_factor = 2 ** (pitch_shift / 12)
            new_length = int(len(audio) / shift_factor)
            audio = F.interpolate(
                audio.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Volume normalization
        if random.random() < 0.5:
            volume = random.uniform(0.7, 1.3)
            audio = audio * volume
        
        # Add small amount of noise
        if random.random() < 0.2:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        
        return audio
    
    def _generate_payload(self) -> torch.Tensor:
        """Generate random payload for training"""
        # Generate random bytes
        payload_bytes = torch.randint(0, 256, (self.payload_size,), device=self.device)
        
        # Convert to bits
        payload_bits = []
        for byte_val in payload_bytes:
            for i in range(8):
                payload_bits.append((byte_val >> i) & 1)
        
        return torch.tensor(payload_bits, dtype=torch.float32, device=self.device)
    
    def _extract_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract a random segment from audio"""
        target_length = int(self.sample_rate * self.duration)
        
        if len(audio) < target_length:
            # Pad if too short
            padding = target_length - len(audio)
            audio = F.pad(audio, (0, padding))
        else:
            # Random crop if too long
            start = random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        
        return audio
    
    def __len__(self) -> int:
        return self.total_segments
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample"""
        # Select random audio file
        file_idx = idx % len(self.audio_files)
        audio_file = self.audio_files[file_idx]
        
        # Load audio on CPU (multiprocessing safe)
        audio = self._load_audio_cpu(audio_file)
        
        # Extract segment
        audio = self._extract_segment(audio)
        
        # Generate payload
        payload = self._generate_payload()
        
        # Ensure audio is [1, T] for model input
        audio = audio.unsqueeze(0)  # [1, T]
        
        return audio, payload

def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = 'cuda',
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    import multiprocessing as mp
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Training dataset with augmentation (CPU-only for multiprocessing)
    train_dataset = AudioWatermarkDataset(
        data_dir=train_dir,
        device='cpu',  # Use CPU for multiprocessing
        augment=True,
        **kwargs
    )
    
    # Validation dataset without augmentation (CPU-only for multiprocessing)
    val_dataset = AudioWatermarkDataset(
        data_dir=val_dir,
        device='cpu',  # Use CPU for multiprocessing
        augment=False,
        **kwargs
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=gpu_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=gpu_collate_fn
    )
    
    return train_loader, val_loader

def gpu_collate_fn(batch):
    """
    Custom collate function to move data to GPU in main process
    """
    audio_batch, payload_batch = zip(*batch)
    
    # Stack tensors and move to GPU
    audio_batch = torch.stack(audio_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
    payload_batch = torch.stack(payload_batch).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return audio_batch, payload_batch

# Test the dataset
if __name__ == "__main__":
    # Test with dummy data
    print("Testing AudioWatermarkDataset...")
    
    # Create dummy data directory structure
    os.makedirs("test_data/train", exist_ok=True)
    os.makedirs("test_data/val", exist_ok=True)
    
    # Create dummy audio files
    for split in ['train', 'val']:
        for i in range(3):
            dummy_audio = torch.randn(22050)  # 1 second
            torchaudio.save(f"test_data/{split}/audio_{i}.wav", dummy_audio.unsqueeze(0), 22050)
    
    # Test dataset
    dataset = AudioWatermarkDataset("test_data/train", device='cpu')
    print(f"Dataset length: {len(dataset)}")
    
    # Test sample
    audio, payload = dataset[0]
    print(f"Audio shape: {audio.shape}")
    print(f"Payload shape: {payload.shape}")
    print("✓ Dataset test passed!")
