#!/usr/bin/env python3
"""
Test script for training setup
"""

import sys
sys.path.insert(0, '.')

print("Testing training setup...")

try:
    from train_imperceptibility import setup_training_environment, create_sample_audio_data
    print("✓ Imports successful")
    
    # Test setup
    print("\nTesting setup_training_environment...")
    result = setup_training_environment('./data', create_samples=True, num_samples=3)
    print(f"Setup result: {result}")
    
    # Check if data was created
    from pathlib import Path
    data_path = Path('./data')
    if data_path.exists():
        print("✓ Data directory created")
        train_files = list((data_path / 'train').glob('*.wav'))
        val_files = list((data_path / 'val').glob('*.wav'))
        print(f"✓ Train files: {len(train_files)}")
        print(f"✓ Val files: {len(val_files)}")
    else:
        print("❌ Data directory not created")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
