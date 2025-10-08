# Micro-Chunking Watermarking System

This document describes the implementation of the micro-chunking watermarking system based on your supervisor's suggestions and your friend's detailed architecture plan.

## Overview

The micro-chunking system transforms the original slot-coordinate-dependent watermarking into a coordinate-independent system using:

- **Micro-chunks (10-20ms) with 50% overlap** per 1-second segment
- **Deterministic time-frequency mapper** for coordinate-independent embedding
- **Per-second sync anchors** for alignment without stored coordinates
- **Psychoacoustic gating** to ensure imperceptibility
- **Window-level PHM scoring** for detailed quality assessment

## Key Benefits

- **Robustness**: Overlap + self-containment survive trims/tempo edits
- **Coordinate Independence**: Sync enables alignment without slot coordinates
- **Imperceptibility**: Psychoacoustic gating keeps placements under masking
- **Robustness**: Time-frequency dispersion boosts robustness

## Architecture Components

### 1. Micro-Chunking System (`pipeline/micro_chunking.py`)

**MicroChunker**: Splits 1-second audio into overlapping micro-chunks
- 15ms windows with 50% overlap (configurable)
- Handles padding and edge cases
- Provides chunking metadata

**DeterministicMapper**: Maps window indices to (freq_bin, time_frame) slots
- Seeded randomness for deterministic mapping
- Content-aware mapping support
- Same mapping for encode/decode without storing coordinates

**Key Functions**:
- `chunk_1s_segment()`: Split audio into micro-chunks
- `map_window_to_slots()`: Deterministic slot mapping
- `apply_psychoacoustic_gate()`: Filter slots based on masking
- `fuse_overlapping_windows()`: Fuse bits from overlapping windows

### 2. Sync Anchor System (`pipeline/sync_anchors.py`)

**SyncAnchor**: Embeds and detects sync patterns for alignment
- Chirp, tone, or noise patterns (configurable)
- Robust detection with correlation
- Alignment without stored coordinates

**SyncDetector**: Advanced sync detection with multiple strategies
- Time-domain and frequency-domain detection
- Robust detection combining both methods

**Key Functions**:
- `embed_sync_pattern()`: Add sync pattern to audio
- `detect_sync_pattern()`: Find sync pattern and return offset
- `align_audio()`: Align audio based on detected offset

### 3. Window-Level Training (`pipeline/window_level_training.py`)

**WindowLevelTrainer**: Training pipeline for micro-chunking system
- Window-level label generation
- Overlap duplication for redundancy
- Psychoacoustic gating integration

**Key Functions**:
- `build_window_level_plan()`: Create window-level plans
- `duplicate_labels_across_overlaps()`: Add redundancy
- `compute_window_level_loss()`: Calculate window-level losses

### 4. Training Script (`train_window_level.py`)

New training script that replaces 1s targets with window-level labels:
- Micro-chunking with overlap
- Sync anchor integration
- Window-level loss computation
- PHM integration for metrics

**Key Features**:
- Configurable window size and overlap
- Deterministic mapping with seeds
- Sync pattern strength control
- Window-level BER tracking

### 5. Inference System (`inference/window_level_inference.py`)

**WindowLevelInference**: Complete inference pipeline
- Micro-windower with sync alignment
- Coordinate-independent decoding
- Overlap fusion for robust recovery

**Key Functions**:
- `encode_audio()`: Encode payload using micro-chunks
- `decode_audio()`: Decode with sync alignment
- `_fuse_windows()`: Fuse overlapping windows

### 6. PHM Integration (`phm/window_level_phm.py`)

**WindowLevelPHM**: PHM for window-level bit scoring
- Perceptual and technical analysis per window
- Bit-level quality metrics
- Aggregation across overlapping windows

**Key Functions**:
- `score_window_bits()`: Score individual windows
- `aggregate_window_scores()`: Combine overlapping scores
- `detect_watermark_presence()`: Presence detection
- `assess_reliability()`: Reliability assessment

## Usage

### Training

```bash
# Train with micro-chunking system
python train_window_level.py \
    --data_dir data/train \
    --val_dir data/val \
    --window_ms 15 \
    --overlap_ratio 0.5 \
    --target_bits_per_window 50 \
    --sync_strength 0.1
```

### Inference

```bash
# Encode watermark
python inference/window_level_inference.py \
    --input audio.wav \
    --output watermarked.wav \
    --payload "test message" \
    --model window_level_checkpoints/window_level_best.pt

# Decode watermark
python inference/window_level_inference.py \
    --input watermarked.wav \
    --decode_only \
    --model window_level_checkpoints/window_level_best.pt
```

## Configuration

### Key Parameters

- `window_ms`: Micro-chunk duration (10-20ms recommended)
- `overlap_ratio`: Overlap between chunks (0.5 = 50% overlap)
- `target_bits_per_window`: Bits per micro-chunk
- `mapper_seed`: Seed for deterministic mapping
- `sync_strength`: Sync pattern strength (0.0-1.0)
- `base_symbol_amp`: Base symbol amplitude

### Training Configuration

```python
@dataclass
class TrainConfig:
    # Window-level settings
    window_ms: int = 15
    overlap_ratio: float = 0.5
    target_bits_per_window: int = 50
    mapper_seed: int = 42
    sync_strength: float = 0.1
    
    # Loss weights
    w_bits: float = 1.0
    w_mse: float = 0.25
    w_perc: float = 0.01
```

## Implementation Details

### Micro-Chunking Process

1. **Audio Segmentation**: Split 1s audio into 15ms windows with 50% overlap
2. **Deterministic Mapping**: Map each window to (freq, time) slots using seeded randomness
3. **Psychoacoustic Gating**: Filter slots based on masking thresholds
4. **Symbol Placement**: Place BPSK symbols at selected slots
5. **Overlap Fusion**: Fuse overlapping windows for robust decoding

### Sync Anchor Process

1. **Pattern Embedding**: Add distinctive sync pattern at start of each 1s segment
2. **Pattern Detection**: Use correlation to find sync pattern in received audio
3. **Alignment**: Align audio based on detected sync offset
4. **Coordinate Independence**: No need to store slot coordinates

### PHM Integration

1. **Window-Level Analysis**: Process each micro-chunk individually
2. **Perceptual Features**: Extract perceptual features from audio
3. **Technical Features**: Analyze bit confidence, SNR, etc.
4. **Fusion**: Combine perceptual and technical features
5. **Quality Assessment**: Provide presence, reliability, and artifact scores

## Comparison with Original System

| Aspect | Original System | Micro-Chunking System |
|--------|----------------|----------------------|
| Chunking | 1s non-overlapping | 15ms with 50% overlap |
| Coordinates | Stored slot coordinates | Deterministic mapping |
| Alignment | Coordinate-dependent | Sync anchor-based |
| Robustness | Limited | High (overlap + sync) |
| Decoding | Requires exact coordinates | Coordinate-independent |
| Redundancy | Basic RS only | Overlap + RS |

## File Structure

```
pipeline/
├── micro_chunking.py          # Core micro-chunking system
├── sync_anchors.py            # Sync anchor implementation
└── window_level_training.py   # Window-level training pipeline

phm/
└── window_level_phm.py        # PHM for window-level scoring

inference/
└── window_level_inference.py  # Complete inference pipeline

train_window_level.py          # New training script
```

## Next Steps

1. **Test the system** with your existing datasets
2. **Tune parameters** for optimal performance
3. **Compare results** with the original system
4. **Integrate PHM training** if needed
5. **Add advanced features** like attack detection

The system is now ready for testing and should provide much better robustness and coordinate independence compared to the original slot-based approach!
