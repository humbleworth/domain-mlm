# CANINE Domain MLM Pretraining Project

## Project Overview
This is a CANINE-based MLM pretraining system for domain names. The project continues pretraining the base `google/canine-c` model on domain-specific data using masked language modeling at the character level.

## Key Commands

### Training
```bash
# Basic training (BF16 mixed precision enabled by default for A100)
python train_mlm.py

# With Weights & Biases logging
python train_mlm.py --use_wandb

# Resume from checkpoint
python train_mlm.py --resume_from ./domain-canine-model/checkpoint-step-50000

# Using streaming for the 255M domain dataset
python train_mlm.py --streaming

# Custom hyperparameters
python train_mlm.py --epochs 10 --batch_size 128 --learning_rate 5e-5 --warmup_steps 1000

# Quick test with limited samples
python train_mlm.py --num_samples 100000 --save_steps 1000

# Use local file instead of HuggingFace dataset
python train_mlm.py --dataset data/domains.txt

# Disable mixed precision if needed
python train_mlm.py --no_mixed_precision

# View all options
python train_mlm.py --help
```

### Inference
```bash
python inference_example.py
```

## Architecture
- **Base Model**: `google/canine-c` (CANINE-S, 132M parameters)
- **Task**: Character-level Masked Language Modeling (MLM)
- **Model Head**: Custom MLM head with:
  - Linear projection (768 → 768)
  - GELU activation
  - LayerNorm
  - Output projection (768 → vocab_size)
- **Tokenizer**: CANINE tokenizer (handles raw Unicode code points)
- **Custom Training Loop**: Optimized for A100 with manual mixed precision

## Data Flow
1. **Input**: 
   - Default: HuggingFace dataset `humbleworth/registered-domains` (255M domains)
   - Alternative: Local file from `data/domains.txt` (one per line)
2. **Preprocessing**: 
   - HuggingFace: Direct use with optional streaming for large dataset
   - Local: Deduplication and empty line filtering
3. **Tokenization**: Convert to Unicode code points (max length: 128)
4. **Masking**: 15% of characters masked with private use area character (U+E000)
5. **Training**: Autoregressive prediction of masked characters
6. **Mixed Precision**: FP16/BF16 support for 2-3x speedup
7. **Output**: Fine-tuned CANINE model saved in HF format

## Training Details
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-5 with linear warmup (500 steps default)
- **Batch Size**: 256 per device (optimized for A100 40GB)
- **Gradient Accumulation**: 2 steps (effective batch size: 512)
- **Gradient Clipping**: 1.0 max norm
- **Epochs**: 5 (default)
- **Mixed Precision**: 
  - BF16 by default on A100 (via autocast + GradScaler)
  - TF32 enabled for matrix operations
  - Automatic fallback to FP16/FP32
- **Evaluation**: Perplexity on 10% validation split
- **Early Stopping**: Patience of 3 epochs
- **Checkpoints**: 
  - Step-based: Every 10,000 steps
  - Epoch-based: After each epoch
  - Keeps only 3 most recent checkpoints
- **Default Dataset**: `humbleworth/registered-domains` (255M domains)
- **DataLoader Optimizations**:
  - `pin_memory=True` for faster GPU transfer
  - `persistent_workers=True` for efficiency
  - `prefetch_factor=2` for data pipeline
  - Auto-detect optimal number of workers

## Common Issues & Solutions

### GPU Memory
- Reduce `--batch_size` if OOM errors occur
- Use gradient accumulation steps if needed
- Minimum recommended: 16GB VRAM

### NumPy Version
- Must use `numpy<2.0` for CANINE compatibility
- Check with: `pip show numpy`
- Fix with: `pip install "numpy<2.0"`

### First Run
- Initial download of base model (~500MB)
- May take 5-10 minutes depending on connection

### Data Issues
- Ensure `data/domains.txt` exists and is not empty
- Script automatically deduplicates domains
- Warning displayed if <1M domains (may affect quality)

## Hardware Requirements
- **GPU**: NVIDIA A100 40GB (optimized defaults)
  - Batch size 256 with BF16 uses ~20-25GB VRAM
  - Other GPUs: Adjust batch size accordingly
  - FP16: Any modern GPU (reduces memory by ~50%)
  - BF16: Ampere+ GPUs (A100, RTX 30xx+)
- **Fallback**: CPU/MPS supported but slow
- **RAM**: 16GB minimum (streaming mode reduces requirements)
- **Storage**: ~2GB for models and checkpoints

## A100-Specific Optimizations
- **Mixed Precision**: BF16 with manual GradScaler control
- **TF32**: Enabled for all matrix operations
- **Batch Size**: 256 with gradient accumulation = 512 effective
- **Memory Usage**: ~20-25GB VRAM (leaves room for larger batches)
- **Training Speed**: 
  - ~500K steps per epoch on 255M domains
  - ~24-36 hours per epoch
  - 2-3x faster than FP32
- **Deterministic Training**: Reproducible results with fixed seeds
- **Progress Tracking**: Real-time loss, perplexity, and learning rate

## Monitoring
- Use `--use_wandb` flag for Weights & Biases logging
- Project name: `domain-canine-pretrain`
- Tracks: loss, perplexity, learning rate, global steps
- Real-time progress bar showing:
  - Running average loss (last 100 batches)
  - Current learning rate
  - Perplexity estimate
  - Batch processing speed

## Key Features
- **Custom Training Loop**: Full control over training process
- **Manual Mixed Precision**: Using PyTorch's autocast and GradScaler
- **Step-based Checkpointing**: Save model every 10k steps
- **Resume Training**: Continue from any checkpoint
- **Optimized DataLoader**: Fast data pipeline with prefetching
- **Gradient Clipping**: Stable training with large batches
- **Deterministic Mode**: Reproducible results
- **Automatic Worker Detection**: Optimal CPU utilization

## Next Steps
After pretraining:
1. Load model for downstream tasks
2. Fine-tune for domain classification
3. Use for domain generation/completion
4. Export to ONNX for production

## Troubleshooting
- **Import errors**: Check requirements.txt versions
- **CUDA errors**: Verify PyTorch CUDA version matches system
- **Slow training**: Ensure GPU is being used (check logs)
- **High loss**: Normal for first epoch; should decrease

## File Structure
```
domain-mlm/
├── CLAUDE.md            # This file (project instructions)
├── README.md            # User documentation
├── requirements.txt     # Dependencies
├── train_mlm.py        # Main training script
├── inference_mlm.py    # Inference/demo script
├── test_setup.py       # Setup verification script
├── data/
│   └── domains.txt     # Input data (optional, uses HF by default)
└── docs/               # Additional documentation
    ├── spec.md         # Original specification
    ├── technical.md    # Technical details
    └── training_checklist.md  # Training checklist
```

## Important Guidelines
- **DO NOT** place documentation files in the root directory
- All documentation should go in the `docs/` folder
- Only essential files (README.md, CLAUDE.md, scripts, requirements.txt) belong in root