# Domain CANINE MLM Pretraining

Pre-train CANINE on domain names for character-level masked language modeling, optimized for NVIDIA A100 GPUs.

## Overview

This project continues pretraining Google's CANINE-c (character-level) model on domain name data using masked language modeling (MLM). The implementation is optimized for fast training on A100 GPUs with custom training loop, mixed precision, and efficient data loading.

## Features

- **Custom Training Loop**: Full control over training process with manual mixed precision
- **A100 Optimized**: BF16 mixed precision, TF32 operations, large batch sizes
- **Efficient Data Pipeline**: Pin memory, persistent workers, prefetching
- **Step-based Checkpointing**: Save model every 10k steps
- **Resume Training**: Continue from any checkpoint
- **HuggingFace Integration**: Uses `humbleworth/registered-domains` dataset (255M domains)

## Installation

```bash
pip install -r requirements.txt
```

Ensure you have CUDA-compatible PyTorch installed. For A100 GPUs, PyTorch 2.0+ is recommended.

## Quick Start

### Basic Training

```bash
# Train with default settings (optimized for A100)
python train_mlm.py

# With Weights & Biases logging
python train_mlm.py --use_wandb

# Resume from checkpoint
python train_mlm.py --resume_from ./domain-canine-model/checkpoint-step-50000
```

### Custom Configuration

```bash
# Adjust hyperparameters
python train_mlm.py \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --save_steps 5000

# Quick test with limited data
python train_mlm.py --num_samples 100000 --epochs 2

# Use local dataset
python train_mlm.py --dataset data/domains.txt
```

### Inference

```bash
# Interactive masked token prediction
python inference_mlm.py --interactive

# Load specific checkpoint
python inference_mlm.py --model_path ./domain-canine-model/best-model
```

## Model Architecture

- **Base Model**: `google/canine-c` (132M parameters)
- **Task**: Character-level masked language modeling
- **Tokenizer**: CANINE tokenizer (raw Unicode code points)
- **MLM Head**: Linear → GELU → LayerNorm → Linear
- **Masking**: 15% of characters replaced with mask token (U+E000)

## Training Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| Batch Size | 256 | Per-device batch size |
| Gradient Accumulation | 2 | Effective batch size = 512 |
| Learning Rate | 1e-5 | With linear warmup |
| Warmup Steps | 500 | Linear schedule warmup |
| Mixed Precision | BF16 | Automatic on A100 |
| Optimizer | AdamW | Weight decay 0.01 |
| Max Length | 128 | Maximum domain length |
| Save Steps | 10,000 | Checkpoint frequency |

## Hardware Requirements

- **Recommended**: NVIDIA A100 40GB (optimized defaults)
- **Minimum**: Any GPU with 16GB+ VRAM
- **Memory Usage**: ~20-25GB with default settings
- **Training Speed**: ~24-36 hours per epoch on 255M domains

## Dataset

Default dataset: `humbleworth/registered-domains` on HuggingFace
- 255M registered domain names
- Automatic streaming for large-scale training
- 90/10 train/validation split

To use a local dataset:
```bash
python train_mlm.py --dataset data/domains.txt
```

Format: One domain per line in plain text.

## Monitoring

Training progress is displayed in real-time with:
- Running average loss (last 100 batches)
- Current learning rate
- Perplexity estimate
- Batch processing speed

Enable W&B logging for detailed metrics:
```bash
python train_mlm.py --use_wandb --wandb_project my-project
```

## Checkpoints

The training script saves:
- **Step checkpoints**: Every 10k steps (keeps latest 3)
- **Epoch checkpoints**: After each epoch
- **Best model**: Lowest validation loss
- **Final model**: Last checkpoint

Each checkpoint includes:
- CANINE encoder weights
- MLM head weights
- Optimizer state
- Training configuration

## Advanced Options

```bash
# Disable mixed precision
python train_mlm.py --no_mixed_precision

# Force FP16 instead of BF16
python train_mlm.py --fp16 --bf16 false

# Streaming mode for huge datasets
python train_mlm.py --streaming

# Custom number of workers
python train_mlm.py --num_workers 8
```

## License

MIT