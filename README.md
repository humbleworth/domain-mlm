# Domain CANINE Pretraining

Pretrain CANINE on domain names for masked language modeling.

## Overview

This project continues pretraining Google's CANINE-c (character-level) model on domain name data using masked language modeling (MLM). The resulting model can be used for downstream tasks like domain classification, generation, or similarity matching.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Basic training with default parameters:
```bash
python train_mlm.py
```

Custom training configuration:
```bash
python train_mlm.py --epochs 10 --batch_size 32 --lr 5e-5 --use_wandb
```

### Inference

Load and use the pretrained model:
```bash
python inference_example.py
```

## Requirements

- Python 3.8+
- GPU with 16GB+ VRAM (recommended)
- Domain data in `data/domains.txt` (one domain per line)

## Model Details

- **Base Model**: `google/canine-c` (132M parameters)
- **Objective**: Character-level masked language modeling
- **Input**: Raw domain strings (Unicode code points)
- **Output**: Character predictions for masked positions

## Training Parameters

- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size per device (default: 64)
- `--lr`: Learning rate (default: 1e-5)
- `--use_wandb`: Enable Weights & Biases logging
- `--output_dir`: Model save directory (default: ./domain-canine-model)

## License

MIT