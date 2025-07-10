# CANINE Domain MLM Pretraining Project

## Project Overview
This is a CANINE-based MLM pretraining system for domain names. The project continues pretraining the base `google/canine-c` model on domain-specific data using masked language modeling at the character level.

## Key Commands

### Training
```bash
# Basic training
python train_mlm.py

# With custom parameters
python train_mlm.py --epochs 10 --batch_size 32 --lr 5e-5 --use_wandb --output_dir ./custom-model

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
- **Model Head**: CanineForMaskedLM (autoregressive character prediction)
- **Tokenizer**: CANINE tokenizer (handles raw Unicode code points)

## Data Flow
1. **Input**: Domain names from `data/domains.txt` (one per line)
2. **Preprocessing**: Deduplication and empty line filtering
3. **Tokenization**: Convert to Unicode code points (max length: 128)
4. **Masking**: 15% of characters masked in spans
5. **Training**: Autoregressive prediction of masked characters
6. **Output**: Fine-tuned CANINE model saved in HF format

## Training Details
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate**: 1e-5 (default) with linear warmup
- **Batch Size**: 64 per device (adjustable)
- **Epochs**: 5 (default)
- **Evaluation**: Perplexity on 10% validation split
- **Early Stopping**: Patience of 2 epochs
- **Checkpoints**: Saved every epoch

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
- **GPU**: Recommended (16GB+ VRAM ideal)
- **Fallback**: CPU/MPS supported but slow
- **RAM**: 16GB minimum
- **Storage**: ~2GB for models and checkpoints

## Monitoring
- Use `--use_wandb` flag for Weights & Biases logging
- Project name: `domain-canine-pretrain`
- Tracks: loss, perplexity, learning rate

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
├── CLAUDE.md            # This file
├── README.md            # User documentation
├── requirements.txt     # Dependencies
├── train_mlm.py        # Main training script
├── inference_example.py # Demo script
├── data/
│   └── domains.txt     # Input data
└── docs/               # Additional documentation
```