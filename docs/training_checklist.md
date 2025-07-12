# Training Checklist

This checklist ensures the codebase is ready for MLM training on the A100.

## Pre-Training Setup

### Environment
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify numpy version < 2.0: `pip show numpy`
- [ ] Run setup test: `python test_setup.py`

### GPU Verification
- [ ] Check GPU: `nvidia-smi`
- [ ] Verify A100 detected in training script
- [ ] Confirm BF16 support

### Dataset
- [ ] Default: Uses HuggingFace `humbleworth/registered-domains` (255M domains)
- [ ] Alternative: Place local domains in `data/domains.txt` (one per line)
- [ ] Test dataset loading: `python train_mlm.py --num_samples 1000 --epochs 1`

## Training Commands

### Basic Training
```bash
# Full training with defaults (recommended)
python train_mlm.py --use_wandb

# Quick test run
python train_mlm.py --num_samples 10000 --epochs 1 --save_steps 100
```

### Resume Training
```bash
# From step checkpoint
python train_mlm.py --resume_from ./domain-canine-model/checkpoint-step-50000

# From epoch checkpoint
python train_mlm.py --resume_from ./domain-canine-model/checkpoint-epoch-2
```

### Custom Configuration
```bash
# Adjust batch size for memory constraints
python train_mlm.py --batch_size 128 --gradient_accumulation_steps 4

# Different learning rate
python train_mlm.py --learning_rate 2e-5 --warmup_steps 1000
```

## Monitoring

### Console Output
- Loss (running average of last 100 batches)
- Learning rate
- Perplexity
- Progress bar with ETA

### Weights & Biases
- Enable: `--use_wandb`
- Project: `domain-canine-pretrain`
- Tracks: loss, perplexity, learning rate, global steps

### Checkpoints
- Step checkpoints: Every 10,000 steps (keeps latest 3)
- Epoch checkpoints: After each epoch
- Best model: Lowest validation loss
- Location: `--output_dir` (default: `./domain-canine-model`)

## Expected Performance

### A100 40GB
- Batch size: 256 (default)
- Memory usage: ~20-25GB
- Speed: ~500k steps per epoch
- Time per epoch: ~24-36 hours

### Memory Issues
If OOM errors occur:
1. Reduce batch size: `--batch_size 128`
2. Increase gradient accumulation: `--gradient_accumulation_steps 4`
3. Reduce max length: `--max_length 64`

## Post-Training

### Model Outputs
```
domain-canine-model/
├── best-model/           # Lowest validation loss
├── final-model/          # Last checkpoint
├── checkpoint-epoch-N/   # Epoch checkpoints
└── checkpoint-step-N/    # Step checkpoints
```

### Inference Testing
```bash
# Test masked prediction
python inference_mlm.py --model_path ./domain-canine-model/best-model

# Interactive mode
python inference_mlm.py --interactive
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Check requirements: `pip install -r requirements.txt`
   - Verify numpy<2.0: `pip install "numpy<2.0"`

2. **CUDA Errors**
   - Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU access: `nvidia-smi`

3. **Dataset Errors**
   - Test with local file: `--dataset data/domains.txt`
   - Use streaming: `--streaming`
   - Limit samples: `--num_samples 10000`

4. **Memory Errors**
   - Reduce batch size
   - Enable gradient checkpointing (not implemented yet)
   - Monitor with `nvidia-smi -l 1`

5. **NaN Loss**
   - Reduce learning rate
   - Check data for invalid characters
   - Disable mixed precision: `--no_mixed_precision`

## Key Features

- ✅ Custom training loop (not using HF Trainer)
- ✅ Manual mixed precision with GradScaler
- ✅ Optimized DataLoader (pin_memory, persistent_workers)
- ✅ Step-based checkpointing
- ✅ Resume from checkpoint
- ✅ Gradient clipping
- ✅ Deterministic training
- ✅ Real-time progress tracking
- ✅ W&B integration
- ✅ Error handling and recovery

## Notes for Training Agent

1. The script is optimized for A100 but works on other GPUs
2. BF16 is default, falls back to FP16 automatically
3. All checkpoints include both model and training state
4. The tokenizer is saved with each checkpoint
5. Streaming mode available for very large datasets
6. Custom number of workers can be set with `--num_workers`

Good luck with training!