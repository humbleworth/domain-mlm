# Domain-Focused Masked Language Model Pre-Training Spec

This document provides a high-level specification and guide for pre-training a masked language model (MLM) specialized for domain names. The model will operate in a character-level, tokenization-free manner similar to Google's CANINE model, using masked autoregressive character prediction as the pre-training objective. This serves as a foundational pre-training step to create domain-adapted representations, which can later be fine-tuned for tasks like domain classification (e.g., catch/drop prediction), generation, or valuation.

The focus is on efficiency, scalability, and domain-specific adaptation. We'll leverage Hugging Face's Transformers library for implementation, with Automatic Mixed Precision (AMP) enabled to optimize GPU memory and speed during training.

## 1. Project Overview

### Objectives
- **Primary Goal**: Pre-train a character-level transformer encoder on a large corpus of registered domain names using a masked language modeling (MLM) task. The model will predict masked character spans autoregressively, adapting to domain name patterns (e.g., hyphens, numbers, TLD structures, pronounceable words).
- **Why Domain-Focused?**: Domains are short, structured text (avg. ~16 chars) with unique patterns (e.g., TLDs, hyphens, numeric mixes). Pre-training on domains creates embeddings better suited for downstream tasks like:
  - Domain catch/drop prediction.
  - Domain generation or suggestion.
  - Malicious domain detection.
  - TLD classification or clustering.
- **Similarities to CANINE**: Tokenization-free (direct Unicode code points for characters); character-level encoding; MLM with autoregressive span prediction; multilingual support (domains often include non-English chars).
- **Key Enhancements**: Domain-specific masking (e.g., higher probability for masking TLDs or hyphens); AMP for faster training on large datasets.

### Expected Outcomes
- A pre-trained model checkpoint in Hugging Face format.
- Improved performance on domain-related fine-tuning tasks (e.g., 5-10% AUC/F1 lift over generic CANINE).
- Model size: Start with CANINE-S (132M params) for efficiency; scale to larger if needed.

### Assumptions and Constraints
- Hardware: NVIDIA GPU (e.g., A100 or RTX series) with ≥16GB VRAM for AMP.
- Libraries: Transformers ≥4.30, Datasets ≥2.10, Torch ≥2.0, Accelerate ≥0.20.
- No additional data augmentation; rely on the raw dataset's scale.
- Training time: ~1-3 days per epoch on a single A100, depending on batch size.

## 2. Dataset

### Source
- **Hugging Face Dataset**: [humbleworth/registered-domains](https://huggingface.co/datasets/humbleworth/registered-domains)
  - Size: ~255M domains (4.1 GB text file).
  - Format: One domain per line (e.g., "example.com"), shuffled, lowercase, ASCII-only.
  - TLDs: 1,274 unique (top: .com ~54.5%).
  - Characteristics: 8.8% with numbers, 11.4% with hyphens; lengths 4-77 chars.

### Preprocessing
- **Loading**: Use `datasets.load_dataset("humbleworth/registered-domains")` for streaming mode to handle large size without full RAM load.
- **Filtering/Cleaning**:
  - Remove any non-ASCII or invalid domains (though dataset is already 100% ASCII).
  - Optional: Stratified sampling by TLD for balanced representation (e.g., cap .com at 50% of batch).
- **Splitting**: 99% train, 1% validation (random split; ~2.5M val samples).
- **Input Preparation**:
  - Convert domains to character-level sequences: `input_ids = [ord(char) for char in domain]`.
  - Pad/truncate to max_length (e.g., 64 chars, covering >99% of domains).
  - No traditional tokenization; direct code points as in CANINE.

### Data Statistics (From Provided README)
- Total: 255M domains.
- Use cases alignment: Ideal for learning domain patterns, as it's comprehensive and diverse.

## 3. Model Architecture

### Base Model
- **Starting Point**: Fine-tune from `google/canine-c` (132M params, pre-trained on 104 languages with MLM).
  - Why? Already tokenization-free and character-level; adapts well to short sequences like domains.

### Customizations
- **Encoder**: CANINE transformer (12 layers, hidden size 768).
- **MLM Head**: Autoregressive prediction for masked spans (as in CANINE-c):
  - Mask 15-20% of characters in spans (e.g., 1-3 chars per mask).
  - Predict masked chars autoregressively using a linear head over character vocabulary (Unicode range: 0-65,535, but restrict to domain-common chars like a-z, 0-9, ., - for efficiency).
- **Pooling/Output**: Mean pooling over character embeddings for domain representations (stable for short texts).
- **Vocabulary**: Fixed to Unicode code points; no learned embeddings beyond CANINE's.

### Hyperparameters
- Hidden size: 768.
- Dropout: 0.1.
- Max sequence length: 64.

## 4. Training Setup

### Objective
- **Loss**: Masked autoregressive character loss (cross-entropy over predicted chars in masked spans).
- **Metrics**: Perplexity (primary); validation loss; optional next-sentence prediction if pairing domains.

### Optimization
- **Optimizer**: AdamW (lr=2e-5 for encoder, 1e-4 for MLM head; weight decay=0.01).
- **Scheduler**: Linear warmup (10% of steps) + cosine decay.
- **Batch Size**: Effective 512+ (per-GPU 128 + gradient accumulation 4).
- **Epochs**: 3-5 (monitor val loss for early stopping).
- **AMP**: Enable via `torch.cuda.amp` (or Accelerate library) for FP16/FP32 mixed precision:
  - Reduces memory by ~50%; speeds up by 2-3x on compatible GPUs.
  - Handle via `GradScaler` for stable gradients.

### Hardware & Scaling
- **Single GPU**: A100 (40GB) for large batches.
- **Multi-GPU**: Use Accelerate or DDP for distributed training.
- **Monitoring**: Weights & Biases (W&B) for logs; TensorBoard for curves.

## 5. Implementation Steps

### Step 1: Environment Setup
- Install: `pip install transformers datasets torch accelerate wandb`.
- Ensure CUDA ≥11.0 for AMP.

### Step 2: Data Preparation
- Load dataset in streaming mode.
- Create custom Dataset class for character encoding and masking.

### Step 3: Model Initialization
- Load CANINE: `CanineModel.from_pretrained('google/canine-c')`.
- Add MLM head: Custom nn.Module for autoregressive prediction.

### Step 4: Training Loop
- Use Trainer API from Transformers for simplicity (handles AMP, multi-GPU).
- Custom collator for dynamic masking.
- Train: `trainer.train()` with AMP enabled via `fp16=True`.
- Evaluate: Compute perplexity on val set.

### Step 5: Evaluation & Saving
- Validate on held-out domains (e.g., mask and predict TLDs).
- Save: `model.save_pretrained("domain-mlm-model")`.
- Push to Hugging Face Hub for sharing.

### Potential Challenges & Mitigations
- **Imbalanced TLDs**: Oversample rare TLDs in batches.
- **Short Sequences**: Adjust masking rate to avoid over-masking.
- **AMP Stability**: Monitor for NaN losses; fallback to FP32 if unstable.
- **Scale**: If OOM, reduce batch size or use gradient checkpointing.

### Timeline Estimate
- Setup & Prototyping: 1-2 days.
- Full Training: 1-2 weeks (depending on hardware; iterate with subsets first).
- Evaluation/Fine-Tuning Prep: 1 day.

This spec provides a flexible blueprint—refine based on initial experiments. For code snippets or detailed scripts, provide more specifics (e.g., exact hyperparameters or hardware).