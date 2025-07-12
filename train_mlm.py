#!/usr/bin/env python3
"""
CANINE MLM Pretraining Script for Domain Names

This script continues pretraining the google/canine-c model on domain data
using masked language modeling at the character level.
"""

import argparse
import logging
import math
import os
import sys
import random
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import (
    CanineModel,
    CanineConfig,
    CanineTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Check numpy version for CANINE compatibility
if not np.__version__.startswith('1.'):
    print("Warning: CANINE requires numpy<2.0. Current version:", np.__version__)
    print("Please run: pip install 'numpy<2.0'")
    sys.exit(1)


class CanineForMaskedLM(nn.Module):
    """Custom CANINE model with MLM head since it's not built-in."""
    
    def __init__(self, config):
        super().__init__()
        self.canine = CanineModel(config)
        self.config = config
        
        # MLM head - predicting Unicode codepoints
        # CANINE's max_position_embeddings is the vocab size (number of Unicode codepoints)
        vocab_size = config.max_position_embeddings
        hidden_size = config.hidden_size
        
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights."""
        for module in self.mlm_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
    @classmethod
    def from_pretrained(cls, model_name):
        config = CanineConfig.from_pretrained(model_name)
        model = cls(config)
        # Load pretrained CANINE encoder
        model.canine = CanineModel.from_pretrained(model_name)
        return model
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.canine(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.max_position_embeddings),
                labels.view(-1)
            )
            
        from transformers.modeling_outputs import MaskedLMOutput
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class DomainMLMDataset(torch.utils.data.Dataset):
    """Custom dataset for domain MLM."""
    
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


class CanineDataCollatorForLanguageModeling:
    """Custom data collator for CANINE MLM that uses character-level masking."""
    
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        # CANINE special tokens
        self.cls_token_id = 57344  # [CLS]
        self.sep_token_id = 57345  # [SEP] 
        self.pad_token_id = 0       # [PAD]
        self.mask_token_id = 0xE000  # Private use area character as mask (following train_canine_overfit.py)
        
    def __call__(self, examples):
        # Stack all tensors
        batch = {}
        for key in examples[0].keys():
            batch[key] = torch.stack([ex[key] for ex in examples])
        
        # Clone input_ids for labels
        batch['labels'] = batch['input_ids'].clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(batch['labels'].shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = (
            (batch['input_ids'] == self.pad_token_id) | 
            (batch['input_ids'] == self.cls_token_id) | 
            (batch['input_ids'] == self.sep_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Apply masking
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['labels'][~masked_indices] = -100  # Only compute loss on masked tokens
        
        # Following the CANINE paper and train_canine_overfit.py approach:
        # Replace masked positions with the mask token (0xE000)
        batch['input_ids'][masked_indices] = self.mask_token_id
        
        return batch


# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrain CANINE on domain names using MLM"
    )
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per device (optimized for A100)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    
    # Model arguments
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Masking probability for MLM")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="humbleworth/registered-domains", help="HuggingFace dataset or local file")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (None for all)")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation set size as fraction")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (default: auto)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./domain-canine-model", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    
    # Mixed precision and optimization
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16 mixed precision (default for A100)")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision")
    
    # Logging and debugging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="domain-canine-pretrain", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint directory")
    
    return parser.parse_args()


def load_and_prepare_data(dataset_name: str, dataset_config: Optional[str] = None, 
                         dataset_split: str = "train", num_samples: Optional[int] = None,
                         streaming: bool = False, test_size: float = 0.1) -> Tuple[Dataset, Dataset]:
    """Load domain data from HuggingFace dataset or local file."""
    # Check if it's a local file
    if os.path.exists(dataset_name):
        logger.info(f"Loading data from local file: {dataset_name}")
        
        # Load domains from file
        with open(dataset_name, 'r', encoding='utf-8') as f:
            domains = [line.strip() for line in f if line.strip()]
        
        # Remove duplicates
        original_count = len(domains)
        domains = list(set(domains))
        dedupe_count = len(domains)
        
        logger.info(f"Loaded {original_count} domains, {dedupe_count} after deduplication")
        
        # Create dataset
        dataset = Dataset.from_dict({"text": domains})
    else:
        # Load from HuggingFace
        logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
        
        # Load dataset with optional configuration
        load_kwargs = {
            "path": dataset_name,
            "split": dataset_split,
            "streaming": streaming
        }
        if dataset_config:
            load_kwargs["name"] = dataset_config
            
        dataset = load_dataset(**load_kwargs)
        
        # For streaming datasets, we can't get the exact count
        if streaming:
            logger.info("Using streaming mode - exact dataset size unknown")
        else:
            logger.info(f"Loaded {len(dataset):,} examples from HuggingFace")
    
    # Limit samples if requested
    if num_samples and not streaming:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset):,} samples")
    
    # For streaming datasets, take a subset
    if streaming and num_samples:
        dataset = dataset.take(num_samples)
        logger.info(f"Will stream up to {num_samples:,} samples")
    
    # Log dataset statistics (for non-streaming datasets)
    if not streaming:
        # Get text column (handle different column names)
        text_column = None
        for col in ["text", "domain", "domains", "content", "raw"]:
            if col in dataset.column_names:
                text_column = col
                break
        
        if text_column:
            # Rename column to 'text' if needed
            if text_column != "text":
                dataset = dataset.rename_column(text_column, "text")
                logger.info(f"Renamed column '{text_column}' to 'text'")
            
            # Calculate statistics
            texts = dataset["text"][:10000]  # Sample for statistics
            domain_lengths = [len(d) for d in texts]
            logger.info(f"Average domain length: {np.mean(domain_lengths):.1f} characters")
            logger.info(f"Max domain length: {max(domain_lengths)} characters")
            logger.info(f"Min domain length: {min(domain_lengths)} characters")
        else:
            raise ValueError(f"Could not find text column in dataset. Available columns: {dataset.column_names}")
    
    # Split into train/validation
    if streaming:
        # For streaming, we'll manually split
        # Note: This is a simple split strategy for streaming datasets
        # The eval dataset will see different data than train
        train_dataset = dataset
        if num_samples:
            skip_amount = int(num_samples * (1 - test_size))
            take_amount = int(num_samples * test_size)
        else:
            skip_amount = 10000000  # Skip first 10M for eval
            take_amount = 1000000   # Take 1M for eval
        eval_dataset = dataset.skip(skip_amount).take(take_amount)
        logger.info("Created train/eval splits for streaming dataset")
    else:
        dataset_dict = dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['test']
        
        logger.info(f"Train examples: {len(train_dataset):,}")
        logger.info(f"Validation examples: {len(eval_dataset):,}")
    
    return train_dataset, eval_dataset


def train_epoch(model, dataloader, optimizer, scheduler, device, args, epoch, scaler, step_checkpoint_files: List[str], tokenizer):
    """Train for one epoch with optimized settings."""
    model.train()
    total_loss = 0
    accumulated_loss = 0
    accumulated_samples = 0
    
    # Track recent losses for display
    recent_losses = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision forward pass
        if args.use_amp and scaler is not None:
            with autocast():
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
        
        # Accumulate loss for display
        accumulated_loss += loss.item() * args.gradient_accumulation_steps
        accumulated_samples += len(input_ids)
        
        # Update weights after accumulating gradients
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            if args.use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Calculate effective batch stats
            avg_accumulated_loss = accumulated_loss / args.gradient_accumulation_steps
            
            # Track recent losses
            recent_losses.append(avg_accumulated_loss)
            if len(recent_losses) > 100:
                recent_losses.pop(0)
            
            # Update progress bar
            avg_recent_loss = np.mean(recent_losses) if recent_losses else avg_accumulated_loss
            current_lr = scheduler.get_last_lr()[0]
            
            # Safely calculate perplexity (avoid overflow)
            try:
                ppl = math.exp(min(avg_recent_loss, 20))  # Cap at e^20 to avoid overflow
            except (OverflowError, ValueError):
                ppl = float('inf')
            
            pbar.set_postfix({
                'loss': f'{avg_recent_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'ppl': f'{ppl:.2f}' if ppl < 1e6 else 'inf'
            })
            
            # Log to wandb
            if args.use_wandb and batch_idx % args.logging_steps == 0:
                global_step = epoch * len(dataloader) + batch_idx + 1
                try:
                    wandb.log({
                        'train_loss': avg_accumulated_loss,
                        'train_perplexity': math.exp(avg_accumulated_loss),
                        'learning_rate': current_lr,
                        'epoch': epoch + 1,
                        'global_step': global_step
                    })
                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {e}")
            
            # Reset accumulators
            accumulated_loss = 0
            accumulated_samples = 0
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Save checkpoint every N steps
        global_step = epoch * len(dataloader) + batch_idx + 1
        if (batch_idx + 1) % args.save_steps == 0:
            step_checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-step-{global_step}')
            os.makedirs(step_checkpoint_dir, exist_ok=True)
            
            # Save model and training state
            try:
                model.canine.save_pretrained(step_checkpoint_dir)
                tokenizer.save_pretrained(step_checkpoint_dir)
                torch.save({
                'epoch': epoch,
                'step': batch_idx + 1,
                'global_step': global_step,
                'mlm_head_state_dict': model.mlm_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'train_loss': total_loss / (batch_idx + 1),
                'args': args,
                }, os.path.join(step_checkpoint_dir, 'training_state.bin'))
                
                logger.info(f"Saved checkpoint at step {global_step}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
            
            # Keep only the latest N step checkpoints
            step_checkpoint_files.append(step_checkpoint_dir)
            if len(step_checkpoint_files) > args.save_total_limit:
                old_checkpoint = step_checkpoint_files.pop(0)
                if os.path.exists(old_checkpoint):
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    # Handle remaining gradients
    if len(dataloader) % args.gradient_accumulation_steps != 0:
        if args.use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
            optimizer.step()
        
        scheduler.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, args):
    """Evaluate the model and compute perplexity."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if args.use_amp:
                with autocast():
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids, attention_mask, labels)
                loss = outputs.loss
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    # Safely calculate perplexity
    try:
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    except (OverflowError, ValueError):
        perplexity = float('inf')
    
    return avg_loss, perplexity


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility (using transformers set_seed like train_canine_overfit.py)
    set_seed(args.seed)
    
    # Enable deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup W&B logging
    if args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"canine-mlm-{args.epochs}ep-{args.batch_size}bs",
                config=vars(args)
            )
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            args.use_wandb = False
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # A100-specific optimizations
        if "A100" in gpu_name:
            logger.info("Detected NVIDIA A100 - enabling optimizations")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for matrix operations")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS)")
        args.use_amp = False  # AMP not supported on MPS
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow)")
        args.use_amp = False
    
    # Check mixed precision settings
    if args.no_mixed_precision:
        args.use_amp = False
        args.fp16 = False
        args.bf16 = False
        logger.info("Mixed precision disabled")
    elif args.use_amp and device.type == "cuda":
        if args.bf16 and torch.cuda.is_bf16_supported():
            logger.info("Using BF16 mixed precision")
            # Set autocast dtype
            torch.set_autocast_gpu_dtype(torch.bfloat16)
        elif args.fp16:
            logger.info("Using FP16 mixed precision")
            torch.set_autocast_gpu_dtype(torch.float16)
        else:
            # Default to BF16 on A100, FP16 otherwise
            if "A100" in torch.cuda.get_device_name() and torch.cuda.is_bf16_supported():
                torch.set_autocast_gpu_dtype(torch.bfloat16)
                logger.info("Using BF16 mixed precision (A100 default)")
            else:
                torch.set_autocast_gpu_dtype(torch.float16)
                logger.info("Using FP16 mixed precision (default)")
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler() if args.use_amp and device.type == "cuda" else None
    
    # Load and prepare data
    train_dataset, eval_dataset = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        streaming=args.streaming,
        test_size=args.test_size
    )
    
    # Load tokenizer and model
    logger.info("Loading CANINE tokenizer and model...")
    tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
    
    # Initialize or load model
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        model = CanineForMaskedLM(CanineConfig.from_pretrained(args.resume_from))
        model.canine = CanineModel.from_pretrained(args.resume_from)
        
        # Load training state
        state_path = os.path.join(args.resume_from, 'training_state.bin')
        checkpoint = None
        if os.path.exists(state_path):
            try:
                checkpoint = torch.load(state_path, map_location='cpu')
                model.mlm_head.load_state_dict(checkpoint['mlm_head_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resumed from epoch {checkpoint['epoch']}")
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")
                start_epoch = 0
        else:
            logger.warning(f"No training state found at {state_path}")
            start_epoch = 0
    else:
        model = CanineForMaskedLM.from_pretrained('google/canine-c')
        start_epoch = 0
    
    model.to(device)
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Ensure model is in training mode
    model.train()
    
    # Create datasets
    try:
        train_dataset_torch = DomainMLMDataset(train_dataset, tokenizer, args.max_length)
        eval_dataset_torch = DomainMLMDataset(eval_dataset, tokenizer, args.max_length)
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise
    
    # Data collator
    data_collator = CanineDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )
    
    # Determine number of workers
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = min(multiprocessing.cpu_count(), 16)
    logger.info(f"Using {num_workers} DataLoader workers")
    
    # Create DataLoaders with optimizations
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset_torch,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator
    )
    
    eval_loader = DataLoader(
        eval_dataset_torch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Calculate total steps
    total_steps = (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps * args.epochs
    
    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Load optimizer and scheduler states if resuming
    if args.resume_from and 'checkpoint' in locals() and checkpoint is not None:
        try:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state")
            if scaler is not None and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("Loaded scaler state")
        except Exception as e:
            logger.warning(f"Failed to load optimizer/scheduler state: {e}")
            logger.warning("Starting with fresh optimizer/scheduler")
    
    logger.info(f"\nTraining setup:")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {args.warmup_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Starting from epoch: {start_epoch}")
    
    # Training loop
    best_eval_loss = float('inf')
    patience_counter = 0
    step_checkpoint_files = []
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args, epoch, scaler, step_checkpoint_files, tokenizer)
        train_perplexity = math.exp(train_loss)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        
        # Evaluate
        eval_loss, eval_perplexity = evaluate(model, eval_loader, device, args)
        logger.info(f"Eval Loss: {eval_loss:.4f}, Eval Perplexity: {eval_perplexity:.2f}")
        
        # Log to wandb
        if args.use_wandb:
            try:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_perplexity': train_perplexity,
                    'eval_loss': eval_loss,
                    'eval_perplexity': eval_perplexity,
                })
            except Exception as e:
                logger.warning(f"Failed to log epoch metrics to wandb: {e}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-epoch-{epoch+1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.canine.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        torch.save({
            'epoch': epoch,
            'mlm_head_state_dict': model.mlm_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'eval_loss': eval_loss,
            'eval_perplexity': eval_perplexity,
            'args': args,
        }, os.path.join(checkpoint_dir, 'training_state.bin'))
        
        logger.info(f"Saved checkpoint: {checkpoint_dir}")
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            patience_counter = 0
            
            best_model_dir = os.path.join(args.output_dir, 'best-model')
            os.makedirs(best_model_dir, exist_ok=True)
            
            model.canine.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            
            torch.save({
                'epoch': epoch,
                'mlm_head_state_dict': model.mlm_head.state_dict(),
                'best_eval_loss': best_eval_loss,
                'best_eval_perplexity': eval_perplexity,
                'args': args,
            }, os.path.join(best_model_dir, 'training_state.bin'))
            
            logger.info(f"Saved new best model with eval loss: {best_eval_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
                break
    
    # Save final model
    final_dir = os.path.join(args.output_dir, 'final-model')
    os.makedirs(final_dir, exist_ok=True)
    
    model.canine.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    torch.save({
        'epoch': epoch,
        'mlm_head_state_dict': model.mlm_head.state_dict(),
        'final_eval_loss': eval_loss,
        'final_eval_perplexity': eval_perplexity,
        'best_eval_loss': best_eval_loss,
        'args': args,
    }, os.path.join(final_dir, 'training_state.bin'))
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info(f"Final model saved in: {final_dir}")
    
    # Close wandb
    if args.use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()