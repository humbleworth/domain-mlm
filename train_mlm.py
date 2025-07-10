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
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CanineModel,
    CanineConfig,
    CanineTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Check numpy version for CANINE compatibility
if not np.__version__.startswith('1.'):
    print("Warning: CANINE requires numpy<2.0. Current version:", np.__version__)
    print("Please run: pip install 'numpy<2.0'")
    sys.exit(1)


class CanineForMaskedLM(torch.nn.Module):
    """Custom CANINE model with MLM head since it's not built-in."""
    
    def __init__(self, config):
        super().__init__()
        self.canine = CanineModel(config)
        self.config = config
        
        # MLM head - predicting Unicode codepoints
        # CANINE's max_position_embeddings is the vocab size (number of Unicode codepoints)
        vocab_size = config.max_position_embeddings
        hidden_size = config.hidden_size
        
        self.mlm_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights."""
        for module in self.mlm_head.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.LayerNorm):
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
            loss_fct = torch.nn.CrossEntropyLoss()
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per device (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./domain-canine-model",
        help="Output directory for model (default: ./domain-canine-model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.10,
        help="Masking probability for MLM (default: 0.10, optimized for domains)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler (default: 0.1)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps (default: 100)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (fp16)",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/domains.txt",
        help="Path to domains data file (default: data/domains.txt)",
    )
    
    return parser.parse_args()


def load_and_prepare_data(data_path: str) -> Tuple[Dataset, Dataset]:
    """
    Load domain data from file and prepare train/validation splits.
    
    Args:
        data_path: Path to domains.txt file
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    # Check if data file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please ensure you have domains.txt in the data/ directory"
        )
    
    logger.info(f"Loading data from {data_path}")
    
    # Load domains from file
    with open(data_path, 'r', encoding='utf-8') as f:
        domains = [line.strip() for line in f if line.strip()]
    
    # Remove duplicates
    original_count = len(domains)
    domains = list(set(domains))
    dedupe_count = len(domains)
    
    logger.info(f"Loaded {original_count} domains, {dedupe_count} after deduplication")
    
    # Warning if dataset is small
    if dedupe_count < 1_000_000:
        logger.warning(
            f"Dataset contains only {dedupe_count:,} domains. "
            f"For best results, 1M+ domains are recommended."
        )
    
    # Log dataset statistics
    domain_lengths = [len(d) for d in domains]
    logger.info(f"Average domain length: {np.mean(domain_lengths):.1f} characters")
    logger.info(f"Max domain length: {max(domain_lengths)} characters")
    logger.info(f"Min domain length: {min(domain_lengths)} characters")
    
    # Create dataset
    dataset = Dataset.from_dict({"text": domains})
    
    # Split into train/validation (90/10)
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    
    logger.info(f"Train examples: {len(dataset_dict['train']):,}")
    logger.info(f"Validation examples: {len(dataset_dict['test']):,}")
    
    return dataset_dict['train'], dataset_dict['test']


def tokenize_function(examples: Dict[str, list], tokenizer: CanineTokenizer, max_length: int) -> Dict[str, list]:
    """
    Tokenize domains for CANINE model.
    
    Args:
        examples: Batch of examples with 'text' field
        tokenizer: CANINE tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_special_tokens_mask=True,
    )


class CanineDataCollatorForLanguageModeling:
    """Custom data collator for CANINE MLM that uses Unicode codepoint masking."""
    
    def __init__(self, tokenizer, mlm_probability=0.15, mask_token_id=0xE000):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = mask_token_id  # Private use area character
        
    def __call__(self, examples):
        # Stack all tensors
        batch = {}
        for key in examples[0].keys():
            if key == 'special_tokens_mask':
                continue
            batch[key] = torch.stack([torch.tensor(ex[key]) for ex in examples])
        
        # Clone input_ids for labels
        batch['labels'] = batch['input_ids'].clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(batch['labels'].shape, self.mlm_probability)
        
        # Don't mask special tokens (CLS=57344, SEP=57345, PAD=0)
        special_tokens_mask = (
            (batch['input_ids'] == 0) | 
            (batch['input_ids'] == 57344) | 
            (batch['input_ids'] == 57345)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Apply masking
        masked_indices = torch.bernoulli(probability_matrix).bool()
        batch['labels'][~masked_indices] = -100  # Only compute loss on masked tokens
        
        # Replace masked tokens with mask token
        batch['input_ids'][masked_indices] = self.mask_token_id
        
        return batch


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute perplexity metric for evaluation.
    
    Args:
        eval_pred: EvalPrediction object from trainer
        
    Returns:
        Dictionary with perplexity metric
    """
    predictions, labels = eval_pred
    
    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)
    
    # Calculate cross-entropy loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
    loss = loss_fct(torch.from_numpy(predictions), torch.from_numpy(labels))
    
    # Convert to perplexity
    perplexity = math.exp(loss.item())
    
    return {"perplexity": perplexity}


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project="domain-canine-pretrain",
                config=vars(args),
                name=f"canine-mlm-{args.epochs}ep-{args.batch_size}bs-{args.lr}lr"
            )
            logger.info("Weights & Biases logging enabled")
        except ImportError:
            logger.warning("wandb not installed. Disabling W&B logging.")
            args.use_wandb = False
    
    # Detect device (prioritize MPS for Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Metal Performance Shaders (MPS) for M1/M2 acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slow)")
    
    # Load and prepare data
    train_dataset, eval_dataset = load_and_prepare_data(args.data_file)
    
    # Load tokenizer and model
    logger.info("Loading CANINE tokenizer and model...")
    tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
    model = CanineForMaskedLM.from_pretrained('google/canine-c')
    
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset",
    )
    
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval dataset",
    )
    
    # Data collator for MLM - use custom collator for CANINE
    data_collator = CanineDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if args.use_wandb else [],
        push_to_hub=False,
        fp16=args.fp16 and device.type == "cuda",
        dataloader_num_workers=4,
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training results
    with open(os.path.join(args.output_dir, "train_results.txt"), "w") as f:
        f.write(str(train_result))
    
    # Evaluate final model
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Training complete! Model saved to {args.output_dir}")
    logger.info(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"Final perplexity: {eval_results.get('eval_perplexity', 'N/A'):.2f}")
    
    # Close wandb run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()