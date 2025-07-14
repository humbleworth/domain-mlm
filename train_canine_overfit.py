#!/usr/bin/env python3
"""
CANINE overfitting test for domain MLM.
Simplified approach focusing on overfitting.
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    CanineModel,
    CanineTokenizer,
    CanineConfig,
    set_seed,
)

# Check numpy version
if not np.__version__.startswith('1.'):
    print("Warning: Requires numpy<2.0. Current version:", np.__version__)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class CanineForMLM(nn.Module):
    """Custom CANINE model with MLM head."""
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=3, help="Max training samples")
    args = parser.parse_args()
    
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load domains
    logger.info("Loading domains...")
    with open("data/domains.txt", 'r') as f:
        domains = [line.strip() for line in f if line.strip()][:args.max_samples]
    
    logger.info(f"Training on {len(domains)} domains: {domains}")
    
    # Load tokenizer and model
    logger.info("Loading CANINE tokenizer and model...")
    tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
    model = CanineForMLM.from_pretrained('google/canine-c')
    model = model.to(device)
    
    # Only train the MLM head for faster overfitting
    for param in model.canine.parameters():
        param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create simple training data - one example per domain with one masked position
    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    
    # Create multiple training examples by masking different positions
    for domain in domains:
        for mask_pos in range(len(domain)):
            # Tokenize original domain
            encoded = tokenizer(domain, truncation=True, padding="max_length", max_length=32)
            input_ids = torch.tensor(encoded['input_ids'])
            attention_mask = torch.tensor(encoded['attention_mask'])
            
            # Create labels (all -100 except masked position)
            labels = torch.full_like(input_ids, -100)
            
            # Mask one character position (+1 for CLS token)
            token_pos = mask_pos + 1
            labels[token_pos] = input_ids[token_pos].clone()
            input_ids[token_pos] = 0xE000  # Use private use area as mask
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)
    
    # Stack into batch
    train_data = {
        'input_ids': torch.stack(all_input_ids).to(device),
        'labels': torch.stack(all_labels).to(device),
        'attention_mask': torch.stack(all_attention_masks).to(device)
    }
    
    logger.info(f"Created {len(all_input_ids)} training examples")
    
    # Optimizer with higher learning rate for faster convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # Training loop
    model.train()
    logger.info("Starting training...")
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Forward pass
        outputs = model(**train_data)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if epoch % 200 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}")
    
    logger.info(f"Final training loss: {loss.item():.4f}")
    
    # Test overfitting
    logger.info("\nTesting overfitting on training data:")
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for domain in domains:
            domain_correct = 0
            domain_total = 0
            
            # Test each character position
            for pos in range(len(domain)):
                # Tokenize original
                encoded = tokenizer(domain, truncation=True, padding="max_length", max_length=32)
                input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0).to(device)
                attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(device)
                
                # Save original token
                token_pos = pos + 1  # +1 for CLS
                original_token = input_ids[0, token_pos].item()
                original_char = chr(original_token) if original_token < 128 else f'<{original_token}>'
                
                # Mask the position
                input_ids[0, token_pos] = 0xE000
                
                # Get prediction
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, token_pos]
                
                # Get predicted token
                pred_token = torch.argmax(logits).item()
                pred_char = chr(pred_token) if pred_token < 128 else f'<{pred_token}>'
                
                is_correct = (pred_token == original_token)
                if is_correct:
                    correct += 1
                    domain_correct += 1
                total += 1
                domain_total += 1
                
                logger.info(f"  {domain} pos {pos}: '{original_char}' → '{pred_char}' {'✓' if is_correct else '✗'}")
            
            domain_acc = domain_correct / domain_total if domain_total > 0 else 0
            logger.info(f"  Domain accuracy: {domain_acc:.2%} ({domain_correct}/{domain_total})")
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"\nFinal Results:")
    logger.info(f"  Final Loss: {loss.item():.4f}")
    logger.info(f"  Best Loss: {best_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    if accuracy > 0.9:
        logger.info("✓ Successfully achieved overfitting!")
    elif accuracy > 0.5:
        logger.info("⚡ Partial overfitting achieved.")
    else:
        logger.info("✗ More training needed for overfitting.")


if __name__ == "__main__":
    main()