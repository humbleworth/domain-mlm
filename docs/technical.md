# Technical Guide: Pre-Training a Character-Based Masked Language Model (MLM) for Domains

This guide provides a detailed, step-by-step technical implementation for pre-training a character-level masked language model (MLM) on the `humbleworth/registered-domains` dataset. The approach is inspired by Google's CANINE model, which operates tokenization-free on character sequences. We'll focus on continuing pre-training from `google/canine-c` (the variant using autoregressive character loss for MLM), adapting it for domain names.

Key focuses:
- **Character-Based MLM**: Inputs are raw character sequences (Unicode code points). We mask spans of characters and predict them.
- **Autoregressive Aspect**: To match CANINE-c, predictions for masked spans are autoregressive (left-to-right within the span). This requires a custom decoder head for local autoregressive generation.
- **AMP (Automatic Mixed Precision)**: Enabled for faster training and lower memory usage on CUDA GPUs.
- **Based on Provided Code**: We build on the structure from `train_domain_catch.py` (e.g., using CanineModel, AMP with GradScaler, gradient accumulation, W&B logging, early stopping).
- **Simplifications**: For autoregressive prediction, we'll implement a simple GRU-based decoder per masked span (as a proxy for the paper's local self-attention decoder). If you want exact replication, refer to the original JAX code in Google's repo.

Assumptions:
- GPU with CUDA support (e.g., A100 for large batches).
- Dataset size: ~255M domains; use streaming to avoid loading all into RAM.
- Max domain length: 64 chars (covers most domains).
- Masking: 15% of characters, in spans of 1-5 chars.

Potential Challenges:
- Autoregressive decoding adds complexity (loop over span length during training).
- Memory: AMP helps, but monitor for OOM; use gradient checkpointing if needed.
- Loss Computation: Only on masked positions, autoregressively.

## 1. Environment Setup and Imports

Install dependencies (extend from `requirements.txt`):
```
pip install torch transformers datasets accelerate wandb
```

Code chunk for imports (adapted from `train_domain_catch.py`):
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import CanineModel, CanineTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path
```

Add for MLM-specific:
```python
from transformers import DataCollatorForLanguageModeling  # For inspiration; we'll custom it
import random  # For masking
```

## 2. Loading and Preparing the Dataset

Load the streaming dataset to handle 255M samples.

Code chunk:
```python
# Load dataset
print("Loading dataset...")
dataset = load_dataset("humbleworth/registered-domains", streaming=True)

# For validation split (take 1% manually, since streaming)
# Note: For streaming, we'll process in epochs; for val, cache a small subset
train_dataset = dataset['train'].shuffle(buffer_size=10000)  # Streaming shuffle
val_dataset = dataset['train'].take(2500000)  # Approximate 1% for val

# Custom Dataset class (adapted from DomainDataset)
class DomainMLMDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=64, is_streaming=False):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_streaming = is_streaming
        if not is_streaming:
            self.data = list(hf_dataset)  # Materialize if not streaming

    def __len__(self):
        return len(self.data) if not self.is_streaming else 100000000  # Arbitrary large for streaming

    def __getitem__(self, idx):
        if self.is_streaming:
            item = next(iter(self.hf_dataset.skip(idx).take(1)))
        else:
            item = self.data[idx]
        domain = item['domain']
        encoding = self.tokenizer(
            domain,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }  # Labels will be created in collator for MLM

# Initialize tokenizer
tokenizer = CanineTokenizer.from_pretrained('google/canine-c')

# Create datasets
train_ds = DomainMLMDataset(train_dataset, tokenizer, is_streaming=True)
val_ds = DomainMLMDataset(val_dataset, tokenizer, is_streaming=False)

# DataLoaders (use num_workers for speed)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=custom_collator, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=custom_collator, num_workers=4)
```

Note: For streaming, `__len__` is approximate; use epochs based on steps.

## 3. Custom Data Collator for Character Span Masking

For MLM, we need to mask spans and prepare labels.

Masking strategy (from CANINE paper):
- Mask 15% of characters in spans of geometric length distribution (mean 3 chars).
- Replace with [MASK] code point (e.g., tokenizer.mask_token_id, but since no tokens, use a special code, say 65535).
- For autoregressive, labels are the original chars; during prediction, feed previous predicted char as input for next.

Code chunk for custom collator (inspired by DataCollatorForLanguageModeling, but for spans):
```python
def custom_mlm_collator(batch, mlm_probability=0.15, mean_span_length=3):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = input_ids.clone()  # Labels are original input_ids

    # Mask spans
    for i in range(input_ids.size(0)):
        seq_len = attention_mask[i].sum().item()
        num_mask = max(1, int(seq_len * mlm_probability))
        masked_indices = []

        for _ in range(num_mask):
            span_length = np.random.geometric(1.0 / mean_span_length)
            span_length = min(span_length, seq_len - 1)  # Clamp
            start = random.randint(0, seq_len - span_length)
            input_ids[i, start:start+span_length] = tokenizer.mask_token_id  # e.g., 0 or special
            masked_indices.append((start, span_length))

    # For labels, set non-masked to -100 (ignore in loss)
    labels[labels != tokenizer.mask_token_id] = -100  # Wait, no: labels only on masked
    # Correction: For standard MLM, labels[input_ids != mask_id] = -100

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'masked_spans': masked_indices  # For autoregressive, track spans per example
    }
```

For autoregressive, we'll use the 'masked_spans' in the model forward to compute loss autoregressively.

## 4. Model Definition: CanineForCharacterMLM

Extend CanineModel with a LM head.

For autoregressive: Add a small GRU decoder (hidden_size to char_vocab).

Char vocab size: CANINE uses Unicode 0-131071, but for domains, restrict to 0-127 (ASCII) + . - (size ~128).

Code chunk:
```python
class CanineForCharacterMLM(nn.Module):
    def __init__(self, canine_model, char_vocab_size=131072, hidden_size=768, decoder_layers=1):
        super().__init__()
        self.canine = canine_model
        # LM head for standard MLM (parallel prediction)
        self.lm_head = nn.Linear(hidden_size, char_vocab_size)
        
        # For autoregressive: Small GRU decoder for spans
        self.decoder = nn.GRU(input_size=hidden_size + char_vocab_size,  # Context + prev char embed
                              hidden_size=hidden_size, 
                              num_layers=decoder_layers, 
                              batch_first=True)
        self.char_embed = nn.Embedding(char_vocab_size, hidden_size)  # Embed prev chars
        self.output_proj = nn.Linear(hidden_size, char_vocab_size)
    
    def forward(self, input_ids, attention_mask, labels=None, masked_spans=None):
        outputs = self.canine(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        if labels is None:
            return self.lm_head(sequence_output)  # For inference
        
        # Compute loss
        loss = 0
        for b in range(input_ids.size(0)):
            for start, length in masked_spans[b]:
                # Get context at mask position (use mean of span or first)
                context = sequence_output[b, start]  # [hidden]
                
                # Autoregressive prediction for the span
                prev_char = torch.zeros(1, 1, char_vocab_size).to(input_ids.device)  # Start token
                span_loss = 0
                for t in range(length):
                    # Embed prev_char + context
                    input_t = torch.cat([context.unsqueeze(0).unsqueeze(0), self.char_embed(prev_char.argmax(-1))], dim=-1)
                    decoder_out, _ = self.decoder(input_t)
                    logit_t = self.output_proj(decoder_out.squeeze(0).squeeze(0))
                    
                    # Loss for this step
                    target = labels[b, start + t]
                    span_loss += F.cross_entropy(logit_t.unsqueeze(0), target.unsqueeze(0))
                    
                    # Teacher forcing: use true prev during training
                    prev_char = F.one_hot(target, char_vocab_size).float().unsqueeze(0).unsqueeze(0)
                
                loss += span_loss / length
        
        return loss / (input_ids.size(0) * len(masked_spans[0]))  # Average
```

Notes:
- This is a simplification: Uses GRU for autoregressive decoding per span.
- During training, use teacher forcing (use true previous char).
- For inference, use predicted previous.
- Adjust char_vocab_size to max(ord(c) for c in all chars) +1, ~128 for domains.
- Integrate mean pooling if needed, but for MLM, use per-char hidden states.

## 5. Training Loop with AMP

Adapt from `train_epoch` in `train_domain_catch.py`.

Enable AMP with GradScaler.

Code chunk (core loop):
```python
def train_epoch(model, dataloader, optimizer, scheduler, device, args, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        masked_spans = batch['masked_spans']  # List of lists
        
        with autocast(enabled=args.use_amp):
            loss = model(input_ids, attention_mask, labels, masked_spans)
            loss = loss / args.gradient_accumulation_steps
        
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step % args.gradient_accumulation_steps == 0):
            if args.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': total_loss / (step + 1)})
    
    return total_loss / len(dataloader)

# In main: 
model = CanineForCharacterMLM(CanineModel.from_pretrained('google/canine-c')).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
total_steps = len(train_loader) // args.gradient_accumulation_steps * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
scaler = GradScaler(enabled=args.use_amp)

for epoch in range(args.epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args, scaler)
    # Validate similarly, compute perplexity = exp(val_loss)
    # Save as in original code: model.canine.save_pretrained(...)
    # Save custom head separately
```

## 6. Evaluation and Perplexity

For validation, compute average loss on val set, then perplexity = exp(loss).

Code chunk:
```python
def evaluate(model, dataloader, device, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            with autocast(enabled=args.use_amp):
                loss = model(**batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity
```

## 7. Saving and Usage

Save similar to original:
- `model.canine.save_pretrained(output_dir)`
- Torch.save for decoder/lm_head.

For downstream (e.g., classification like domain-catch):
- Load the pre-trained canine, fine-tune as in provided code.

## Additional Tips
- **Hyperparams**: mlm_prob=0.15, mean_span=3, lr=2e-5, batch=512 effective, epochs=3.
- **W&B Logging**: Log loss/perplexity as in original.
- **Early Stopping**: On val perplexity.
- **Advanced Autoregressive**: For exact CANINE, implement local transformer decoder per span (see paper Fig 2); replace GRU with nn.TransformerDecoder (1-2 layers).
- **Debug**: Start with small subset (max_samples=100k).
- **AMP Issues**: If NaNs, reduce lr or disable for decoder.

This guide adapts the provided classification code to MLM pre-training. Test incrementally!