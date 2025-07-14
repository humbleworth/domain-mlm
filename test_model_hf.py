#!/usr/bin/env python3
"""Test the trained CANINE MLM model using HuggingFace standard approach."""

import torch
from transformers import CanineTokenizer, pipeline
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_mlm import CanineForMaskedLM

def test_with_pipeline():
    """Test using HuggingFace's fill-mask pipeline."""
    model_path = "./models/domain-canine-model/final-model"
    print(f"Loading model from: {model_path}")
    
    # Load tokenizer
    tokenizer = CanineTokenizer.from_pretrained(model_path)
    
    # Load model with custom head
    model = CanineForMaskedLM.from_pretrained("google/canine-c")
    model.canine = model.canine.from_pretrained(model_path)
    
    # Load MLM head weights
    training_state_path = os.path.join(model_path, "training_state.bin")
    if os.path.exists(training_state_path):
        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        model.mlm_head.load_state_dict(state["mlm_head_state_dict"])
        print("Loaded MLM head weights\n")
    
    # Create fill-mask pipeline
    mask_token = chr(0xE000)  # CANINE's mask token
    
    # Test domains
    test_domains = [
        f"goo{mask_token}le.com",
        f"face{mask_token}ook.com", 
        f"twi{mask_token}ter.com",
        f"{mask_token}mazon.com",
        f"micro{mask_token}oft.com"
    ]
    
    print("Testing domain completion:")
    print("-" * 50)
    
    for domain in test_domains:
        print(f"\nInput: {domain.replace(mask_token, '[MASK]')}")
        
        # Get model predictions
        inputs = tokenizer(domain, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Find mask position
        mask_pos = (inputs.input_ids == 0xE000).nonzero(as_tuple=True)[1][0]
        
        # Get top prediction
        top_token = predictions[0, mask_pos].argmax()
        predicted_char = chr(top_token.item()) if top_token < 0x110000 else '?'
        
        # Get probability
        probs = torch.softmax(predictions[0, mask_pos], dim=-1)
        top_prob = probs[top_token].item()
        
        filled_domain = domain.replace(mask_token, predicted_char)
        print(f"Prediction: {filled_domain} (confidence: {top_prob:.2%})")

def test_batch_prediction():
    """Test batch prediction for efficiency."""
    model_path = "./models/domain-canine-model/final-model"
    
    # Load model
    tokenizer = CanineTokenizer.from_pretrained(model_path)
    model = CanineForMaskedLM.from_pretrained("google/canine-c")
    model.canine = model.canine.from_pretrained(model_path)
    
    # Load MLM head
    training_state_path = os.path.join(model_path, "training_state.bin")
    if os.path.exists(training_state_path):
        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        model.mlm_head.load_state_dict(state["mlm_head_state_dict"])
    
    model.eval()
    
    # Test generating domain names
    print("\n" + "-" * 50)
    print("\nGenerating new domain patterns:")
    
    mask = chr(0xE000)
    patterns = [
        f"{mask}{mask}{mask}.com",
        f"{mask}{mask}{mask}{mask}.net",
        f"web{mask}{mask}{mask}.org",
        f"{mask}{mask}shop.com",
        f"best{mask}{mask}{mask}.io"
    ]
    
    for pattern in patterns:
        inputs = tokenizer(pattern, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits
        
        # Fill each mask
        result = list(pattern)
        mask_positions = [i for i, c in enumerate(pattern) if c == mask]
        
        for i, char_idx in enumerate(mask_positions):
            # Get token position in input_ids
            token_pos = char_idx + 1  # +1 for [CLS] token
            if token_pos < predictions.shape[1]:
                top_pred = predictions[0, token_pos].argmax()
                pred_char = chr(top_pred.item()) if top_pred < 0x110000 else '?'
                result[char_idx] = pred_char
        
        print(f"Pattern: {pattern.replace(mask, '[M]')} → {''.join(result)}")

if __name__ == "__main__":
    test_with_pipeline()
    test_batch_prediction()