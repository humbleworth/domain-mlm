#!/usr/bin/env python3
"""
CANINE MLM Inference Example

This script demonstrates how to load and use the pretrained CANINE model
for masked language modeling on domain names.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from transformers import CanineForMaskedLM, CanineTokenizer

# Check numpy version for CANINE compatibility
if not np.__version__.startswith('1.'):
    print("Warning: CANINE requires numpy<2.0. Current version:", np.__version__)
    print("Please run: pip install 'numpy<2.0'")
    sys.exit(1)


def load_model_and_tokenizer(model_path: str = "./domain-canine-model") -> Tuple[CanineForMaskedLM, CanineTokenizer]:
    """
    Load the pretrained CANINE model and tokenizer.
    
    Args:
        model_path: Path to the saved model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_mlm.py first to train the model.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = CanineForMaskedLM.from_pretrained(model_path)
    tokenizer = CanineTokenizer.from_pretrained(model_path)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer


def create_masked_input(domain: str, mask_positions: List[int]) -> str:
    """
    Create a masked version of the domain.
    
    Args:
        domain: Original domain string
        mask_positions: List of character positions to mask
        
    Returns:
        Domain string with [MASK] tokens
    """
    domain_chars = list(domain)
    for pos in sorted(mask_positions, reverse=True):
        if 0 <= pos < len(domain_chars):
            domain_chars[pos] = "[MASK]"
    return "".join(domain_chars)


def predict_masked_characters(
    model: CanineForMaskedLM,
    tokenizer: CanineTokenizer,
    masked_domain: str,
    top_k: int = 5
) -> List[List[Tuple[str, float]]]:
    """
    Predict the masked characters in a domain.
    
    Args:
        model: CANINE model
        tokenizer: CANINE tokenizer
        masked_domain: Domain string with [MASK] tokens
        top_k: Number of top predictions to return
        
    Returns:
        List of predictions for each masked position
    """
    device = next(model.parameters()).device
    
    # Tokenize the masked domain
    inputs = tokenizer(masked_domain, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Find masked token positions
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    masked_indices = torch.where(inputs["input_ids"] == mask_token_id)[1]
    
    # Get predictions for each masked position
    predictions = []
    for idx in masked_indices:
        # Get top-k predictions
        probs = torch.softmax(logits[0, idx], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Convert to characters
        position_predictions = []
        for prob, token_id in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            # CANINE uses Unicode code points directly
            try:
                if token_id < tokenizer.vocab_size:
                    char = chr(token_id)
                    if char.isprintable():
                        position_predictions.append((char, float(prob)))
            except:
                continue
        
        predictions.append(position_predictions)
    
    return predictions


def demo_mlm_predictions():
    """Run demonstration of MLM predictions on various domains."""
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Example domains with masking
    examples = [
        ("example.com", [2]),           # ex[MASK]mple.com
        ("google.com", [0, 1]),         # [MASK][MASK]ogle.com
        ("amazon.com", [6]),            # amazon[MASK]com
        ("github.io", [3, 4, 5]),       # git[MASK][MASK][MASK].io
        ("stackoverflow.com", [5, 10]), # stack[MASK]verfl[MASK]w.com
    ]
    
    print("\n" + "="*60)
    print("CANINE Domain MLM Predictions")
    print("="*60 + "\n")
    
    for original_domain, mask_positions in examples:
        # Create masked input
        masked_domain = create_masked_input(original_domain, mask_positions)
        
        print(f"Original: {original_domain}")
        print(f"Masked:   {masked_domain}")
        
        # Get predictions
        predictions = predict_masked_characters(model, tokenizer, masked_domain, top_k=5)
        
        # Display predictions
        print("Predictions:")
        for i, (pos, preds) in enumerate(zip(mask_positions, predictions)):
            actual_char = original_domain[pos] if pos < len(original_domain) else "?"
            print(f"  Position {pos} (actual: '{actual_char}'):")
            for char, prob in preds[:3]:  # Show top 3
                status = "✓" if char == actual_char else " "
                print(f"    {status} '{char}': {prob:.3f}")
        
        print("-" * 60 + "\n")


def interactive_mode():
    """Interactive mode for custom domain predictions."""
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n" + "="*60)
    print("Interactive Domain MLM Mode")
    print("="*60)
    print("\nEnter domains with [MASK] tokens to see predictions.")
    print("Example: exam[MASK]le.com")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            masked_domain = input("Enter masked domain: ").strip()
            if masked_domain.lower() == 'quit':
                break
            
            if "[MASK]" not in masked_domain:
                print("Please include at least one [MASK] token in the domain.")
                continue
            
            # Get predictions
            predictions = predict_masked_characters(model, tokenizer, masked_domain, top_k=5)
            
            # Display predictions
            print(f"\nPredictions for: {masked_domain}")
            mask_count = masked_domain.count("[MASK]")
            for i, preds in enumerate(predictions):
                print(f"\n[MASK] #{i+1}:")
                for char, prob in preds:
                    print(f"  '{char}': {prob:.3f}")
            
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    """Main function to run demonstrations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CANINE MLM Inference Example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./domain-canine-model",
        help="Path to the pretrained model"
    )
    
    args = parser.parse_args()
    
    # Update model path if specified
    if args.model_path != "./domain-canine-model":
        global MODEL_PATH
        MODEL_PATH = args.model_path
    
    if args.interactive:
        interactive_mode()
    else:
        demo_mlm_predictions()


if __name__ == "__main__":
    main()