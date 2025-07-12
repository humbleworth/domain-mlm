#!/usr/bin/env python3
"""
Example script showing how to use the trained CANINE MLM model for inference.
"""

import os
import sys
import torch
import argparse
from transformers import CanineTokenizer, CanineModel, CanineConfig

# Add parent directory to path to import train_mlm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_mlm import CanineForMaskedLM


def load_model(model_path):
    """Load the fine-tuned CANINE MLM model."""
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Load tokenizer
    tokenizer = CanineTokenizer.from_pretrained(model_path)
    
    # Load model configuration and create model
    config = CanineConfig.from_pretrained(model_path)
    model = CanineForMaskedLM(config)
    
    # Load the CANINE encoder
    model.canine = CanineModel.from_pretrained(model_path)
    
    # Load the MLM head weights
    state_dict_path = os.path.join(model_path, "training_state.bin")
    if os.path.exists(state_dict_path):
        try:
            state_dict = torch.load(state_dict_path, map_location='cpu')
            if 'mlm_head_state_dict' in state_dict:
                model.mlm_head.load_state_dict(state_dict['mlm_head_state_dict'])
                print(f"Loaded MLM head from {state_dict_path}")
            elif 'best_mlm_head_state_dict' in state_dict:
                model.mlm_head.load_state_dict(state_dict['best_mlm_head_state_dict'])
                print(f"Loaded best MLM head from {state_dict_path}")
            else:
                print(f"Warning: No MLM head state found in {state_dict_path}")
        except Exception as e:
            print(f"Warning: Failed to load MLM head: {e}")
    else:
        print(f"Warning: No training_state.bin found at {model_path}")
        print("Using randomly initialized MLM head")
    
    model.eval()
    return model, tokenizer


def predict_masked_tokens(model, tokenizer, text, mask_token="[MASK]", device='cpu'):
    """Predict masked tokens in a domain name."""
    # Use the same mask token as training (0xE000 - private use area)
    masked_text = text.replace(mask_token, chr(0xE000))
    
    # Tokenize
    inputs = tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # Find masked positions
    masked_positions = (inputs['input_ids'] == 0xE000).nonzero(as_tuple=True)
    
    results = []
    for batch_idx, seq_idx in zip(masked_positions[0], masked_positions[1]):
        # Get top 5 predictions for this position
        logits = predictions[batch_idx, seq_idx]
        top_k = torch.topk(logits, k=5)
        
        position_results = []
        for score, token_id in zip(top_k.values, top_k.indices):
            # Convert token ID to character
            try:
                if 32 <= token_id < 127:  # Printable ASCII range
                    char = chr(token_id)
                elif token_id == 0:
                    char = "<PAD>"
                elif token_id == 57344:
                    char = "<CLS>"
                elif token_id == 57345:
                    char = "<SEP>"
                else:
                    char = f"<{token_id}>"
            except ValueError:
                char = f"<{token_id}>"
            
            position_results.append({
                'char': char,
                'score': score.item(),
                'prob': torch.softmax(logits, dim=-1)[token_id].item()
            })
        
        results.append(position_results)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Use trained CANINE MLM for domain prediction")
    parser.add_argument("--model_path", type=str, default="./domain-canine-model/best-model",
                        help="Path to the trained model directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)
    model.to(args.device)
    
    # Example predictions
    examples = [
        "goo[MASK]le.com",
        "face[MASK]ook.com",
        "ama[MASK]on.com",
        "[MASK]mazon.com",
        "example.[MASK]om",
        "test-[MASK]omain.net",
        "my[MASK]website.org",
        "git[MASK]ub.com",
        "stack[MASK]verflow.com"
    ]
    
    print("\nExample predictions:")
    print("-" * 80)
    
    for example in examples:
        print(f"\nInput: {example}")
        results = predict_masked_tokens(model, tokenizer, example, device=args.device)
        
        for i, position_results in enumerate(results):
            print(f"  Position {i+1} predictions:")
            for j, pred in enumerate(position_results[:3]):  # Show top 3
                print(f"    {j+1}. '{pred['char']}' (score: {pred['score']:.2f}, prob: {pred['prob']:.2%})")
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("Interactive mode - Enter domains with [MASK] tokens (or 'quit' to exit)")
        print("Example: goo[MASK]le.com")
        print("="*80)
        
        while True:
            try:
                text = input("\nEnter domain: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if '[MASK]' not in text:
                    print("Please include at least one [MASK] token in your input")
                    continue
                
                results = predict_masked_tokens(model, tokenizer, text, device=args.device)
                
                print(f"\nPredictions for: {text}")
                for i, position_results in enumerate(results):
                    print(f"\n  Masked position {i+1}:")
                    for j, pred in enumerate(position_results):
                        print(f"    {j+1}. '{pred['char']}' (score: {pred['score']:.2f}, prob: {pred['prob']:.2%})")
                        
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()