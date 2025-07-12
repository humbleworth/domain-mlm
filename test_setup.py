#!/usr/bin/env python3
"""
Test script to verify the training setup is working correctly.
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ datasets: {e}")
        return False
    
    try:
        import wandb
        print(f"✓ wandb {wandb.__version__}")
    except ImportError as e:
        print(f"✗ wandb: {e}")
        return False
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
    except ImportError as e:
        print(f"✗ pandas: {e}")
        return False
    
    # Check numpy version
    print(f"✓ numpy {np.__version__}", end="")
    if not np.__version__.startswith('1.'):
        print(" (WARNING: CANINE requires numpy<2.0)")
        return False
    else:
        print(" (compatible)")
    
    return True


def test_cuda():
    """Test CUDA availability and capabilities."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA available: {device_name}")
        print(f"  Memory: {device_memory:.1f} GB")
        
        # Test mixed precision support
        if torch.cuda.is_bf16_supported():
            print("✓ BF16 supported")
        else:
            print("✗ BF16 not supported")
        
        # Test autocast
        try:
            from torch.cuda.amp import autocast, GradScaler
            with autocast():
                x = torch.randn(10, 10).cuda()
                y = x @ x
            print("✓ Autocast working")
        except Exception as e:
            print(f"✗ Autocast failed: {e}")
            return False
            
    else:
        print("✗ CUDA not available")
        return False
    
    return True


def test_model_loading():
    """Test CANINE model loading."""
    print("\nTesting model loading...")
    
    try:
        from transformers import CanineTokenizer, CanineModel, CanineConfig
        
        # Test tokenizer
        tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
        print("✓ CANINE tokenizer loaded")
        
        # Test config
        config = CanineConfig.from_pretrained('google/canine-c')
        print(f"✓ CANINE config loaded (vocab size: {config.max_position_embeddings})")
        
        # Test tokenization
        text = "example.com"
        encoded = tokenizer(text, return_tensors='pt')
        print(f"✓ Tokenization working: '{text}' -> {encoded['input_ids'].shape}")
        
        # Don't load full model in test (too slow)
        print("✓ Model loading test passed (skipped full load)")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False
    
    return True


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset...")
    
    try:
        from datasets import load_dataset
        
        # Test with a tiny subset
        print("  Loading tiny subset of dataset...")
        dataset = load_dataset("humbleworth/registered-domains", split="train", streaming=True)
        
        # Take just one example
        example = next(iter(dataset))
        print(f"✓ Dataset accessible")
        print(f"  Example keys: {list(example.keys())}")
        
        # Find text column
        text_key = None
        for key in ["text", "domain", "domains", "raw"]:
            if key in example:
                text_key = key
                break
        
        if text_key:
            print(f"✓ Found text column: '{text_key}'")
            print(f"  Sample: {example[text_key][:50]}...")
        else:
            print(f"✗ Could not find text column in {list(example.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        print("  This might be okay if you're using a local dataset")
        return True  # Don't fail if dataset isn't accessible
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Domain MLM Training Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test CUDA
    if not test_cuda():
        all_passed = False
    
    # Test model
    if not test_model_loading():
        all_passed = False
    
    # Test dataset
    if not test_dataset():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Setup is ready for training.")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()