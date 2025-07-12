#!/usr/bin/env python3
"""
Test script to verify HuggingFace dataset loading functionality
"""

import sys
from train_mlm import load_and_prepare_data

def test_hf_dataset():
    """Test loading a HuggingFace dataset."""
    print("Testing HuggingFace dataset loading...")
    
    # Example with a small test dataset
    # Replace with actual dataset name when available
    dataset_name = "your-username/registered-domains-dataset"
    
    try:
        # Test with limited samples
        train_dataset, eval_dataset = load_and_prepare_data(
            dataset_name=dataset_name,
            num_samples=1000,
            streaming=True
        )
        print("✓ Successfully loaded HuggingFace dataset with streaming")
        
        # Show sample
        print("\nSample domains from dataset:")
        for i, example in enumerate(train_dataset.take(5)):
            print(f"  {i+1}. {example['text']}")
            
    except Exception as e:
        print(f"✗ Failed to load HuggingFace dataset: {e}")
        print("\nFalling back to local file test...")
        
        # Test local file loading
        try:
            train_dataset, eval_dataset = load_and_prepare_data(
                dataset_name="data/domains.txt",
                num_samples=None
            )
            print("✓ Successfully loaded local file")
            print(f"  Train examples: {len(train_dataset)}")
            print(f"  Eval examples: {len(eval_dataset)}")
        except Exception as e2:
            print(f"✗ Failed to load local file: {e2}")

if __name__ == "__main__":
    test_hf_dataset()