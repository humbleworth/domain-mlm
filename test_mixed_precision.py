#!/usr/bin/env python3
"""
Test script to verify mixed precision training functionality
"""

import torch
import sys

def test_mixed_precision():
    """Test mixed precision capabilities."""
    print("Testing Mixed Precision Support...")
    print("-" * 50)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA not available - mixed precision requires GPU")
        return
    
    # Check FP16 support
    print("\nFP16 Support:")
    try:
        # Test FP16 computation
        a = torch.randn(100, 100, dtype=torch.float16, device='cuda')
        b = torch.randn(100, 100, dtype=torch.float16, device='cuda')
        c = torch.matmul(a, b)
        print("✓ FP16 operations supported")
    except Exception as e:
        print(f"✗ FP16 not supported: {e}")
    
    # Check BF16 support
    print("\nBF16 Support:")
    if torch.cuda.is_bf16_supported():
        try:
            # Test BF16 computation
            a = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
            c = torch.matmul(a, b)
            print("✓ BF16 operations supported (Ampere+ GPU detected)")
        except Exception as e:
            print(f"✗ BF16 computation failed: {e}")
    else:
        print("✗ BF16 not supported (requires Ampere+ GPU)")
    
    # Test autocast
    print("\nAutocast Support:")
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        # Simple model test
        model = torch.nn.Linear(10, 10).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        
        # Test forward pass with autocast
        x = torch.randn(32, 10).cuda()
        with autocast():
            y = model(x)
            loss = y.mean()
        
        # Test backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✓ Autocast and GradScaler working correctly")
    except Exception as e:
        print(f"✗ Autocast test failed: {e}")
    
    print("\n" + "-" * 50)
    print("Mixed precision setup summary:")
    print(f"- FP16: {'✓ Available' if torch.cuda.is_available() else '✗ Not available'}")
    print(f"- BF16: {'✓ Available' if torch.cuda.is_bf16_supported() else '✗ Not available'}")
    print("\nRecommendation:")
    if torch.cuda.is_bf16_supported():
        print("Use --bf16 flag for best stability and performance")
    else:
        print("Use --fp16 flag for mixed precision training")

if __name__ == "__main__":
    test_mixed_precision()