#!/usr/bin/env python3
"""Verify Mistral v0.2 model is accessible without authentication"""

import sys

try:
    from transformers import AutoTokenizer
    print("✓ Transformers library installed")
except ImportError:
    print("✗ Transformers not installed. Install with: pip install transformers")
    sys.exit(1)

print("\nTesting Mistral-7B-Instruct-v0.2 access...")
print("This will check if the model can be accessed without authentication.")
print("(It won't download the full model, just verify access)\n")

try:
    # Try to get tokenizer config without downloading everything
    tokenizer = AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.2',
        use_fast=True,
        local_files_only=False
    )
    print("✓ SUCCESS: Mistral-7B-Instruct-v0.2 is accessible!")
    print("  The model is NOT gated and can be used by miners.")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    
except Exception as e:
    if "gated" in str(e).lower() or "403" in str(e):
        print("✗ ERROR: Model appears to be gated")
        print(f"  Error: {e}")
        print("\n  This suggests v0.2 might also be gated now.")
    else:
        print("✗ ERROR: Failed to access model")
        print(f"  Error: {e}")
        print("\n  Check your internet connection and try again.")