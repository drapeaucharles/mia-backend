#!/usr/bin/env python3
"""
Optimization script for Qwen2.5-7B on RTX 3090
Target: 60+ tokens/second
"""

import torch
from vllm import LLM, SamplingParams
import time

def test_optimal_config():
    """Test optimal configuration for RTX 3090"""
    
    # Enable CUDA optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    print("Testing optimal vLLM configuration for RTX 3090...")
    
    # Optimal configuration for RTX 3090 (24GB VRAM)
    model = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",  # Or your specific model path
        quantization="awq",  # AWQ is faster than GPTQ
        dtype="half",  # FP16 for best speed
        gpu_memory_utilization=0.95,  # Use more GPU memory
        max_model_len=4096,  # Larger context for better batching
        max_num_seqs=16,  # More concurrent sequences
        trust_remote_code=True,
        enforce_eager=True,  # Disable graph compilation for consistent speed
        tensor_parallel_size=1  # Single GPU
    )
    
    # Fast sampling parameters (greedy decoding)
    fast_params = SamplingParams(
        temperature=0.0,  # Greedy decoding is fastest
        top_p=1.0,
        max_tokens=200,
        repetition_penalty=1.0  # Disable for speed
    )
    
    # Test single request
    print("\nTesting single request speed...")
    prompt = "Explain quantum computing in simple terms."
    start = time.time()
    outputs = model.generate([prompt], fast_params)
    elapsed = time.time() - start
    
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / elapsed
    print(f"Single request: {tokens} tokens in {elapsed:.2f}s = {speed:.1f} tokens/sec")
    
    # Test batch processing
    print("\nTesting batch processing...")
    prompts = [
        "What is machine learning?",
        "Explain artificial intelligence.",
        "What are neural networks?",
        "How does deep learning work?"
    ]
    
    start = time.time()
    outputs = model.generate(prompts, fast_params)
    elapsed = time.time() - start
    
    total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    speed = total_tokens / elapsed
    print(f"Batch of 4: {total_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tokens/sec")
    
    # Test with your restaurant prompt
    print("\nTesting restaurant query...")
    restaurant_prompt = """You are Maria, a friendly server at Bella Vista Restaurant.

Our complete menu:

Main Courses:
• Ribeye Steak ($38.99) [beef, herbs, butter]
• Grilled Chicken ($24.99) [chicken breast, herbs, lemon]
• Lamb Chops ($35.99) [lamb, rosemary, garlic]
• Beef Medallions ($32.99) [beef tenderloin, mushroom sauce]

Customer: I want to eat steak or things similar
Assistant:"""

    start = time.time()
    outputs = model.generate([restaurant_prompt], fast_params)
    elapsed = time.time() - start
    
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / elapsed
    print(f"Restaurant query: {tokens} tokens in {elapsed:.2f}s = {speed:.1f} tokens/sec")
    print(f"Response: {outputs[0].outputs[0].text}")
    
    return model

if __name__ == "__main__":
    # Set environment for optimal performance
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    print("RTX 3090 Optimization Test for Qwen2.5-7B")
    print("==========================================")
    
    model = test_optimal_config()
    
    print("\n\nRecommended changes for production:")
    print("1. Update mia_miner_production.py with these settings")
    print("2. Use temperature=0 for maximum speed")
    print("3. Set gpu_memory_utilization=0.95")
    print("4. Increase max_num_seqs to 16")
    print("5. Use single-threaded server (threads=1)")