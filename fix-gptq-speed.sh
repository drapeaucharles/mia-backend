#!/bin/bash

echo "Fixing GPTQ inference speed..."

cd ~/mia-gpu-miner || cd /opt/mia-gpu-miner
source venv/bin/activate

# Stop miner
pkill -f mia_miner_unified.py 2>/dev/null || true

echo "1. Reinstalling auto-gptq with optimized kernels..."
pip uninstall -y auto-gptq
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

echo -e "\n2. Installing optimized inference libraries..."
pip install --upgrade transformers accelerate
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash attention not available"

echo -e "\n3. Testing with different GPTQ backend..."
python3 << 'EOF'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("Testing GPTQ with optimizations...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

# Try loading with different settings
print("\nLoading model with optimizations...")
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
tokenizer.pad_token = tokenizer.eos_token

# Try with inject_fused_attention
try:
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-OpenOrca-GPTQ",
        device_map="cuda:0",  # Force single GPU
        trust_remote_code=True,
        revision="main",
        torch_dtype=torch.float16,
        inject_fused_attention=False,  # Sometimes helps with speed
        disable_exllama=False  # Use ExLlama backend if available
    )
except:
    print("Loading with standard settings...")
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-OpenOrca-GPTQ",
        device_map="cuda:0",
        trust_remote_code=True,
        revision="main"
    )

print(f"Model loaded on: {next(model.parameters()).device}")

# Quick benchmark
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

print("\nWarming up...")
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)

print("Running speed test...")
times = []
for i in range(3):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    gen_time = time.time() - start
    times.append(gen_time)
    print(f"  Run {i+1}: {gen_time:.2f}s ({50/gen_time:.1f} tokens/s)")

avg_time = sum(times) / len(times)
print(f"\nAverage: {avg_time:.2f}s ({50/avg_time:.1f} tokens/s)")

if avg_time > 3:
    print("\n⚠️ Still slow! Trying alternative fix...")
    print("You may need to:")
    print("1. pip install exllamav2")
    print("2. Use a different GPTQ model variant")
    print("3. Check for thermal throttling with: nvidia-smi -q -d PERFORMANCE")
else:
    print("\n✅ Speed is good!")
EOF

echo -e "\n4. Creating optimized miner script..."
# Create a version with explicit optimizations
sed -i 's/device_map="auto"/device_map="cuda:0"/g' mia_miner_unified.py 2>/dev/null || true

echo -e "\nIf still slow, run:"
echo "  nvidia-smi -q -d PERFORMANCE  # Check for throttling"
echo "  pip install exllamav2  # Try ExLlama v2 backend"