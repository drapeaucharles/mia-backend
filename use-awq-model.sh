#!/bin/bash

echo "Switching to AWQ model for better performance..."

cd ~/mia-gpu-miner || cd /opt/mia-gpu-miner
source venv/bin/activate

# Install AWQ support
echo "Installing AWQ libraries..."
pip install autoawq

# Test AWQ vs GPTQ speed
cat > test_awq_speed.py << 'EOF'
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing AWQ model (usually faster than GPTQ)...")

# Try AWQ model
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-OpenOrca-AWQ")
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-OpenOrca-AWQ",
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print("Model loaded! Testing speed...")

# Test prompt
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Warmup
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10)

# Speed test
times = []
for i in range(3):
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True
        )
    elapsed = time.time() - start
    times.append(elapsed)
    tokens_per_sec = 30 / elapsed
    print(f"Run {i+1}: {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")

avg_time = sum(times) / len(times)
avg_speed = 30 / avg_time
print(f"\nAverage: {avg_time:.2f}s ({avg_speed:.1f} tok/s)")

if avg_speed > 20:
    print("\n✅ AWQ model is fast! Use this instead of GPTQ.")
else:
    print("\n⚠️ Still slow. Let's try unquantized with 8-bit...")
EOF

python test_awq_speed.py

# Alternative: Use unquantized model with bitsandbytes
echo -e "\nAlternative: Testing 8-bit quantization..."
cat > test_8bit_speed.py << 'EOF'
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("Testing 8-bit quantization (often faster than GPTQ)...")

# Install bitsandbytes if needed
try:
    import bitsandbytes
except:
    print("Installing bitsandbytes...")
    import subprocess
    subprocess.check_call(["pip", "install", "bitsandbytes"])

# 8-bit config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained(
    "Open-Orca/Mistral-7B-OpenOrca",
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Testing speed...")
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Warmup
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=10)

# Test
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
elapsed = time.time() - start

print(f"8-bit model: {elapsed:.2f}s ({30/elapsed:.1f} tok/s)")
EOF

# Create miner with AWQ model
cat > use_awq_miner.py << 'EOF'
#!/usr/bin/env python3
"""Quick script to update miner to use AWQ model"""

content = open("mia_miner_unified.py", "r").read()

# Replace GPTQ model with AWQ
content = content.replace(
    'model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"',
    'model_name = "TheBloke/Mistral-7B-OpenOrca-AWQ"'
)

# Remove any GPTQ-specific code
content = content.replace("from auto_gptq import", "# from auto_gptq import")
content = content.replace("AutoGPTQForCausalLM", "AutoModelForCausalLM")

with open("mia_miner_awq.py", "w") as f:
    f.write(content)

print("Created mia_miner_awq.py using AWQ model")
EOF

python use_awq_miner.py

echo -e "\n${GREEN}Recommendations:${NC}"
echo "1. If AWQ is fast (20+ tok/s), use: ./mia_miner_awq.py"
echo "2. If 8-bit is fast, modify your miner to use bitsandbytes"
echo "3. Consider using vLLM server separately for best performance"
echo ""
echo "Your RTX 3090 should achieve 30-50 tok/s with proper setup!"