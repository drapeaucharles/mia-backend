#!/bin/bash

# Optimize inference speed for MIA miners
echo "Optimizing MIA miner for faster inference..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Find installation
if [ -d "$HOME/mia-gpu-miner" ]; then
    INSTALL_DIR="$HOME/mia-gpu-miner"
else
    INSTALL_DIR="/opt/mia-gpu-miner"
fi

cd "$INSTALL_DIR"
source venv/bin/activate

# Create optimized inference script
cat > "$INSTALL_DIR/test_inference_speed.py" << 'EOF'
#!/usr/bin/env python3
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Testing inference speed optimizations...")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Load model
print("\nLoading model...")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-OpenOrca-GPTQ",
    device_map="auto",
    trust_remote_code=True,
    revision="main"
)

print(f"Model loaded in {time.time() - start:.2f}s")
print(f"Model device: {next(model.parameters()).device}")

# Test different configurations
test_prompt = "Hello! How are you today?"
system_message = "You are MIA, a helpful AI assistant."
formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{test_prompt}<|im_end|>
<|im_start|>assistant
"""

print(f"\nTesting inference speed with prompt: '{test_prompt}'")

# Test 1: Basic generation
print("\n1. Basic generation (current setup):")
start = time.time()
inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
basic_time = time.time() - start
generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(f"Time: {basic_time:.2f}s")
print(f"Response: {response[:100]}...")

# Test 2: With use_cache=True (should be default but let's be explicit)
print("\n2. With explicit use_cache=True:")
start = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
cache_time = time.time() - start
print(f"Time: {cache_time:.2f}s")

# Test 3: Lower precision inference
print("\n3. With torch.cuda.amp.autocast:")
start = time.time()
with torch.cuda.amp.autocast():
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
amp_time = time.time() - start
print(f"Time: {amp_time:.2f}s")

# Test 4: Batch size test
print("\n4. Testing different max_new_tokens:")
for tokens in [20, 50, 100]:
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    print(f"  {tokens} tokens: {time.time() - start:.2f}s")

print("\n" + "="*50)
print("Recommendations:")
if basic_time > 5:
    print("- Inference is slower than expected for RTX 3090")
    print("- Should be 1-3 seconds for 50 tokens")
    print("- Check if model is fully on GPU")
    print("- Consider using Flash Attention if available")
else:
    print("- Inference speed is reasonable")

# Check memory usage
print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"GPU Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
EOF

chmod +x "$INSTALL_DIR/test_inference_speed.py"

echo -e "\n${GREEN}Running speed tests...${NC}"
python test_inference_speed.py

echo -e "\n${YELLOW}To optimize your miner:${NC}"
echo "1. Check if the model is fully loaded on GPU (not CPU)"
echo "2. The GPTQ model should use ~4-6GB VRAM"
echo "3. RTX 3090 should generate 50 tokens in 1-3 seconds"
echo ""
echo "If still slow, we may need to:"
echo "- Use a different GPTQ configuration"
echo "- Enable Flash Attention"
echo "- Check for CPU bottlenecks"