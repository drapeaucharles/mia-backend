#!/bin/bash

# Use ExLlama backend for fast GPTQ inference

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Switching to ExLlama for Fast GPTQ      ║${NC}"
echo -e "${GREEN}║   Should give 20-50 tok/s                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Navigate to miner directory
cd /data/mia-gpu-miner

# Activate venv
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
fi

# Install ExLlamaV2
echo -e "${YELLOW}Installing ExLlamaV2 for fast inference...${NC}"
pip install exllamav2

# Update the miner to use ExLlama backend
echo -e "${YELLOW}Updating miner configuration...${NC}"

# Backup current miner
cp mia_miner_unified.py mia_miner_unified.py.backup

# Create updated version with ExLlama backend
cat > update_miner.py << 'EOF'
import re

# Read current miner
with open('mia_miner_unified.py', 'r') as f:
    content = f.read()

# Find the AutoGPTQ loading section and update it
new_load_section = '''            try:
                from auto_gptq import AutoGPTQForCausalLM
                logger.info("Using AutoGPTQ with ExLlama backend for faster inference...")
                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device="cuda:0",
                    use_triton=False,
                    use_safetensors=True,
                    trust_remote_code=True,
                    inject_fused_attention=False,
                    disable_exllama=False,  # Enable ExLlama backend
                    exllama_config={"version": 2}  # Use ExLlamaV2
                )'''

# Replace the model loading section
pattern = r'try:\s+from auto_gptq.*?inject_fused_attention=False\s*\)'
content = re.sub(pattern, new_load_section, content, flags=re.DOTALL)

# Write updated miner
with open('mia_miner_unified.py', 'w') as f:
    f.write(content)

print("✓ Miner updated to use ExLlama backend")
EOF

python3 update_miner.py

# Alternative: Create a completely new optimized loader
echo -e "${YELLOW}Creating optimized model loader...${NC}"
cat > load_model_fast.py << 'EOF'
#!/usr/bin/env python3
"""Test fast model loading with ExLlama"""
import os
os.environ["HF_HOME"] = "/data/huggingface"

import torch
import time
from transformers import AutoTokenizer

print("Testing optimized model loading...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Open-Orca/Mistral-7B-OpenOrca",
    cache_dir="/data/huggingface"
)

# Try different loading methods
print("\n1. Trying ExLlamaV2...")
try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
    from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
    
    model_path = "/data/models/mistral-gptq"
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    
    model = ExLlamaV2(config)
    print("✓ ExLlamaV2 loaded successfully!")
    
    # Speed test
    print("Testing speed...")
    generator = ExLlamaV2BaseGenerator(model, tokenizer)
    
    start = time.time()
    output = generator.generate_simple("Hello, how are you?", max_new_tokens=50)
    elapsed = time.time() - start
    
    print(f"Generated in {elapsed:.2f}s ({50/elapsed:.1f} tok/s)")
    print("Output:", output[:100], "...")
    
except Exception as e:
    print(f"ExLlamaV2 failed: {e}")

print("\n2. Trying standard transformers with optimization...")
try:
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "/data/models/mistral-gptq",
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_flash_attention_2=True  # Try Flash Attention
    )
    
    # Test speed
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda:0")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    elapsed = time.time() - start
    
    print(f"Transformers: {elapsed:.2f}s ({50/elapsed:.1f} tok/s)")
    
except Exception as e:
    print(f"Transformers optimization failed: {e}")

print("\n3. Recommendations:")
print("- If ExLlamaV2 works: 30-50 tok/s expected")
print("- If only transformers works: 5-15 tok/s expected")
print("- For best speed, use ExLlamaV2 or vLLM")
EOF

python3 load_model_fast.py

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ ExLlama backend setup complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Now restart your miner:${NC}"
echo "  cd /data/mia-gpu-miner"
echo "  ./stop_miner.sh"
echo "  ./start_miner.sh"
echo ""
echo -e "${GREEN}This should improve speed from 3-4 tok/s to 20-50 tok/s!${NC}"