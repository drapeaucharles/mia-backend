#!/bin/bash

# Fix model download - use Mistral 7B v0.2 (not gated)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Fixing model to use Mistral 7B Instruct v0.2${NC}"

cd /opt/mia-gpu-miner

# Update vllm_server.py to use v0.2
sed -i 's/Mistral-7B-Instruct-v0.1/Mistral-7B-Instruct-v0.2/g' vllm_server.py

# Update gpu_miner.py if it references the model
sed -i 's/Mistral-7B-Instruct-v0.1/Mistral-7B-Instruct-v0.2/g' gpu_miner.py

# Now download the correct model
source venv/bin/activate

echo -e "${YELLOW}Downloading Mistral 7B Instruct v0.2 (open access)...${NC}"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading model (14GB)...')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype='auto', low_cpu_mem_usage=True)
print('✓ Model downloaded successfully!')
"

echo -e "${GREEN}Fixed! Now you can run the miner.${NC}"