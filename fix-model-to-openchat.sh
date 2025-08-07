#!/bin/bash
# Emergency fix: Switch to OpenChat 3.5 (open alternative to Mistral)

echo "Fixing model access issue..."
echo "Switching to OpenChat 3.5 (7B, similar performance to Mistral)"

# Update vLLM server
if [ -f "vllm_server.py" ]; then
    sed -i 's|mistralai/Mistral-7B-Instruct-v0.[12]|openchat/openchat_3.5|g' vllm_server.py
    echo "✓ Updated vLLM server"
fi

# Update GPU miner
if [ -f "gpu_miner.py" ]; then
    sed -i 's|mistralai/Mistral-7B-Instruct-v0.[12]|openchat/openchat_3.5|g' gpu_miner.py
    echo "✓ Updated GPU miner"
fi

# Download the new model
echo ""
echo "Downloading OpenChat 3.5 model (13GB)..."
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading OpenChat 3.5...")
tokenizer = AutoTokenizer.from_pretrained('openchat/openchat_3.5')
model = AutoModelForCausalLM.from_pretrained('openchat/openchat_3.5', torch_dtype='auto', low_cpu_mem_usage=True)
print("✓ Model downloaded successfully!")
EOF

echo ""
echo "✓ Fixed! OpenChat 3.5 is now configured."
echo "Please restart your miner."