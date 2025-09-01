#!/bin/bash
# Install everything on /dev/sda1

echo "🔧 Setting up installation on /dev/sda1"
echo "====================================="

# Check if /dev/sda1 is mounted
MOUNT_POINT=$(df -h | grep /dev/sda1 | awk '{print $6}')

if [ -z "$MOUNT_POINT" ]; then
    echo "❌ /dev/sda1 is not mounted!"
    echo "Mounting to /mnt/sda1..."
    mkdir -p /mnt/sda1
    mount /dev/sda1 /mnt/sda1
    MOUNT_POINT="/mnt/sda1"
fi

echo "✅ /dev/sda1 is mounted at: $MOUNT_POINT"
echo "Disk space:"
df -h /dev/sda1

# Create directories on sda1
MINER_DIR="$MOUNT_POINT/qwen-awq-miner"
HF_CACHE="$MOUNT_POINT/huggingface"

echo -e "\nCreating directories on $MOUNT_POINT..."
mkdir -p "$MINER_DIR"
mkdir -p "$HF_CACHE"

# Set Hugging Face cache to sda1
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

# Create symlink from /data if needed
if [ "$MOUNT_POINT" != "/data" ]; then
    echo "Creating symlink from /data to $MOUNT_POINT..."
    rm -rf /data/qwen-awq-miner
    ln -sf "$MINER_DIR" /data/qwen-awq-miner
    ln -sf "$HF_CACHE" /data/huggingface
fi

# Now install the miner
cd "$MINER_DIR"

echo -e "\n📦 Installing miner on /dev/sda1..."

# Check if already installed
if [ -f "miner.py" ] && [ -d "venv" ]; then
    echo "✅ Miner already installed"
    echo "Fixing model to 7B..."
    sed -i 's/32B/7B/g' miner.py
else
    echo "Installing fresh..."
    
    # Install Python venv if needed
    apt-get update && apt-get install -y python3 python3-pip python3-venv
    
    # Create venv
    python3 -m venv venv
    source venv/bin/activate
    
    # Install packages
    pip install --upgrade pip wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install vllm transformers accelerate sentencepiece protobuf
    pip install flask waitress requests
    
    # Create miner.py with 7B model
    cat > miner.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import json
import time
import logging
import requests
import threading
import re
import socket
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams

# Force cache to our sda1 location
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/sda1/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/mnt/sda1/huggingface")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = None
model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 7B model!

def load_model():
    global model
    logger.info(f"Loading {model_name}...")
    logger.info(f"Cache directory: {os.environ.get('HF_HOME')}")
    
    try:
        model = LLM(
            model=model_name,
            dtype="half",
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            quantization="awq",
            download_dir=os.environ.get('HF_HOME')
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    if not tools:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    tool_descriptions = []
    for tool in tools:
        params_str = json.dumps(tool.get("parameters", {}), indent=2)
        tool_descriptions.append(f"""Function: {tool['name']}
Description: {tool.get('description', '')}
Parameters: {params_str}""")
    
    tools_text = "\n\n".join(tool_descriptions)
    ctx = context.get('system_prompt', '') if context else ''
    
    system_prompt = f"""You are a helpful assistant. {ctx}

You have access to these functions:
{tools_text}

To use a tool, respond with:
I'll use the appropriate tool.
{{"name": "tool_name", "parameters": {{"param": "value"}}}}"""
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def extract_tool_call(response: str) -> Optional[Dict]:
    if not response:
        return None
    
    # Look for JSON after "I'll use" or similar phrases
    json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    return None

from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        tools = data.get('tools')
        context = data.get('context')
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.7)
        
        formatted_prompt = format_prompt_with_tools(prompt, tools, context)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        start_time = time.time()
        outputs = model.generate([formatted_prompt], sampling_params)
        gen_time = time.time() - start_time
        
        response_text = outputs[0].outputs[0].text.strip()
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        tool_call = extract_tool_call(response_text) if tools else None
        
        return jsonify({
            'text': response_text,
            'response': response_text,
            'tool_call': tool_call,
            'tokens_generated': tokens_generated,
            'tokens_per_second': round(tokens_generated / gen_time, 1) if gen_time > 0 else 0,
            'model': model_name
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready' if model else 'loading', 'model': model_name})

if __name__ == "__main__":
    load_model()
    logger.info("Starting API server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

    # Create start script
    cat > start_miner.sh << EOF
#!/bin/bash
cd $MINER_DIR
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
python miner.py
EOF
    chmod +x start_miner.sh
fi

echo -e "\n✅ Installation complete on /dev/sda1!"
echo "=================================="
echo "Miner location: $MINER_DIR"
echo "HF cache: $HF_CACHE"
echo "Symlinks created in /data for compatibility"
echo ""
echo "To start:"
echo "  cd $MINER_DIR"
echo "  ./start_miner.sh"
echo ""
echo "Or use the symlink:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_miner.sh"