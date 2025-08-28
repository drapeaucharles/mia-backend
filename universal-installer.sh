#!/bin/bash

# MIA GPU Miner - Universal Installer
# Works on any system by detecting Python version

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MIA GPU Miner - Universal Installer     ║${NC}"
echo -e "${BLUE}║   Auto-detects Python Version             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Detect Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

# Install correct venv package
echo -e "${YELLOW}Installing Python $PYTHON_VERSION venv...${NC}"
apt update
apt install -y python${PYTHON_VERSION}-venv python3-pip wget curl git build-essential || {
    # If specific version fails, try generic
    apt install -y python3-venv python3-pip wget curl git build-essential
}

# Install pip if missing
which pip3 || apt install -y python3-pip

# Kill any existing miners
pkill -f miner 2>/dev/null || true

# Setup directories
INSTALL_DIR="/data/mia"
VENV_DIR="/data/venv"

echo -e "${YELLOW}Setting up in: $INSTALL_DIR${NC}"
rm -rf "$INSTALL_DIR" "$VENV_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment with explicit python
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR" --system-site-packages || {
    # Fallback: use virtualenv
    pip3 install virtualenv
    virtualenv "$VENV_DIR"
}

# Activate venv
source "$VENV_DIR/bin/activate"

# Ensure pip works
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
echo -e "${YELLOW}Installing AI packages...${NC}"
pip install transformers accelerate bitsandbytes flask waitress requests sentencepiece protobuf

# Create miner script
cat > miner.py << 'EOF'
#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/huggingface"
import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mia')

app = Flask(__name__)
backend_url = "https://mia-backend-production.up.railway.app"

# Load model
logger.info("Loading Qwen2.5-7B...")
model_id = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

tokenizer.pad_token = tokenizer.eos_token
logger.info("✓ Model loaded!")

def generate(prompt, max_tokens=150, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Store input length
    input_length = len(inputs['input_ids'][0])
    
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    gen_time = time.time() - start
    
    # Extract only the generated tokens
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response, len(generated_ids), gen_time

@app.route('/generate', methods=['POST'])
def api_generate():
    data = request.json
    response, tokens, gen_time = generate(data.get('prompt', ''), data.get('max_tokens', 150))
    return jsonify({'text': response, 'tokens_generated': tokens, 'tokens_per_second': tokens/gen_time})

def worker():
    # Register
    miner_id = 1
    try:
        resp = requests.post(f"{backend_url}/register_miner", json={"name": f"qwen-{os.getpid()}"})
        if resp.status_code == 200:
            miner_id = resp.json()['miner_id']
            logger.info(f"✓ Registered as miner {miner_id}")
    except: pass
    
    # Work loop
    while True:
        try:
            work = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30).json()
            if work and work.get('request_id'):
                logger.info(f"Job {work['request_id']}: {work.get('prompt', '')[:50]}...")
                response, tokens, gen_time = generate(work.get('prompt', ''), work.get('max_tokens', 150))
                logger.info(f"Response: {response[:50]}...")
                
                requests.post(f"{backend_url}/submit_result", json={
                    'miner_id': int(miner_id),
                    'request_id': work['request_id'],
                    'result': {'response': response, 'tokens_generated': tokens, 'processing_time': gen_time}
                })
                logger.info(f"✓ Submitted ({tokens} tokens at {tokens/gen_time:.1f} tok/s)")
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting server on port 8000...")
    serve(app, host='0.0.0.0', port=8000)
EOF

# Start miner
echo -e "${GREEN}Starting miner...${NC}"
nohup python miner.py > /data/miner.log 2>&1 &
echo $! > /data/miner.pid

echo -e "${GREEN}✓ Miner started! PID: $(cat /data/miner.pid)${NC}"
echo ""
echo "View logs: tail -f /data/miner.log"