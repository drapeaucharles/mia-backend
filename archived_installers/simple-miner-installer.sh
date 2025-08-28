#!/bin/bash

# MIA GPU Miner - Simple Working Installer
# Uses standard model without AWQ to avoid CUDA issues

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - Simple Installer        ║${NC}"
echo -e "${GREEN}║      No AWQ - Maximum Compatibility       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt update -qq
apt install -y python3-venv python3-pip python3-dev wget curl git 2>/dev/null || true

# Use /data for Vast.ai
INSTALL_DIR="/data/mia-gpu-miner"
VENV_DIR="/data/venv-simple"

# Clean previous installation
rm -rf "$VENV_DIR"

echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create fresh virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch first
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install transformers and other deps (skip vLLM for now)
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install transformers accelerate flask waitress requests aiohttp

# Create simple miner without vLLM
cat > simple_miner.py << 'EOF'
#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

import time
import logging
import requests
import threading
import torch
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-simple')

app = Flask(__name__)

# Load model
logger.info("Loading Mistral-7B model...")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True  # 8-bit quantization for memory efficiency
)

logger.info("Model loaded successfully!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            gen_time = time.time() - start
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip()
        
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        
        logger.info(f"Generated {tokens_generated} tokens in {gen_time:.2f}s")
        
        return jsonify({
            'text': response,
            'tokens_generated': tokens_generated,
            'generation_time': gen_time
        })
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

def worker_loop():
    """Get jobs from MIA backend"""
    backend_url = "https://mia-backend-production.up.railway.app"
    miner_id = None
    
    # Register miner
    try:
        resp = requests.post(
            f"{backend_url}/register_miner",
            json={"name": f"simple-miner-{os.getpid()}"},
            timeout=10
        )
        if resp.status_code == 200:
            miner_id = resp.json()['miner_id']
            logger.info(f"Registered as miner {miner_id}")
    except Exception as e:
        logger.warning(f"Registration failed: {e}, using default ID")
        miner_id = 1
    
    while True:
        try:
            # Get work
            work_resp = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if work_resp.status_code == 200:
                work = work_resp.json()
                
                if work and 'request_id' in work:
                    logger.info(f"Got job {work['request_id']}")
                    
                    # Generate response
                    prompt = work.get('prompt', '')
                    max_tokens = work.get('max_tokens', 150)
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=work.get('temperature', 0.7),
                            do_sample=True
                        )
                    
                    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = generated[len(prompt):].strip()
                    
                    # Submit result
                    result_resp = requests.post(
                        f"{backend_url}/submit_result",
                        json={
                            'miner_id': int(miner_id),
                            'request_id': work['request_id'],
                            'result': {'response': response}
                        },
                        timeout=10
                    )
                    
                    if result_resp.status_code == 200:
                        logger.info("Result submitted successfully")
                    else:
                        logger.error(f"Failed to submit: {result_resp.status_code}")
                        
        except Exception as e:
            logger.error(f"Worker error: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    # Start worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    
    # Start Flask server
    logger.info("Starting Flask server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

chmod +x simple_miner.py

# Kill any existing miners
pkill -f mia_miner.py 2>/dev/null || true
pkill -f simple_miner.py 2>/dev/null || true

# Start the miner
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Installation complete! Starting simple miner...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""

nohup python3 simple_miner.py > /data/simple_miner.log 2>&1 &
MINER_PID=$!
echo $MINER_PID > /data/miner.pid

echo -e "${GREEN}✓ Simple miner started with PID $MINER_PID${NC}"
echo ""
echo "Commands:"
echo "  View logs:    tail -f /data/simple_miner.log"
echo "  Check status: ps aux | grep simple_miner"
echo "  Stop miner:   kill \$(cat /data/miner.pid)"
echo ""
echo -e "${GREEN}Your miner is now running!${NC}"