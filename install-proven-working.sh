#!/bin/bash

# MIA GPU Miner - PROVEN WORKING VERSIONS
# Uses exact package versions from when it worked

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - Proven Config           ║${NC}"
echo -e "${GREEN}║   Using July 2025 Working Versions        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt update -qq
apt install -y python3-venv python3-pip python3-dev wget curl git 2>/dev/null || true

# Use /data for Vast.ai
INSTALL_DIR="/data/mia-gpu-miner"
VENV_DIR="/data/venv-proven"

# Clean previous attempts
echo -e "${YELLOW}Cleaning previous installations...${NC}"
pkill -f mia_miner 2>/dev/null || true
rm -rf "$VENV_DIR"

echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create fresh virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip to specific version
pip install pip==23.2.1 setuptools==68.0.0 wheel==0.41.0

# Install EXACT PyTorch version that was working
echo -e "${YELLOW}Installing PyTorch 2.0.1 (proven working)...${NC}"
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM 0.1.7 - the version from July 2025 before issues
echo -e "${YELLOW}Installing vLLM 0.1.7 (last known good version)...${NC}"
pip install vllm==0.1.7

# Install other deps with specific versions
echo -e "${YELLOW}Installing other dependencies...${NC}"
pip install flask==2.3.2 waitress==2.1.2 requests==2.31.0 aiohttp==3.8.5

# Download the known working miner
echo -e "${YELLOW}Downloading proven miner script...${NC}"
cat > mia_miner_proven.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Proven AWQ Configuration"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-proven')

app = Flask(__name__)

# Initialize model with proven config
logger.info("Loading Mistral-7B-OpenOrca-AWQ...")
try:
    llm = LLM(
        model="TheBloke/Mistral-7B-OpenOrca-AWQ",
        quantization="awq",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_num_batched_tokens=4096,
        trust_remote_code=True
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load AWQ model: {e}")
    logger.info("Falling back to standard model...")
    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95
    )

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens
        )
        
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - start
        
        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)
        tps = tokens / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generated {tokens} tokens in {gen_time:.2f}s ({tps:.1f} tok/s)")
        
        return jsonify({
            'text': text,
            'tokens_generated': tokens,
            'tokens_per_second': tps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def worker():
    backend_url = "https://mia-backend-production.up.railway.app"
    
    # Register
    try:
        resp = requests.post(f"{backend_url}/register_miner", 
                           json={"name": f"proven-miner-{os.getpid()}"})
        miner_id = resp.json()['miner_id'] if resp.status_code == 200 else 1
        logger.info(f"Miner ID: {miner_id}")
    except:
        miner_id = 1
    
    while True:
        try:
            # Get work
            work_resp = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30)
            if work_resp.status_code == 200:
                work = work_resp.json()
                if work and 'request_id' in work:
                    logger.info(f"Processing job {work['request_id']}")
                    
                    # Generate
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=work.get('max_tokens', 150)
                    )
                    outputs = llm.generate([work['prompt']], sampling_params)
                    
                    # Submit
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': int(miner_id),
                        'request_id': work['request_id'],
                        'result': {'response': outputs[0].outputs[0].text}
                    })
        except Exception as e:
            logger.error(f"Worker error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting server on port 8000...")
    serve(app, host='0.0.0.0', port=8000)
EOF

chmod +x mia_miner_proven.py

# Kill any existing processes
pkill -f mia_miner 2>/dev/null || true

# Start the miner
echo ""
echo -e "${GREEN}Starting proven configuration miner...${NC}"
echo ""

nohup python3 mia_miner_proven.py > /data/proven_miner.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo -e "${GREEN}✓ Miner started with PID $PID${NC}"
echo ""
echo "View logs: tail -f /data/proven_miner.log"
echo ""
echo -e "${YELLOW}This uses the exact versions that worked in July 2025${NC}"
echo -e "${YELLOW}If AWQ fails, it will fallback to standard Mistral-7B${NC}"