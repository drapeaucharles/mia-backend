#!/bin/bash

# MIA GPU Miner - ONE LINE INSTALLER
# Handles everything automatically including dependencies

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - One Line Installer      ║${NC}"
echo -e "${GREEN}║      Automatic Setup & Start              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system dependencies first
echo -e "${YELLOW}Installing system dependencies...${NC}"
apt update -qq
apt install -y python3-venv python3-pip python3-dev wget curl git 2>/dev/null || true

# Use /data for Vast.ai
INSTALL_DIR="/data/mia-gpu-miner"
VENV_DIR="/data/venv"

echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install wheel
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 11.8
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install vLLM 0.2.2 (version without puccinialin issue)
echo -e "${YELLOW}Installing vLLM 0.2.2...${NC}"
pip install vllm==0.2.2

# Install other dependencies
echo -e "${YELLOW}Installing other dependencies...${NC}"
pip install flask waitress requests aiohttp

# Download the production miner
echo -e "${YELLOW}Downloading miner script...${NC}"
wget -q -O mia_miner.py https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia_miner_production.py || {
    # If production miner fails, create simple one
    cat > mia_miner.py << 'MINER_EOF'
#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-miner')

app = Flask(__name__)

# Initialize model
logger.info("Loading model...")
llm = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", quantization="awq", gpu_memory_utilization=0.9)
logger.info("Model loaded!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        outputs = llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text
        
        return jsonify({'text': text, 'tokens_generated': len(outputs[0].outputs[0].token_ids)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def worker():
    """Get jobs from MIA backend"""
    backend_url = "https://mia-backend-production.up.railway.app"
    
    # Register
    try:
        resp = requests.post(f"{backend_url}/register_miner", json={"name": f"miner-{os.getpid()}"})
        if resp.status_code == 200:
            miner_id = resp.json()['miner_id']
            logger.info(f"Registered as miner {miner_id}")
        else:
            miner_id = 1
    except:
        miner_id = 1
    
    while True:
        try:
            # Get work
            work_resp = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30)
            if work_resp.status_code == 200:
                work = work_resp.json()
                if work and 'request_id' in work:
                    logger.info(f"Got job {work['request_id']}")
                    
                    # Generate
                    sampling_params = SamplingParams(temperature=0.7, max_tokens=work.get('max_tokens', 150))
                    outputs = llm.generate([work['prompt']], sampling_params)
                    
                    # Submit result
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': int(miner_id),
                        'request_id': work['request_id'],
                        'result': {'response': outputs[0].outputs[0].text}
                    })
        except Exception as e:
            logger.error(f"Worker error: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    # Start worker thread
    threading.Thread(target=worker, daemon=True).start()
    
    # Start Flask
    logger.info("Starting Flask server on port 8000...")
    serve(app, host='0.0.0.0', port=8000)
MINER_EOF
}

chmod +x mia_miner.py

# Create auto-start script
cat > auto_start.sh << 'EOF'
#!/bin/bash
cd /data/mia-gpu-miner
source /data/venv/bin/activate
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"
export CUDA_VISIBLE_DEVICES=0

# Start miner with auto-restart on failure
while true; do
    echo "Starting miner..."
    python3 mia_miner.py
    echo "Miner crashed, restarting in 5 seconds..."
    sleep 5
done
EOF
chmod +x auto_start.sh

# Start the miner immediately
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Installation complete! Starting miner...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}The miner will start automatically in 5 seconds...${NC}"
echo -e "${YELLOW}To view logs later: tail -f /data/miner.log${NC}"
echo ""

# Wait a moment
sleep 5

# Start miner in background
cd "$INSTALL_DIR"
source "$VENV_DIR/bin/activate"
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"

# Kill any existing miner
pkill -f mia_miner.py 2>/dev/null || true

# Start new miner
nohup python3 mia_miner.py > /data/miner.log 2>&1 &
MINER_PID=$!
echo $MINER_PID > /data/miner.pid

echo -e "${GREEN}✓ Miner started with PID $MINER_PID${NC}"
echo ""
echo "Commands:"
echo "  View logs:    tail -f /data/miner.log"
echo "  Check status: ps aux | grep mia_miner"
echo "  Stop miner:   kill \$(cat /data/miner.pid)"
echo "  Restart:      cd $INSTALL_DIR && ./auto_start.sh"
echo ""
echo -e "${GREEN}Your miner is now running and earning tokens!${NC}"