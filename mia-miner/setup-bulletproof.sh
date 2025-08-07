#!/bin/bash

# MIA GPU Miner Bulletproof Setup - Works on fresh Ubuntu 20.04/22.04
# Handles ALL edge cases and dependency issues

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}"
echo "███╗   ███╗██╗ █████╗     ███╗   ███╗██╗███╗   ██╗███████╗██████╗ "
echo "████╗ ████║██║██╔══██╗    ████╗ ████║██║████╗  ██║██╔════╝██╔══██╗"
echo "██╔████╔██║██║███████║    ██╔████╔██║██║██╔██╗ ██║█████╗  ██████╔╝"
echo "██║╚██╔╝██║██║██╔══██║    ██║╚██╔╝██║██║██║╚██╗██║██╔══╝  ██╔══██╗"
echo "██║ ╚═╝ ██║██║██║  ██║    ██║ ╚═╝ ██║██║██║ ╚████║███████╗██║  ██║"
echo "╚═╝     ╚═╝╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo "Bulletproof GPU Miner Setup v2.0"
echo "================================"

# Detect if we need sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
    echo "Running as root"
else
    SUDO_CMD="sudo"
    echo "Running as user, will use sudo"
fi

# Step 1: Update system
echo -e "\n${YELLOW}[1/8] Updating system packages...${NC}"
$SUDO_CMD apt-get update -qq
$SUDO_CMD apt-get upgrade -y -qq

# Step 2: Install NVIDIA drivers if needed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "\n${YELLOW}[2/8] Installing NVIDIA drivers...${NC}"
    # Add NVIDIA PPA
    $SUDO_CMD apt-get install -y -qq software-properties-common
    $SUDO_CMD add-apt-repository -y ppa:graphics-drivers/ppa
    $SUDO_CMD apt-get update -qq
    
    # Install latest stable driver
    $SUDO_CMD apt-get install -y -qq nvidia-driver-535
    
    echo -e "${YELLOW}NVIDIA drivers installed. Reboot required after setup!${NC}"
    NEEDS_REBOOT=1
else
    echo -e "\n${GREEN}[2/8] NVIDIA drivers already installed${NC}"
    nvidia-smi || true
fi

# Step 3: Install system dependencies (handle different Ubuntu versions)
echo -e "\n${YELLOW}[3/8] Installing system dependencies...${NC}"

# First install software-properties-common to add repositories
$SUDO_CMD apt-get install -y -qq software-properties-common

# Add deadsnakes PPA for Python versions
$SUDO_CMD add-apt-repository -y ppa:deadsnakes/ppa
$SUDO_CMD apt-get update -qq

# Install Python 3.10 (works on both Ubuntu 20.04 and 22.04)
$SUDO_CMD apt-get install -y -qq \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils \
    python3-pip git curl wget build-essential \
    libssl-dev libffi-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev

# Install CUDA toolkit components
$SUDO_CMD apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || true

# Step 4: Create installation directory
INSTALL_DIR="/opt/mia-gpu-miner"
echo -e "\n${YELLOW}[4/8] Setting up installation directory...${NC}"
$SUDO_CMD rm -rf "$INSTALL_DIR"
$SUDO_CMD mkdir -p "$INSTALL_DIR"
$SUDO_CMD chown -R $USER:$USER "$INSTALL_DIR" 2>/dev/null || true
cd "$INSTALL_DIR"

# Step 5: Create Python environment with Python 3.10
echo -e "\n${YELLOW}[5/8] Creating Python 3.10 environment...${NC}"
python3.10 -m venv venv
source venv/bin/activate

# Ensure pip is updated
python -m pip install --upgrade pip setuptools wheel

# Step 6: Install PyTorch and dependencies
echo -e "\n${YELLOW}[6/8] Installing AI packages (this may take 5-10 minutes)...${NC}"

# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM and its dependencies separately to avoid conflicts
pip install numpy==1.24.3
pip install vllm==0.2.7

# Install other packages
pip install \
    transformers==4.36.2 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    requests==2.31.0 \
    huggingface-hub==0.20.3 \
    accelerate==0.26.1 \
    sentencepiece==0.1.99 \
    protobuf==4.25.2 \
    safetensors==0.4.1 \
    psutil==5.9.6

# Step 7: Create the server and miner scripts
echo -e "\n${YELLOW}[7/8] Creating server and miner scripts...${NC}"

# Create vLLM server
cat > "$INSTALL_DIR/vllm_server.py" << 'EOF'
#!/usr/bin/env python3
"""
vLLM Server with automatic model fallback
Tries multiple open models if one fails
"""

import os
import sys
import torch
import logging
from typing import Optional
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    model: str

app = FastAPI(title="MIA vLLM Server")

# Model fallback list
MODELS = [
    "Open-Orca/Mistral-7B-OpenOrca",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "openchat/openchat_3.5"
]

llm: Optional[LLM] = None
current_model: str = ""

def load_model():
    """Try to load models from the fallback list"""
    global llm, current_model
    
    for model_name in MODELS:
        try:
            logger.info(f"Attempting to load {model_name}...")
            
            # Check GPU
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Load model
            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                download_dir="/opt/mia-gpu-miner/models"
            )
            
            current_model = model_name
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            continue
    
    logger.error("Failed to load any model!")
    return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.error("No models could be loaded!")
        sys.exit(1)

@app.get("/")
async def root():
    return {"status": "ready", "model": current_model}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": current_model,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt based on model
        if "OpenOrca" in current_model or "OpenHermes" in current_model:
            formatted_prompt = f"<|im_start|>user\n{request.prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "Nous-Hermes" in current_model:
            formatted_prompt = f"### Instruction:\n{request.prompt}\n\n### Response:\n"
        elif "openchat" in current_model:
            formatted_prompt = f"GPT4 Correct User: {request.prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        else:
            formatted_prompt = request.prompt
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        
        outputs = llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated,
            model=current_model
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
EOF

# Create miner client
cat > "$INSTALL_DIR/gpu_miner.py" << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner Client
"""

import os
import sys
import time
import json
import requests
import logging
import socket
import psutil
from datetime import datetime

# Configuration
MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend.up.railway.app")
VLLM_SERVER_URL = "http://localhost:8000"
WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "0x" + "0" * 40)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MIAMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        
    def wait_for_vllm(self, max_retries=60):
        """Wait for vLLM server"""
        logger.info("Waiting for vLLM server...")
        for i in range(max_retries):
            try:
                r = requests.get(f"{VLLM_SERVER_URL}/health", timeout=5)
                if r.status_code == 200:
                    logger.info(f"vLLM ready with model: {r.json()['model']}")
                    return True
            except:
                pass
            time.sleep(5)
        return False
    
    def register(self):
        """Register with backend"""
        try:
            # Get public IP
            public_ip = requests.get('https://api.ipify.org', timeout=10).text
            
            data = {
                "wallet_address": WALLET_ADDRESS,
                "endpoint_url": f"http://{public_ip}:8000",
                "model": "Mistral-7B-OpenOrca",
                "max_tokens": 4096
            }
            
            r = self.session.post(f"{MIA_BACKEND_URL}/register_miner", json=data)
            r.raise_for_status()
            
            self.miner_id = r.json()["miner_id"]
            logger.info(f"Registered! Miner ID: {self.miner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def run(self):
        """Main loop"""
        if not self.wait_for_vllm():
            logger.error("vLLM server not responding")
            return
        
        while not self.register():
            time.sleep(30)
        
        logger.info("Starting work loop...")
        
        while True:
            try:
                # Poll for work
                r = self.session.get(
                    f"{MIA_BACKEND_URL}/miner/{self.miner_id}/work",
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    if work.get("request_id"):
                        logger.info(f"Processing request {work['request_id']}")
                        
                        # Process
                        try:
                            gen_r = self.session.post(
                                f"{VLLM_SERVER_URL}/generate",
                                json={
                                    "prompt": work["prompt"],
                                    "max_tokens": work.get("max_tokens", 500),
                                    "temperature": work.get("temperature", 0.7)
                                },
                                timeout=120
                            )
                            gen_r.raise_for_status()
                            result = gen_r.json()
                            
                            # Submit
                            self.session.post(
                                f"{MIA_BACKEND_URL}/miner/{self.miner_id}/submit",
                                json={
                                    "request_id": work["request_id"],
                                    "result": {
                                        "success": True,
                                        "response": result["text"],
                                        "tokens_generated": result["tokens_generated"]
                                    }
                                }
                            )
                            logger.info(f"Completed {work['request_id']}")
                            
                        except Exception as e:
                            logger.error(f"Processing error: {e}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(10)
            
            time.sleep(5)

if __name__ == "__main__":
    miner = MIAMiner()
    miner.run()
EOF

# Create start script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd /opt/mia-gpu-miner
source venv/bin/activate

# Kill existing
pkill -f vllm_server.py || true
pkill -f gpu_miner.py || true
sleep 2

# Start vLLM
echo "Starting vLLM server..."
python vllm_server.py > vllm.log 2>&1 &
echo "Waiting 30s for model load..."
sleep 30

# Start miner
echo "Starting miner..."
python gpu_miner.py 2>&1 | tee miner.log
EOF

chmod +x "$INSTALL_DIR/start.sh"

# Create systemd service
$SUDO_CMD cat > /etc/systemd/system/mia-miner.service << EOF
[Unit]
Description=MIA GPU Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mia-gpu-miner
ExecStart=/opt/mia-gpu-miner/start.sh
Restart=always
RestartSec=30
Environment="WALLET_ADDRESS=$WALLET_ADDRESS"

[Install]
WantedBy=multi-user.target
EOF

# Step 8: Download a model
echo -e "\n${YELLOW}[8/8] Pre-downloading AI model (this will take 10-20 minutes)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

python3 << 'PYEOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.makedirs("/opt/mia-gpu-miner/models", exist_ok=True)

# Try to download the first available model
models = [
    "Open-Orca/Mistral-7B-OpenOrca",
    "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "teknium/OpenHermes-2.5-Mistral-7B"
]

for model in models:
    try:
        print(f"Attempting to download {model}...")
        tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/opt/mia-gpu-miner/models")
        print(f"✓ Downloaded {model} tokenizer")
        
        # Just download, don't load into memory
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir="/opt/mia-gpu-miner/models",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(f"✓ Downloaded {model} successfully!")
        break
    except Exception as e:
        print(f"Failed to download {model}: {e}")
        continue
PYEOF

# Fix permissions
$SUDO_CMD chown -R $USER:$USER "$INSTALL_DIR"

# Setup complete
echo -e "\n${GREEN}✓ Setup complete!${NC}"
echo -e "\nTo start the miner:"
echo -e "  ${YELLOW}sudo systemctl daemon-reload${NC}"
echo -e "  ${YELLOW}sudo systemctl enable mia-miner${NC}"
echo -e "  ${YELLOW}sudo systemctl start mia-miner${NC}"
echo -e "\nTo check logs:"
echo -e "  ${YELLOW}sudo journalctl -u mia-miner -f${NC}"

if [ "${NEEDS_REBOOT}" == "1" ]; then
    echo -e "\n${RED}IMPORTANT: Reboot required for NVIDIA drivers!${NC}"
    echo -e "Run: ${YELLOW}sudo reboot${NC}"
fi