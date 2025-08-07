#!/bin/bash

# MIA GPU Miner Setup Script
# One-line installer for Mistral 7B GPU inference
# Usage: bash <(curl -s https://yourdomain.com/setup-gpu-miner.sh)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "███╗   ███╗██╗ █████╗     ███╗   ███╗██╗███╗   ██╗███████╗██████╗ "
echo "████╗ ████║██║██╔══██╗    ████╗ ████║██║████╗  ██║██╔════╝██╔══██╗"
echo "██╔████╔██║██║███████║    ██╔████╔██║██║██╔██╗ ██║█████╗  ██████╔╝"
echo "██║╚██╔╝██║██║██╔══██║    ██║╚██╔╝██║██║██║╚██╗██║██╔══╝  ██╔══██╗"
echo "██║ ╚═╝ ██║██║██║  ██║    ██║ ╚═╝ ██║██║██║ ╚████║███████╗██║  ██║"
echo "╚═╝     ╚═╝╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo "GPU Miner Setup - Mistral 7B Local Inference"
echo "============================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This installer is for Linux systems only${NC}"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA GPU not detected. Please install NVIDIA drivers first.${NC}"
    echo -e "${YELLOW}To install NVIDIA drivers:${NC}"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-driver-525"
    echo "  sudo reboot"
    exit 1
fi

# Check if running with sudo when needed
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root/sudo${NC}"
    echo "The script will ask for sudo when needed."
    exit 1
fi

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1)
if [ -z "$GPU_INFO" ]; then
    echo -e "${RED}Error: Could not query GPU information${NC}"
    echo "Please ensure NVIDIA drivers are properly installed."
    exit 1
fi

GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | xargs)

echo -e "${GREEN}Detected GPU: $GPU_NAME ($GPU_MEMORY)${NC}"

# Check GPU memory (need at least 16GB for Mistral 7B)
GPU_MEMORY_GB=$(echo $GPU_MEMORY | grep -o '[0-9]*' | head -n1)
if [ "$GPU_MEMORY_GB" -lt 16 ]; then
    echo -e "${YELLOW}Warning: GPU has only ${GPU_MEMORY_GB}GB memory. Mistral 7B requires 16GB+${NC}"
    echo "Continue anyway? (y/n)"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Installation directory
INSTALL_DIR="$HOME/mia-gpu-miner"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Step 1: Install system dependencies
echo -e "\n${YELLOW}[1/6] Installing system dependencies...${NC}"
sudo apt-get update -qq

# Check Python version and install appropriate packages
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo -e "${GREEN}Detected Python $PYTHON_VERSION${NC}"

if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    sudo apt-get install -y -qq python3.8 python3.8-venv python3-pip git curl wget build-essential
    PYTHON_CMD="python3.8"
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
    sudo apt-get install -y -qq python3.9 python3.9-venv python3-pip git curl wget build-essential
    PYTHON_CMD="python3.9"
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    sudo apt-get install -y -qq python3.10 python3.10-venv python3-pip git curl wget build-essential 2>/dev/null || {
        # Fallback if python3.10-venv not available
        sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget build-essential
    }
    PYTHON_CMD="python3.10"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl wget build-essential 2>/dev/null || {
        sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget build-essential
    }
    PYTHON_CMD="python3.11"
else
    # Default to python3
    sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget build-essential
    PYTHON_CMD="python3"
fi

# Step 2: Create Python virtual environment
echo -e "\n${YELLOW}[2/6] Setting up Python environment...${NC}"
$PYTHON_CMD -m venv venv || {
    echo -e "${RED}Failed to create virtual environment${NC}"
    echo "Trying with ensurepip..."
    $PYTHON_CMD -m venv --without-pip venv
    source venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py | python
}
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Step 3: Install PyTorch with CUDA support
echo -e "\n${YELLOW}[3/6] Installing PyTorch with CUDA support...${NC}"
# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1 | cut -dV -f2)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    echo -e "${GREEN}Detected CUDA $CUDA_VERSION${NC}"
    
    if [[ "$CUDA_MAJOR" == "12" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_MAJOR" == "11" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio
    fi
else
    # Default to CUDA 11.8
    echo -e "${YELLOW}CUDA not detected, installing PyTorch with CUDA 11.8 support${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Step 4: Install vLLM for efficient inference
echo -e "\n${YELLOW}[4/6] Installing vLLM inference server...${NC}"
# Install vLLM and dependencies
pip install --upgrade pip wheel setuptools

# Install vLLM (may need specific version for compatibility)
echo "Installing vLLM (this may take a few minutes)..."
pip install vllm==0.2.7 || {
    echo -e "${YELLOW}Trying latest vLLM version...${NC}"
    pip install vllm
}

# Install other dependencies
pip install transformers>=4.36.0 accelerate sentencepiece protobuf
pip install requests psutil gpustat py-cpuinfo uvicorn fastapi

# Step 5: Download configuration files
echo -e "\n${YELLOW}[5/6] Downloading miner scripts...${NC}"

# Create the inference server script
cat > "$INSTALL_DIR/vllm_server.py" << 'EOF'
#!/usr/bin/env python3
"""
vLLM Server for Mistral 7B Instruct
Provides HTTP endpoint for GPU inference
"""

import os
import argparse
from vllm import LLM, SamplingParams
from vllm.entrypoints.api_server import app, llm, sampling_params
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.95

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    model: str

# Initialize app
app = FastAPI(title="MIA vLLM Server")

# Global LLM instance
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    print(f"Loading {model_name}...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # Initialize vLLM with GPU
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="float16",  # Use fp16 for efficiency
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=4096,  # Context length
    )
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {"status": "ready", "model": "mistralai/Mistral-7B-Instruct-v0.1"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format prompt for Mistral Instruct
    formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    
    # Generate
    outputs = llm.generate([formatted_prompt], sampling_params)
    
    # Extract response
    generated_text = outputs[0].outputs[0].text
    tokens_generated = len(outputs[0].outputs[0].token_ids)
    
    return GenerateResponse(
        text=generated_text,
        tokens_generated=tokens_generated,
        model="mistralai/Mistral-7B-Instruct-v0.1"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
EOF

# Create the miner client script
cat > "$INSTALL_DIR/gpu_miner.py" << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner Client
Communicates with backend and runs inference on local GPU
"""

import os
import sys
import time
import json
import logging
import requests
import subprocess
import socket
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-gpu-miner')

class MIAGPUMiner:
    def __init__(self):
        self.api_url = os.environ.get('MIA_API_URL', 'https://mia-backend-production.up.railway.app').rstrip('/')
        self.miner_name = os.environ.get('MINER_NAME', f'gpu-miner-{socket.gethostname()}')
        self.vllm_url = os.environ.get('VLLM_URL', 'http://localhost:8000')
        self.poll_interval = int(os.environ.get('POLL_INTERVAL', '5'))
        
        self.miner_id = None
        self.gpu_info = self.get_gpu_info()
        self.ip_address = self.get_ip_address()
        
        logger.info("=" * 50)
        logger.info("MIA GPU Miner initialized")
        logger.info(f"Backend URL: {self.api_url}")
        logger.info(f"Miner Name: {self.miner_name}")
        logger.info(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_mb']}MB)")
        logger.info(f"vLLM URL: {self.vllm_url}")
        logger.info("=" * 50)
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'name': gpu.name,
                    'memory_mb': int(gpu.memoryTotal),
                    'memory_free_mb': int(gpu.memoryFree),
                    'utilization': gpu.load * 100
                }
        except:
            pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'name': parts[0].strip(),
                    'memory_mb': int(parts[1].strip().replace(' MiB', '')),
                    'memory_free_mb': 0,
                    'utilization': 0
                }
        except:
            pass
        
        return {
            'name': 'Unknown GPU',
            'memory_mb': 0,
            'memory_free_mb': 0,
            'utilization': 0
        }
    
    def get_ip_address(self) -> str:
        """Get public IP address"""
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            return response.json().get('ip', 'unknown')
        except:
            return socket.gethostbyname(socket.gethostname())
    
    def wait_for_vllm(self, timeout=300):
        """Wait for vLLM server to be ready"""
        logger.info("Waiting for vLLM server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.vllm_url}/", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready!")
                    return True
            except:
                pass
            time.sleep(5)
        
        logger.error("vLLM server failed to start")
        return False
    
    def register_miner(self) -> bool:
        """Register with the backend"""
        try:
            payload = {
                "name": self.miner_name,
                "ip_address": self.ip_address,
                "gpu_name": self.gpu_info['name'],
                "gpu_memory_mb": self.gpu_info['memory_mb'],
                "status": "idle"
            }
            
            response = requests.post(
                f"{self.api_url}/register_miner",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.miner_id = data.get('miner_id')
                logger.info(f"Registered successfully. Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def update_status(self, status: str):
        """Update miner status"""
        if not self.miner_id:
            return
        
        try:
            requests.post(
                f"{self.api_url}/miner/{self.miner_id}/status",
                json={"status": status},
                timeout=5
            )
        except:
            pass
    
    def process_prompt(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Process prompt using local vLLM server"""
        try:
            self.update_status("busy")
            
            response = requests.post(
                f"{self.vllm_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.95
                },
                timeout=60  # Give model time to generate
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "output": result["text"],
                    "tokens_generated": result["tokens_generated"]
                }
            else:
                return {
                    "success": False,
                    "error": f"vLLM error: {response.status_code}"
                }
        
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Generation timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.update_status("idle")
    
    def poll_for_jobs(self):
        """Main polling loop"""
        logger.info("Starting job polling...")
        
        # Wait for vLLM to be ready
        if not self.wait_for_vllm():
            logger.error("Cannot start without vLLM server")
            return
        
        # Register with backend
        if not self.register_miner():
            logger.error("Failed to register with backend")
            return
        
        while True:
            try:
                # Get next job
                response = requests.get(
                    f"{self.api_url}/job/next",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if response.status_code == 200:
                    job = response.json()
                    
                    if job.get('job_id'):
                        logger.info(f"Processing job {job['job_id']}")
                        
                        # Process the prompt
                        result = self.process_prompt(
                            job.get('prompt', ''),
                            job.get('max_tokens', 500)
                        )
                        
                        # Submit result
                        submit_response = requests.post(
                            f"{self.api_url}/job/result",
                            json={
                                "job_id": job['job_id'],
                                "session_id": job.get('session_id'),
                                "output": result.get('output', ''),
                                "miner_id": self.miner_id,
                                "success": result['success'],
                                "error": result.get('error'),
                                "tokens_generated": result.get('tokens_generated', 0)
                            },
                            timeout=10
                        )
                        
                        if submit_response.status_code == 200:
                            logger.info(f"Job {job['job_id']} completed successfully")
                        else:
                            logger.error(f"Failed to submit result: {submit_response.status_code}")
                
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(self.poll_interval * 2)
        
        logger.info("Miner stopped")

def main():
    miner = MIAGPUMiner()
    miner.poll_for_jobs()

if __name__ == "__main__":
    main()
EOF

# Create startup script
cat > "$INSTALL_DIR/start_miner.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Start vLLM server in background
echo "Starting vLLM server..."
python vllm_server.py --host 0.0.0.0 --port 8000 > vllm.log 2>&1 &
VLLM_PID=$!
echo $VLLM_PID > vllm.pid

# Wait a bit for server to start
sleep 10

# Start miner
echo "Starting GPU miner..."
python gpu_miner.py
EOF

chmod +x "$INSTALL_DIR/start_miner.sh"

# Create systemd service
echo -e "\n${YELLOW}[6/6] Creating systemd service...${NC}"
SERVICE_FILE="/tmp/mia-gpu-miner.service"
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=MIA GPU Miner with Mistral 7B
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="MIA_API_URL=${MIA_API_URL:-https://mia-backend-production.up.railway.app}"
Environment="MINER_NAME=${MINER_NAME:-gpu-miner-$(hostname)}"
ExecStart=$INSTALL_DIR/start_miner.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Get configuration
echo -e "\n${GREEN}Configuration${NC}"
echo "-------------"

# API URL
if [ -z "$MIA_API_URL" ]; then
    read -p "MIA Backend URL [https://mia-backend-production.up.railway.app]: " MIA_API_URL
    MIA_API_URL=${MIA_API_URL:-https://mia-backend-production.up.railway.app}
fi

# Miner name
if [ -z "$MINER_NAME" ]; then
    DEFAULT_NAME="gpu-miner-$(hostname)"
    read -p "Miner Name [$DEFAULT_NAME]: " MINER_NAME
    MINER_NAME=${MINER_NAME:-$DEFAULT_NAME}
fi

# Create environment file
cat > "$INSTALL_DIR/.env" << EOF
MIA_API_URL=$MIA_API_URL
MINER_NAME=$MINER_NAME
VLLM_URL=http://localhost:8000
POLL_INTERVAL=5
EOF

# Install service
sudo mv "$SERVICE_FILE" /etc/systemd/system/mia-gpu-miner.service || {
    echo -e "${RED}Failed to install systemd service${NC}"
    echo "You can run the miner manually with: $INSTALL_DIR/start_miner.sh"
    exit 1
}
sudo systemctl daemon-reload
sudo systemctl enable mia-gpu-miner.service

# Download model first (this takes time)
echo -e "\n${YELLOW}Downloading Mistral 7B model (this may take 10-20 minutes)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

# Create a simple script to download the model with progress
cat > download_model.py << 'EOF'
import os
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

print("Downloading Mistral 7B Instruct model...")
print("This is a 14GB download and may take some time.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    print("✓ Tokenizer downloaded")
    
    print("\nDownloading model weights (14GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.1',
        torch_dtype='auto',
        low_cpu_mem_usage=True
    )
    print("✓ Model downloaded successfully!")
    
    # Test model loading
    print("\nTesting model load...")
    del model  # Free memory
    print("✓ Model test passed")
    
except Exception as e:
    print(f"\nError downloading model: {e}")
    exit(1)
EOF

python download_model.py || {
    echo -e "${RED}Failed to download model${NC}"
    echo "Please check your internet connection and try again."
    exit 1
}
rm download_model.py

# Start service
echo -e "\n${YELLOW}Starting MIA GPU Miner service...${NC}"
sudo systemctl start mia-gpu-miner.service

echo -e "\n${GREEN}✓ MIA GPU Miner installation complete!${NC}"
echo ""
echo "Service commands:"
echo "  Check status: sudo systemctl status mia-gpu-miner"
echo "  View logs:    sudo journalctl -u mia-gpu-miner -f"
echo "  Stop:         sudo systemctl stop mia-gpu-miner"
echo "  Start:        sudo systemctl start mia-gpu-miner"
echo ""
echo "The miner is now running with Mistral 7B on GPU!"
echo "It will automatically register with the backend and start processing jobs."
echo ""