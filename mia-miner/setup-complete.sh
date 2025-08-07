#!/bin/bash

# MIA GPU Miner Complete Setup - Single Line Installer
# Handles everything: NVIDIA drivers, CUDA, PyTorch, Models, and Miner
# Works on fresh Ubuntu 20.04/22.04 VPS with GPU

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
echo "Complete GPU Miner Setup - One Line Installer"
echo "=============================================="

# Detect if we need sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
else
    SUDO_CMD="sudo"
fi

# Update system
echo -e "\n${YELLOW}[1/10] Updating system packages...${NC}"
$SUDO_CMD apt-get update -qq
$SUDO_CMD apt-get upgrade -y -qq

# Install NVIDIA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "\n${YELLOW}[2/10] Installing NVIDIA drivers...${NC}"
    $SUDO_CMD apt-get install -y -qq software-properties-common
    $SUDO_CMD add-apt-repository -y ppa:graphics-drivers/ppa
    $SUDO_CMD apt-get update -qq
    $SUDO_CMD apt-get install -y -qq nvidia-driver-535
    echo -e "${YELLOW}NVIDIA drivers installed. System will need reboot after setup.${NC}"
    NEEDS_REBOOT=1
else
    echo -e "\n${GREEN}[2/10] NVIDIA drivers already installed${NC}"
    nvidia-smi
fi

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "\n${GREEN}Detected Python $PYTHON_VERSION${NC}"

# Install system dependencies based on Python version
echo -e "\n${YELLOW}[3/10] Installing system dependencies...${NC}"

# For Ubuntu 20.04 (Python 3.8) or 22.04 (Python 3.10)
if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    $SUDO_CMD apt-get install -y -qq \
        python3.8 python3.8-venv python3.8-dev python3-pip \
        git curl wget build-essential \
        libssl-dev libffi-dev python3-setuptools
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    $SUDO_CMD apt-get install -y -qq \
        python3 python3-venv python3-dev python3-pip \
        git curl wget build-essential \
        libssl-dev libffi-dev python3-setuptools
else
    # Generic Python 3 installation
    $SUDO_CMD apt-get install -y -qq \
        python3 python3-venv python3-dev python3-pip \
        git curl wget build-essential \
        libssl-dev libffi-dev python3-setuptools
fi

# Install CUDA dependencies if available
$SUDO_CMD apt-get install -y -qq libnccl2 libnccl-dev 2>/dev/null || true

# Create installation directory
INSTALL_DIR="/opt/mia-gpu-miner"
echo -e "\n${YELLOW}[4/10] Creating installation directory...${NC}"
$SUDO_CMD mkdir -p "$INSTALL_DIR"
$SUDO_CMD chown -R $USER:$USER "$INSTALL_DIR" 2>/dev/null || true
cd "$INSTALL_DIR"

# Create Python virtual environment
echo -e "\n${YELLOW}[5/10] Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "\n${YELLOW}[6/10] Installing PyTorch with CUDA support...${NC}"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo -e "\n${YELLOW}[7/10] Installing AI dependencies...${NC}"
pip install \
    transformers==4.36.2 \
    vllm==0.2.7 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    requests==2.31.0 \
    pydantic==2.5.3 \
    huggingface-hub==0.20.3 \
    accelerate==0.26.1 \
    sentencepiece==0.1.99 \
    protobuf==4.25.2 \
    bitsandbytes==0.42.0

# Create vLLM server with Mistral-OpenOrca model (best open alternative)
echo -e "\n${YELLOW}[8/10] Creating vLLM server...${NC}"
cat > "$INSTALL_DIR/vllm_server.py" << 'EOF'
#!/usr/bin/env python3
"""
vLLM Server for Mistral-7B-OpenOrca
High-performance multilingual model based on Mistral architecture
"""

import os
import sys
import argparse
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging

# Set up logging
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

# Initialize app
app = FastAPI(title="MIA vLLM Server")

# Global LLM instance
llm = None
# Using OpenOrca Mistral - best performance + multilingual + no restrictions
MODEL_NAME = "Open-Orca/Mistral-7B-OpenOrca"

@app.on_event("startup")
async def startup_event():
    global llm
    
    logger.info(f"Loading {MODEL_NAME}...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        # Initialize vLLM with GPU
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            download_dir="/opt/mia-gpu-miner/models",
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

@app.get("/")
async def root():
    return {"status": "ready", "model": MODEL_NAME}

@app.get("/health")
async def health():
    return {"status": "healthy", "cuda_available": torch.cuda.is_available()}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt for Mistral/OpenOrca
        formatted_prompt = f"<|im_start|>user\n{request.prompt}<|im_end|>\n<|im_start|>assistant\n"
        
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
            model=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
EOF

# Create GPU miner client
echo -e "\n${YELLOW}[9/10] Creating miner client...${NC}"
cat > "$INSTALL_DIR/gpu_miner.py" << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner Client
Connects to MIA network and processes inference requests
"""

import os
import sys
import time
import json
import requests
import logging
from datetime import datetime
import traceback
import socket
import psutil

# Configuration
MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend.up.railway.app")
VLLM_SERVER_URL = "http://localhost:8000"
POLL_INTERVAL = 5  # seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MIAMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        self.hostname = socket.gethostname()
        self.start_time = datetime.utcnow()
        
    def get_system_info(self):
        """Get system information"""
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A"
        except:
            gpu_name = "Unknown"
            gpu_memory = "Unknown"
            
        return {
            "hostname": self.hostname,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_name": gpu_name,
            "gpu_memory": gpu_memory,
            "platform": sys.platform,
            "python_version": sys.version.split()[0]
        }
    
    def wait_for_vllm(self, max_retries=30):
        """Wait for vLLM server to be ready"""
        logger.info("Waiting for vLLM server to start...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{VLLM_SERVER_URL}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("vLLM server is ready!")
                    return True
            except:
                pass
            time.sleep(10)
        logger.error("vLLM server failed to start!")
        return False
    
    def register_miner(self):
        """Register with MIA backend"""
        logger.info(f"Registering with MIA backend at {MIA_BACKEND_URL}")
        
        system_info = self.get_system_info()
        
        data = {
            "wallet_address": os.getenv("WALLET_ADDRESS", "0x" + "0" * 40),
            "endpoint_url": f"http://{requests.get('https://api.ipify.org').text}:8000",
            "model": "Open-Orca/Mistral-7B-OpenOrca",
            "max_tokens": 4096,
            "system_info": system_info
        }
        
        try:
            response = self.session.post(
                f"{MIA_BACKEND_URL}/miner/register",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            self.miner_id = result["miner_id"]
            logger.info(f"Registered successfully! Miner ID: {self.miner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def process_request(self, request_data):
        """Process an inference request"""
        try:
            # Forward to vLLM server
            response = self.session.post(
                f"{VLLM_SERVER_URL}/generate",
                json={
                    "prompt": request_data["prompt"],
                    "max_tokens": request_data.get("max_tokens", 500),
                    "temperature": request_data.get("temperature", 0.7),
                    "top_p": request_data.get("top_p", 0.95)
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "response": result["text"],
                "tokens_generated": result["tokens_generated"],
                "model": result["model"]
            }
            
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run(self):
        """Main miner loop"""
        # Wait for vLLM server
        if not self.wait_for_vllm():
            logger.error("Cannot start without vLLM server")
            return
        
        # Register miner
        while not self.register_miner():
            logger.info("Retrying registration in 30 seconds...")
            time.sleep(30)
        
        logger.info("Starting main mining loop...")
        consecutive_errors = 0
        
        while True:
            try:
                # Poll for work
                response = self.session.get(
                    f"{MIA_BACKEND_URL}/miner/{self.miner_id}/work",
                    timeout=10
                )
                
                if response.status_code == 200:
                    work = response.json()
                    if work.get("request_id"):
                        logger.info(f"Received work: {work['request_id']}")
                        
                        # Process the request
                        result = self.process_request(work)
                        
                        # Submit result
                        submit_response = self.session.post(
                            f"{MIA_BACKEND_URL}/miner/{self.miner_id}/submit",
                            json={
                                "request_id": work["request_id"],
                                "result": result
                            },
                            timeout=30
                        )
                        submit_response.raise_for_status()
                        logger.info(f"Submitted result for {work['request_id']}")
                        
                consecutive_errors = 0
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in main loop: {e}")
                
                if consecutive_errors > 10:
                    logger.error("Too many consecutive errors, exiting...")
                    break
                
                time.sleep(min(consecutive_errors * 5, 60))
                continue
            
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    logger.info("Starting MIA GPU Miner...")
    miner = MIAMiner()
    miner.run()
EOF

# Create startup script
echo -e "\n${YELLOW}[10/10] Creating startup script...${NC}"
cat > "$INSTALL_DIR/start_miner.sh" << 'EOF'
#!/bin/bash

cd /opt/mia-gpu-miner
source venv/bin/activate

# Kill any existing processes
pkill -f vllm_server.py || true
pkill -f gpu_miner.py || true

echo "Starting vLLM server..."
python vllm_server.py > vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to start
echo "Waiting for vLLM to load model (this may take 2-3 minutes)..."
sleep 30

# Check if vLLM is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM failed to start! Check vllm.log for errors"
    exit 1
fi

# Start miner
echo "Starting GPU miner..."
python gpu_miner.py 2>&1 | tee miner.log
EOF

chmod +x "$INSTALL_DIR/start_miner.sh"

# Create systemd service
echo -e "\n${YELLOW}Creating systemd service...${NC}"
$SUDO_CMD cat > /etc/systemd/system/mia-miner.service << EOF
[Unit]
Description=MIA GPU Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mia-gpu-miner
ExecStart=/opt/mia-gpu-miner/start_miner.sh
Restart=always
RestartSec=10
StandardOutput=append:/opt/mia-gpu-miner/miner.log
StandardError=append:/opt/mia-gpu-miner/miner.log

[Install]
WantedBy=multi-user.target
EOF

# Download the model
echo -e "\n${YELLOW}Downloading AI model (this will take 10-20 minutes)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate
python3 << 'PYEOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.makedirs("/opt/mia-gpu-miner/models", exist_ok=True)
print("Downloading Mistral-7B-OpenOrca model...")
print("This is a 14GB download with excellent multilingual support...")

try:
    tokenizer = AutoTokenizer.from_pretrained('Open-Orca/Mistral-7B-OpenOrca', cache_dir="/opt/mia-gpu-miner/models")
    model = AutoModelForCausalLM.from_pretrained(
        'Open-Orca/Mistral-7B-OpenOrca', 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        cache_dir="/opt/mia-gpu-miner/models"
    )
    print("✓ Model downloaded successfully!")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("The miner will download it on first run.")
PYEOF

# Set permissions
$SUDO_CMD chown -R $USER:$USER "$INSTALL_DIR"

# Enable and start service
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\nTo start the miner:"
echo -e "  ${YELLOW}sudo systemctl enable mia-miner${NC}"
echo -e "  ${YELLOW}sudo systemctl start mia-miner${NC}"
echo -e "\nTo check status:"
echo -e "  ${YELLOW}sudo systemctl status mia-miner${NC}"
echo -e "  ${YELLOW}sudo journalctl -u mia-miner -f${NC}"
echo -e "\nOr run directly:"
echo -e "  ${YELLOW}cd /opt/mia-gpu-miner && ./start_miner.sh${NC}"

if [ "${NEEDS_REBOOT}" == "1" ]; then
    echo -e "\n${RED}IMPORTANT: System reboot required for NVIDIA drivers!${NC}"
    echo -e "Run: ${YELLOW}sudo reboot${NC}"
fi