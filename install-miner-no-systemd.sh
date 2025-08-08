#!/bin/bash

# MIA GPU Miner Installer - No SystemD Version (for Docker/WSL/Containers)
# One-line: bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner-no-systemd.sh)

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
echo "GPU Miner Setup - No SystemD Version"
echo "===================================="

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA GPU required${NC}"
    exit 1
fi

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
GPU_MEMORY_MB=$(echo $GPU_MEMORY | grep -o '[0-9]*' | head -n1)

echo -e "${GREEN}GPU: $GPU_NAME ($GPU_MEMORY)${NC}"

if [ "$GPU_MEMORY_MB" -lt 8000 ]; then
    echo -e "${RED}Error: Minimum 8GB VRAM required${NC}"
    exit 1
fi

# Setup directories
INSTALL_DIR="$HOME/mia-gpu-miner"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Install system dependencies if we have sudo
if command -v sudo &> /dev/null; then
    echo -e "\n${YELLOW}Installing system dependencies...${NC}"
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget build-essential 2>/dev/null || true
else
    echo -e "\n${YELLOW}No sudo access, assuming dependencies are installed${NC}"
fi

# Create venv
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch
echo -e "\n${YELLOW}Installing PyTorch...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo -e "\n${YELLOW}Installing inference packages...${NC}"
pip install transformers accelerate sentencepiece protobuf
pip install requests psutil gpustat py-cpuinfo
pip install flask waitress
pip install auto-gptq optimum

# Create unified miner script (same as before)
cat > "$INSTALL_DIR/mia_miner_unified.py" << 'EOF'
#!/usr/bin/env python3
"""
MIA Unified GPU Miner - Model Server and Client in One
"""
import os
import sys
import time
import torch
import socket
import logging
import requests
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-unified')

# Global model variables
model = None
tokenizer = None
app = Flask(__name__)

class ModelServer:
    """Embedded model server"""
    
    def __init__(self):
        self.server_thread = None
        self.model_loaded = False
    
    def load_model(self):
        """Load the AI model"""
        global model, tokenizer
        
        logger.info("Loading AI model...")
        model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load GPTQ model for lower VRAM usage
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=False,
                revision="gptq-4bit-32g-actorder_True"
            )
            
            self.model_loaded = True
            logger.info("✓ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        def run_server():
            logger.info("Starting inference server on port 8000...")
            serve(app, host="0.0.0.0", port=8000, threads=4)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(5)  # Give server time to start

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "model": "Mistral-7B-OpenOrca-GPTQ"
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        
        # Format prompt for model
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        
        return jsonify({
            "text": response,
            "tokens_generated": tokens_generated,
            "model": "Mistral-7B-OpenOrca-GPTQ"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

class MinerClient:
    """Miner client that connects to backend"""
    
    def __init__(self, model_server):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        self.miner_id = None
        self.model_server = model_server
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'name': parts[0].strip(),
                    'memory_mb': int(parts[1].strip().replace(' MiB', ''))
                }
        except:
            pass
        return {'name': 'Unknown GPU', 'memory_mb': 0}
    
    def wait_for_model(self):
        """Wait for model server to be ready"""
        logger.info("Waiting for model server...")
        
        for i in range(60):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "ready":
                        logger.info("✓ Model server ready")
                        return True
            except:
                pass
            time.sleep(5)
        
        return False
    
    def register(self):
        """Register with MIA backend"""
        try:
            gpu_info = self.get_gpu_info()
            
            # Get public IP
            try:
                ip = requests.get('https://api.ipify.org', timeout=10).text
            except:
                ip = "unknown"
            
            data = {
                "name": self.miner_name,
                "ip_address": ip,
                "gpu_name": gpu_info['name'],
                "gpu_memory_mb": gpu_info['memory_mb'],
                "status": "idle"
            }
            
            logger.info(f"Registering miner: {self.miner_name}")
            logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_mb']}MB)")
            
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                resp = r.json()
                self.miner_id = resp.get('miner_id')
                logger.info(f"✓ Registered successfully! Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code} - {r.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def update_status(self, status):
        """Update miner status"""
        if not self.miner_id:
            return
        
        try:
            requests.post(
                f"{self.backend_url}/miner/{self.miner_id}/status",
                json={"status": status},
                timeout=5
            )
        except:
            pass
    
    def process_job(self, job):
        """Process a single job"""
        try:
            self.update_status("busy")
            
            logger.info(f"Processing job: {job['request_id']}")
            start_time = time.time()
            
            # Call local model server
            r = requests.post(
                f"{self.local_url}/generate",
                json={
                    "prompt": job.get("prompt", ""),
                    "max_tokens": job.get("max_tokens", 500)
                },
                timeout=120
            )
            
            if r.status_code == 200:
                result = r.json()
                processing_time = time.time() - start_time
                
                # Submit result
                submit_data = {
                    "miner_id": self.miner_id,
                    "request_id": job["request_id"],
                    "result": {
                        "response": result.get("text", ""),
                        "tokens_generated": result.get("tokens_generated", 0),
                        "processing_time": processing_time
                    }
                }
                
                submit_r = requests.post(
                    f"{self.backend_url}/submit_result",
                    json=submit_data,
                    timeout=30
                )
                
                if submit_r.status_code == 200:
                    logger.info(f"✓ Completed job {job['request_id']} in {processing_time:.2f}s")
                    return True
                else:
                    logger.error(f"Failed to submit result: {submit_r.status_code}")
                    return False
            else:
                logger.error(f"Generation failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return False
        finally:
            self.update_status("idle")
    
    def run_mining_loop(self):
        """Main mining loop"""
        logger.info("Starting mining loop...")
        consecutive_errors = 0
        
        while True:
            try:
                # Get work from backend
                r = requests.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    
                    if work and work.get("request_id"):
                        # Process the job
                        if self.process_job(work):
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                    else:
                        # No work available
                        consecutive_errors = 0
                else:
                    logger.warning(f"Failed to get work: {r.status_code}")
                    consecutive_errors += 1
                
                # Check if too many errors
                if consecutive_errors > 10:
                    logger.error("Too many consecutive errors, restarting...")
                    time.sleep(60)
                    consecutive_errors = 0
                
                # Wait before next poll
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Mining loop error: {e}")
                consecutive_errors += 1
                time.sleep(min(consecutive_errors * 5, 60))

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("MIA Unified GPU Miner Starting")
    logger.info("=" * 60)
    
    # Initialize model server
    model_server = ModelServer()
    
    # Load model first
    if not model_server.load_model():
        logger.error("Failed to load model, exiting")
        sys.exit(1)
    
    # Start inference server
    model_server.start_server()
    
    # Initialize miner client
    miner = MinerClient(model_server)
    
    # Wait for model server to be ready
    if not miner.wait_for_model():
        logger.error("Model server failed to start")
        sys.exit(1)
    
    # Register with backend
    attempts = 0
    while not miner.register():
        attempts += 1
        if attempts > 5:
            logger.error("Failed to register after 5 attempts")
            sys.exit(1)
        logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
        time.sleep(30)
    
    # Start mining
    try:
        miner.run_mining_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x "$INSTALL_DIR/mia_miner_unified.py"

# Create run script
cat > "$INSTALL_DIR/run_miner.sh" << 'EOF'
#!/bin/bash
cd ~/mia-gpu-miner
source venv/bin/activate
exec python mia_miner_unified.py
EOF

chmod +x "$INSTALL_DIR/run_miner.sh"

# Download model
echo -e "\n${YELLOW}Downloading GPTQ model (this may take 5-10 minutes)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

python3 -c "
print('Downloading Mistral-7B-OpenOrca-GPTQ...')
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GPTQ')
    print('✓ Tokenizer downloaded')
    model = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Mistral-7B-OpenOrca-GPTQ',
        device_map='auto',
        trust_remote_code=False,
        revision='gptq-4bit-32g-actorder_True'
    )
    print('✓ GPTQ Model downloaded successfully!')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Model download failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Installation complete!${NC}"
echo ""
echo "To run the miner:"
echo ""
echo "  Option 1 - Direct run:"
echo "    cd ~/mia-gpu-miner && ./run_miner.sh"
echo ""
echo "  Option 2 - Background with screen:"
echo "    screen -dmS mia-miner ~/mia-gpu-miner/run_miner.sh"
echo "    screen -r mia-miner  # to view logs"
echo ""
echo "  Option 3 - Background with nohup:"
echo "    cd ~/mia-gpu-miner"
echo "    nohup ./run_miner.sh > miner.log 2>&1 &"
echo ""
echo "Your miner is ready to start!"