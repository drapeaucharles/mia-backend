#!/bin/bash

# MIA GPU Miner Unified Installer - Fixed Version
# One-line installer: bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner-unified.sh)

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
echo "GPU Miner Setup - Fixed Version"
echo "==============================="

# Check Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: Linux required${NC}"
    exit 1
fi

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

# Check memory
if [ "$GPU_MEMORY_MB" -lt 8000 ]; then
    echo -e "${RED}Error: Minimum 8GB VRAM required${NC}"
    exit 1
fi

# Setup directories
INSTALL_DIR="/opt/mia-gpu-miner"
sudo mkdir -p "$INSTALL_DIR"
sudo chown $USER:$USER "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git curl wget build-essential

# Create venv
echo -e "\n${YELLOW}Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch FIRST (required for auto-gptq)
echo -e "\n${YELLOW}Installing PyTorch with CUDA 11.8...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA {torch.cuda.is_available()}')" || exit 1

# Install auto-gptq (requires PyTorch to be installed first)
echo -e "\n${YELLOW}Installing auto-gptq...${NC}"
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# Install other requirements
echo -e "\n${YELLOW}Installing inference packages...${NC}"
pip install transformers accelerate sentencepiece protobuf optimum
pip install requests psutil gpustat py-cpuinfo
pip install flask waitress

# Create simple inference server
cat > "$INSTALL_DIR/inference_server.py" << 'EOF'
#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    logger.info("Loading model...")
    
    # IMPORTANT: Use the original model's tokenizer, not the GPTQ one
    tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
    model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        revision="main"
    )
    
    logger.info("Model loaded successfully!")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready", "model": "Mistral-7B-OpenOrca-GPTQ"})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        
        # Use proper ChatML format
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize with proper settings
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the generated tokens only (not including the input)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any remaining special tokens that might have slipped through
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>", "").strip()
        
        return jsonify({
            "text": response,
            "tokens_generated": len(outputs[0]) - len(inputs.input_ids[0]),
            "model": "Mistral-7B-OpenOrca-GPTQ"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    load_model()
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
EOF

# Create miner client
cat > "$INSTALL_DIR/miner_client.py" << 'EOF'
#!/usr/bin/env python3
import os
import time
import requests
import logging
import socket
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIAMiner:
    def __init__(self):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        self.miner_id = None
        
    def get_gpu_info(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            parts = result.stdout.strip().split(',')
            return {
                'name': parts[0].strip(),
                'memory_mb': int(parts[1].strip().replace(' MiB', ''))
            }
        except:
            return {'name': 'Unknown GPU', 'memory_mb': 0}
    
    def wait_for_server(self):
        logger.info("Waiting for inference server...")
        for i in range(60):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200:
                    logger.info("✓ Inference server ready")
                    return True
            except:
                pass
            time.sleep(5)
        return False
    
    def register(self):
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
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                resp = r.json()
                self.miner_id = resp.get('miner_id')
                logger.info(f"✓ Registered! Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code} - {r.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def update_status(self, status):
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
    
    def run(self):
        logger.info("Starting MIA GPU Miner")
        
        if not self.wait_for_server():
            logger.error("Inference server not responding")
            return
        
        # Register
        if not self.register():
            logger.error("Failed to register")
            return
        
        logger.info("Starting work loop...")
        errors = 0
        
        while True:
            try:
                # Get work
                r = requests.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    
                    if work and work.get("request_id"):
                        logger.info(f"Processing job: {work['request_id']}")
                        self.update_status("busy")
                        
                        # Process
                        start_time = time.time()
                        gen_r = requests.post(
                            f"{self.local_url}/generate",
                            json={
                                "prompt": work.get("prompt", ""),
                                "max_tokens": work.get("max_tokens", 500)
                            },
                            timeout=120
                        )
                        
                        if gen_r.status_code == 200:
                            result = gen_r.json()
                            processing_time = time.time() - start_time
                            
                            # Submit result
                            submit_r = requests.post(
                                f"{self.backend_url}/submit_result",
                                json={
                                    "miner_id": self.miner_id,
                                    "request_id": work["request_id"],
                                    "result": {
                                        "response": result.get("text", ""),
                                        "tokens_generated": result.get("tokens_generated", 0),
                                        "processing_time": processing_time
                                    }
                                },
                                timeout=30
                            )
                            
                            if submit_r.status_code == 200:
                                logger.info(f"✓ Completed job {work['request_id']}")
                            else:
                                logger.error(f"Submit failed: {submit_r.status_code}")
                        else:
                            logger.error(f"Generation failed: {gen_r.status_code}")
                        
                        self.update_status("idle")
                
                errors = 0
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                errors += 1
                logger.error(f"Error: {e}")
                if errors > 10:
                    logger.error("Too many errors, exiting")
                    break
                time.sleep(min(errors * 5, 60))

if __name__ == "__main__":
    miner = MIAMiner()
    miner.run()
EOF

# Create start script
cat > "$INSTALL_DIR/start_miner.sh" << 'EOF'
#!/bin/bash
cd /opt/mia-gpu-miner
source venv/bin/activate

echo "Starting inference server..."
python inference_server.py > server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > server.pid

sleep 30

echo "Starting miner client..."
python miner_client.py
EOF

chmod +x "$INSTALL_DIR/start_miner.sh"
chmod +x "$INSTALL_DIR/inference_server.py"
chmod +x "$INSTALL_DIR/miner_client.py"

# Create systemd service
sudo tee /etc/systemd/system/mia-gpu-miner.service > /dev/null << EOF
[Unit]
Description=MIA GPU Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/mia-gpu-miner
ExecStart=/opt/mia-gpu-miner/start_miner.sh
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Download model
echo -e "\n${YELLOW}Downloading GPTQ model (this may take 5-10 minutes)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

python3 -c "
print('Downloading Mistral-7B-OpenOrca models...')
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Download proper tokenizer
    print('Downloading tokenizer from Open-Orca/Mistral-7B-OpenOrca...')
    tokenizer = AutoTokenizer.from_pretrained('Open-Orca/Mistral-7B-OpenOrca')
    print('✓ Tokenizer downloaded')
    # Download GPTQ model
    print('Downloading GPTQ model from TheBloke/Mistral-7B-OpenOrca-GPTQ...')
    model = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Mistral-7B-OpenOrca-GPTQ',
        device_map='auto',
        trust_remote_code=True,
        revision='main'
    )
    print('✓ GPTQ Model downloaded successfully!')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable mia-gpu-miner.service
sudo systemctl start mia-gpu-miner.service

echo -e "\n${GREEN}✓ Installation complete!${NC}"
echo ""
echo "Commands:"
echo "  Status: sudo systemctl status mia-gpu-miner"
echo "  Logs:   sudo journalctl -u mia-gpu-miner -f"
echo "  Stop:   sudo systemctl stop mia-gpu-miner"
echo "  Start:  sudo systemctl start mia-gpu-miner"
echo ""
echo "Your miner is now running!"