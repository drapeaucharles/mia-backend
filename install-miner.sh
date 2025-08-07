#!/bin/bash

# MIA GPU Miner - Unified Installer
# One-line: bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner.sh)

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
echo "Unified GPU Miner Installer"
echo "==========================="

# Check Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: Linux required${NC}"
    exit 1
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: NVIDIA GPU drivers not found${NC}"
    echo "Install with: sudo apt install nvidia-driver-525"
    exit 1
fi

# Get GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
GPU_MEMORY_MB=$(echo $GPU_INFO | cut -d',' -f2 | grep -o '[0-9]*')

echo -e "${GREEN}GPU: $GPU_NAME (${GPU_MEMORY_MB}MB)${NC}"

if [ "$GPU_MEMORY_MB" -lt 8000 ]; then
    echo -e "${RED}Error: Minimum 8GB VRAM required${NC}"
    exit 1
fi

# Setup directory
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

# Install packages
pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm transformers accelerate sentencepiece protobuf
pip install flask waitress requests psutil gpustat
pip install auto-gptq optimum

# Create unified miner
cat > "$INSTALL_DIR/mia_miner.py" << 'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import torch
import socket
import logging
import requests
import threading
import subprocess
from flask import Flask, request, jsonify
from waitress import serve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mia-miner')

app = Flask(__name__)
model = None
model_ready = False

def load_model():
    global model, model_ready
    logger.info("Loading model...")
    
    try:
        # Try vLLM first with GPTQ model
        from vllm import LLM, SamplingParams
        model = LLM(
            model="TheBloke/Mistral-7B-OpenOrca-GPTQ",
            trust_remote_code=True,
            quantization="gptq",
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )
        model_ready = True
        logger.info("✓ Model loaded with vLLM (GPTQ)")
        return True
    except Exception as e:
        logger.error(f"vLLM failed: {e}, trying transformers...")
        
        # Fallback to transformers with GPTQ
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            global tokenizer
            model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                revision="gptq-4bit-32g-actorder_True"
            )
            model_ready = True
            logger.info("✓ Model loaded with transformers")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

@app.route("/health")
def health():
    return jsonify({"status": "ready" if model_ready else "loading"})

@app.route("/generate", methods=["POST"])
def generate():
    if not model_ready:
        return jsonify({"error": "Model not ready"}), 503
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 500)
    
    try:
        if hasattr(model, 'generate') and hasattr(model, 'get_tokenizer'):
            # vLLM
            from vllm import SamplingParams
            outputs = model.generate([prompt], SamplingParams(
                temperature=0.7,
                max_tokens=max_tokens
            ))
            text = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)
        else:
            # Transformers
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            tokens = len(outputs[0]) - len(inputs.input_ids[0])
        
        return jsonify({"text": text, "tokens_generated": tokens})
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

def run_server():
    logger.info("Starting model server on port 8000...")
    serve(app, host="0.0.0.0", port=8000, threads=4)

def run_miner():
    backend_url = "https://mia-backend-production.up.railway.app"
    miner_name = f"gpu-miner-{socket.gethostname()}"
    miner_id = None
    
    # Wait for model
    logger.info("Waiting for model server...")
    for i in range(60):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            if r.status_code == 200 and r.json()["status"] == "ready":
                logger.info("✓ Model server ready")
                break
        except:
            pass
        time.sleep(5)
    else:
        logger.error("Model server timeout")
        return
    
    # Get GPU info
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        parts = result.stdout.strip().split(',')
        gpu_name = parts[0].strip()
        gpu_memory = int(parts[1].strip().replace(' MiB', ''))
    except:
        gpu_name = "Unknown GPU"
        gpu_memory = 0
    
    # Register
    logger.info(f"Registering {miner_name}...")
    try:
        r = requests.post(f"{backend_url}/register_miner", json={
            "name": miner_name,
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory
        }, timeout=30)
        
        if r.status_code == 200:
            miner_id = r.json()['miner_id']
            logger.info(f"✓ Registered! ID: {miner_id}")
        else:
            logger.error(f"Registration failed: {r.status_code}")
            return
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return
    
    # Work loop
    logger.info("Starting work loop...")
    errors = 0
    
    while True:
        try:
            # Get work
            r = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=10)
            
            if r.status_code == 200:
                work = r.json()
                
                if work.get("request_id"):
                    logger.info(f"Processing job {work['request_id']}")
                    
                    # Generate
                    start = time.time()
                    gen_r = requests.post("http://localhost:8000/generate", json={
                        "prompt": work.get("prompt", ""),
                        "max_tokens": work.get("max_tokens", 500)
                    }, timeout=120)
                    
                    if gen_r.status_code == 200:
                        result = gen_r.json()
                        elapsed = time.time() - start
                        
                        # Submit
                        submit_r = requests.post(f"{backend_url}/submit_result", json={
                            "miner_id": miner_id,
                            "request_id": work["request_id"],
                            "result": {
                                "response": result["text"],
                                "tokens_generated": result["tokens_generated"],
                                "processing_time": elapsed
                            }
                        }, timeout=30)
                        
                        if submit_r.status_code == 200:
                            logger.info(f"✓ Completed in {elapsed:.2f}s")
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
    if not load_model():
        sys.exit(1)
    
    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(5)
    
    # Run miner
    run_miner()
EOF

chmod +x "$INSTALL_DIR/mia_miner.py"

# Create start script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd /opt/mia-gpu-miner
source venv/bin/activate
exec python mia_miner.py
EOF

chmod +x "$INSTALL_DIR/start.sh"

# Create systemd service
sudo tee /etc/systemd/system/mia-miner.service > /dev/null << EOF
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

[Install]
WantedBy=multi-user.target
EOF

# Download model
echo -e "\n${YELLOW}Downloading GPTQ model (this may take 10-15 minutes)...${NC}"
source venv/bin/activate

# Install GPTQ support
pip install auto-gptq optimum

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Mistral-7B-OpenOrca-GPTQ...')
try:
    # Download GPTQ model
    tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-OpenOrca-GPTQ')
    model = AutoModelForCausalLM.from_pretrained(
        'TheBloke/Mistral-7B-OpenOrca-GPTQ',
        device_map='auto',
        trust_remote_code=True,
        revision='gptq-4bit-32g-actorder_True'
    )
    print('✓ GPTQ model downloaded successfully')
except Exception as e:
    print(f'Error downloading model: {e}')
    exit(1)
"

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mia-miner.service
sudo systemctl start mia-miner.service

echo -e "\n${GREEN}✓ Installation complete!${NC}"
echo ""
echo "Commands:"
echo "  Status: sudo systemctl status mia-miner"
echo "  Logs:   sudo journalctl -u mia-miner -f"
echo ""
echo "Your miner is now running!"