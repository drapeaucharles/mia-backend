#!/bin/bash

# MIA GPU Miner - Python 3.8 Compatible Version
# Works with older Python versions

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - Python 3.8 Compatible   ║${NC}"
echo -e "${GREEN}║         Optimized for Vast.ai             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${YELLOW}Python version: $PYTHON_VERSION${NC}"

# Create directory
INSTALL_DIR="$HOME/mia-gpu-miner"
echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Install compatible versions for Python 3.8
echo -e "${YELLOW}Installing Python 3.8 compatible packages...${NC}"

# Create libs directory
mkdir -p libs

# Install PyTorch 1.13.1 (last version supporting Python 3.8)
echo -e "${YELLOW}Installing PyTorch 1.13.1 (Python 3.8 compatible)...${NC}"
pip3 install --target=libs torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install compatible transformers
echo -e "${YELLOW}Installing compatible transformers...${NC}"
pip3 install --target=libs transformers==4.30.0 accelerate==0.20.0
pip3 install --target=libs flask waitress requests
pip3 install --target=libs sentencepiece protobuf huggingface-hub

# Download model
echo -e "${YELLOW}Downloading model...${NC}"
cat > download_model.py << 'EOF'
import sys
sys.path.insert(0, 'libs')

from huggingface_hub import snapshot_download

print("Downloading Mistral-7B-OpenOrca-GPTQ...")
model_path = snapshot_download(
    repo_id="TheBloke/Mistral-7B-OpenOrca-GPTQ",
    local_dir="./models/mistral-gptq",
    resume_download=True,
    ignore_patterns=["*.md", "*.txt", ".gitattributes"]
)
print(f"Model downloaded to: {model_path}")
EOF

python3 download_model.py

# Create Python 3.8 compatible miner
echo -e "${YELLOW}Creating miner...${NC}"
cat > mia_miner_unified.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Python 3.8 Compatible
"""
import sys
sys.path.insert(0, 'libs')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import socket
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia')

# Global model variables
model = None
tokenizer = None
app = Flask(__name__)

class ModelServer:
    def __init__(self):
        self.model_loaded = False
    
    def load_model(self):
        global model, tokenizer
        
        logger.info("Loading model (Python 3.8 compatible)...")
        
        try:
            # Check CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                device = "cuda:0"
            else:
                logger.warning("CUDA not detected, using CPU")
                device = "cpu"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            logger.info("Loading GPTQ model...")
            model = AutoModelForCausalLM.from_pretrained(
                "./models/mistral-gptq",
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            model.eval()
            self.model_loaded = True
            logger.info("✓ Model loaded!")
            
            # Test speed
            self._test_speed()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _test_speed(self):
        logger.info("Testing speed...")
        inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
        
        # Warmup
        with torch.no_grad():
            model.generate(inputs.input_ids, max_new_tokens=10)
        
        # Test
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=30)
        elapsed = time.time() - start
        
        logger.info(f"Speed: {30/elapsed:.1f} tokens/second")
    
    def start_server(self):
        def run_server():
            logger.info("Starting server on port 8000...")
            serve(app, host="0.0.0.0", port=8000)
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        time.sleep(3)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready" if model else "loading"})

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = min(data.get("max_tokens", 200), 500)
        
        # Format prompt
        formatted = f"""<|im_start|>system
You are MIA, a helpful multilingual AI assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        # Generate
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = response.replace("<|im_end|>", "").strip()
        
        gen_time = time.time() - start_time
        tokens_generated = len(generated_ids)
        
        return jsonify({
            "text": response,
            "tokens_generated": int(tokens_generated),
            "time": round(gen_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

class MinerClient:
    def __init__(self):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        self.miner_id = None
    
    def get_gpu_info(self):
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'memory_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024)
            }
        return {'name': 'CPU', 'memory_mb': 0}
    
    def wait_for_model(self):
        logger.info("Waiting for model...")
        for i in range(30):
            try:
                r = requests.get("http://localhost:8000/health", timeout=5)
                if r.status_code == 200 and r.json().get("status") == "ready":
                    logger.info("✓ Model ready")
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def register(self):
        try:
            gpu_info = self.get_gpu_info()
            data = {
                "name": self.miner_name,
                "ip_address": "vastai",
                "gpu_name": gpu_info['name'],
                "gpu_memory_mb": gpu_info['memory_mb'],
                "status": "idle"
            }
            
            logger.info(f"Registering {self.miner_name}...")
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                self.miner_id = r.json().get('miner_id')
                logger.info(f"✓ Registered! ID: {self.miner_id}")
                return True
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
        return False
    
    def process_job(self, job):
        try:
            logger.info(f"Processing job {job['request_id']}")
            
            r = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": job.get("prompt", ""),
                    "max_tokens": job.get("max_tokens", 200)
                },
                timeout=60
            )
            
            if r.status_code == 200:
                result = r.json()
                
                submit_data = {
                    "miner_id": self.miner_id,
                    "request_id": job["request_id"],
                    "result": {
                        "response": result.get("text", ""),
                        "tokens_generated": result.get("tokens_generated", 0),
                        "processing_time": result.get("time", 0)
                    }
                }
                
                requests.post(
                    f"{self.backend_url}/submit_result",
                    json=submit_data,
                    timeout=30
                )
                
                logger.info(f"✓ Job complete")
                return True
                
        except Exception as e:
            logger.error(f"Job error: {e}")
        return False
    
    def run_mining_loop(self):
        logger.info("Starting mining...")
        
        while True:
            try:
                r = requests.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    if work and work.get("request_id"):
                        self.process_job(work)
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except:
                time.sleep(10)

def main():
    logger.info("MIA GPU Miner - Python 3.8 Compatible")
    
    # Load model
    server = ModelServer()
    if not server.load_model():
        return
    
    server.start_server()
    
    # Initialize client
    client = MinerClient()
    
    if not client.wait_for_model():
        return
    
    # Test multilingual
    logger.info("Testing multilingual support...")
    for prompt in ["Hello", "Bonjour", "Hola", "你好"]:
        try:
            r = requests.post(
                "http://localhost:8000/generate",
                json={"prompt": prompt, "max_tokens": 20},
                timeout=10
            )
            if r.status_code == 200:
                logger.info(f"✓ {prompt}: OK")
        except:
            pass
    
    # Register and mine
    if client.register():
        client.run_mining_loop()

if __name__ == "__main__":
    main()
EOF

chmod +x mia_miner_unified.py

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/libs:$PYTHONPATH"
python3 mia_miner_unified.py
EOF
chmod +x run.sh

echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""
echo -e "${YELLOW}Using Python 3.8 compatible versions:${NC}"
echo "• PyTorch 1.13.1 (CUDA 11.7)"
echo "• Transformers 4.30.0"
echo "• Full multilingual support"
echo ""
echo -e "${YELLOW}To start mining:${NC}"
echo "  cd $INSTALL_DIR"
echo "  ./run.sh"
echo ""