#!/bin/bash

# MIA GPU Miner - Simple Fast Version for Vast.ai
# Uses pre-built optimizations without complex dependencies

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - Simple Fast (Vast.ai)   ║${NC}"
echo -e "${GREEN}║    Optimized without complex builds        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Create directory
INSTALL_DIR="$HOME/mia-gpu-miner"
echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Simple Python setup - no venv issues
echo -e "${YELLOW}Setting up Python environment...${NC}"
export PATH="/usr/local/bin:$PATH"
export PYTHONPATH="$INSTALL_DIR/libs:$PYTHONPATH"
mkdir -p libs

# Install dependencies directly
echo -e "${YELLOW}Installing core dependencies...${NC}"
pip3 install --target=libs torch --index-url https://download.pytorch.org/whl/cu118
pip3 install --target=libs transformers accelerate flask waitress requests
pip3 install --target=libs sentencepiece protobuf huggingface-hub

# Download the optimal model
echo -e "${YELLOW}Downloading optimized model...${NC}"
cat > download_model.py << 'EOF'
import sys
sys.path.insert(0, 'libs')

from huggingface_hub import snapshot_download
import os

# Download the GPTQ model (proven to work)
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

# Create optimized miner
echo -e "${YELLOW}Creating optimized miner...${NC}"
cat > mia_miner_unified.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Optimized Simple Version
"""
import sys
sys.path.insert(0, 'libs')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import time
import torch
import socket
import logging
import requests
import threading
import subprocess
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optimize PyTorch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia')
logging.getLogger("transformers").setLevel(logging.WARNING)

# Global model variables
model = None
tokenizer = None
app = Flask(__name__)

class ModelServer:
    def __init__(self):
        self.server_thread = None
        self.model_loaded = False
    
    def load_model(self):
        global model, tokenizer
        
        logger.info("Loading optimized model...")
        
        try:
            # Check CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                device = "cuda:0"
            else:
                logger.warning("CUDA not detected, using CPU (will be slow)")
                device = "cpu"
            
            # Load tokenizer
            tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
            logger.info(f"Loading tokenizer from {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_path = "./models/mistral-gptq"
            logger.info(f"Loading model from {model_path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Optimize model
            model.eval()
            if hasattr(torch, 'compile') and device != "cpu":
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model, mode="reduce-overhead")
            
            self.model_loaded = True
            logger.info("✓ Model loaded and optimized!")
            
            # Speed test
            self._test_speed()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _test_speed(self):
        """Quick speed test"""
        logger.info("Testing inference speed...")
        prompt = "Hello"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=10)
        
        # Test
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        elapsed = time.time() - start
        
        tokens = 30
        speed = tokens / elapsed
        logger.info(f"Speed test: {elapsed:.2f}s for {tokens} tokens ({speed:.1f} tok/s)")
        
        if speed < 10:
            logger.warning("⚠️ Running slower than expected. Check GPU utilization.")
    
    def start_server(self):
        def run_server():
            logger.info("Starting server on port 8000...")
            serve(app, host="0.0.0.0", port=8000, threads=2)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(3)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "loading",
        "model": "Mistral-7B-OpenOrca-Optimized"
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = min(data.get("max_tokens", 200), 500)  # Cap for speed
        
        # ChatML format
        system_msg = "You are MIA, a helpful multilingual AI assistant."
        formatted = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        # Tokenize efficiently
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        # Generate with optimizations
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=model.device.type == "cuda"):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for speed
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = response.replace("<|im_end|>", "").strip()
        
        gen_time = time.time() - start_time
        tokens_generated = len(generated_ids)
        
        return jsonify({
            "text": response,
            "tokens_generated": int(tokens_generated),
            "generation_time": round(gen_time, 2),
            "tokens_per_second": round(tokens_generated / gen_time, 1)
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

class MinerClient:
    def __init__(self):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}-fast"
        self.miner_id = None
    
    def get_gpu_info(self):
        try:
            if torch.cuda.is_available():
                return {
                    'name': torch.cuda.get_device_name(0),
                    'memory_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024)
                }
        except:
            pass
        return {'name': 'CPU', 'memory_mb': 0}
    
    def wait_for_model(self):
        logger.info("Waiting for model server...")
        for i in range(30):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200 and r.json().get("status") == "ready":
                    logger.info("✓ Model server ready")
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def test_multilingual(self):
        """Test multilingual support"""
        logger.info("Testing multilingual support...")
        test_prompts = [
            ("English", "Hello, how are you?"),
            ("French", "Bonjour, comment allez-vous?"),
            ("Spanish", "Hola, ¿cómo estás?"),
            ("Chinese", "你好，你好吗？"),
            ("Arabic", "مرحبا، كيف حالك؟")
        ]
        
        for lang, prompt in test_prompts:
            try:
                start = time.time()
                r = requests.post(
                    f"{self.local_url}/generate",
                    json={"prompt": prompt, "max_tokens": 30},
                    timeout=30
                )
                elapsed = time.time() - start
                
                if r.status_code == 200:
                    result = r.json()
                    speed = result.get('tokens_per_second', 0)
                    logger.info(f"{lang}: {elapsed:.1f}s ({speed:.1f} tok/s)")
            except:
                pass
    
    def register(self):
        try:
            gpu_info = self.get_gpu_info()
            
            try:
                ip = requests.get('https://api.ipify.org', timeout=10).text
            except:
                ip = "vastai"
            
            data = {
                "name": self.miner_name,
                "ip_address": ip,
                "gpu_name": gpu_info['name'],
                "gpu_memory_mb": gpu_info['memory_mb'],
                "status": "idle"
            }
            
            logger.info(f"Registering: {self.miner_name}")
            logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_mb']}MB)")
            
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                self.miner_id = r.json().get('miner_id')
                logger.info(f"✓ Registered! ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code}")
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
    
    def process_job(self, job):
        try:
            self.update_status("busy")
            
            logger.info(f"Processing job: {job['request_id']}")
            
            r = requests.post(
                f"{self.local_url}/generate",
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
                        "processing_time": result.get("generation_time", 0)
                    }
                }
                
                submit_r = requests.post(
                    f"{self.backend_url}/submit_result",
                    json=submit_data,
                    timeout=30
                )
                
                if submit_r.status_code == 200:
                    speed = result.get('tokens_per_second', 0)
                    logger.info(f"✓ Job complete: {speed:.1f} tok/s")
                    return True
                    
        except Exception as e:
            logger.error(f"Job error: {e}")
        finally:
            self.update_status("idle")
        return False
    
    def run_mining_loop(self):
        logger.info("Starting mining loop...")
        errors = 0
        
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
                        if self.process_job(work):
                            errors = 0
                        else:
                            errors += 1
                    else:
                        errors = 0
                else:
                    errors += 1
                
                if errors > 10:
                    logger.error("Too many errors, pausing...")
                    time.sleep(60)
                    errors = 0
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                errors += 1
                time.sleep(min(errors * 5, 60))

def main():
    logger.info("=" * 60)
    logger.info("MIA GPU Miner - Simple Fast Version")
    logger.info("=" * 60)
    
    # Load model
    server = ModelServer()
    if not server.load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Start server
    server.start_server()
    
    # Initialize client
    client = MinerClient()
    
    # Wait for server
    if not client.wait_for_model():
        logger.error("Server failed to start")
        sys.exit(1)
    
    # Test multilingual
    client.test_multilingual()
    
    # Register
    attempts = 0
    while not client.register():
        attempts += 1
        if attempts > 5:
            logger.error("Registration failed")
            sys.exit(1)
        logger.info(f"Retrying in 30s... ({attempts}/5)")
        time.sleep(30)
    
    # Start mining
    try:
        client.run_mining_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x mia_miner_unified.py

# Create simple run script
cat > run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/libs:$PYTHONPATH"
python3 mia_miner_unified.py
EOF
chmod +x run.sh

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ MIA GPU Miner installed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Features:${NC}"
echo "• Simple installation (no complex dependencies)"
echo "• GPTQ model for good speed"
echo "• Multilingual support (tested)"
echo "• PyTorch optimizations enabled"
echo "• ~4-5GB VRAM usage"
echo ""
echo -e "${YELLOW}To start mining:${NC}"
echo "  cd $INSTALL_DIR"
echo "  ./run.sh"
echo ""
echo -e "${GREEN}Happy mining!${NC}"