#!/bin/bash

# MIA GPU Miner Installer - GGUF Version for Vast.ai
# Uses llama-cpp-python for better compatibility

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - GGUF (Vast.ai Edition)  ║${NC}"
echo -e "${GREEN}║    Reliable GPU Inference with llama.cpp  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Create directory
INSTALL_DIR="$HOME/mia-gpu-miner"
echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Install Python if needed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Installing Python...${NC}"
    apt-get update && apt-get install -y python3 python3-pip python3-venv wget
fi

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install llama-cpp-python with CUDA support
echo -e "${YELLOW}Installing llama-cpp-python with GPU support...${NC}"
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install other dependencies
echo -e "${YELLOW}Installing other dependencies...${NC}"
pip install flask waitress requests huggingface-hub

# Download GGUF model
echo -e "${YELLOW}Downloading Mistral 7B OpenOrca GGUF model...${NC}"
python3 << 'EOF'
from huggingface_hub import hf_hub_download
import os

model_id = "TheBloke/Mistral-7B-OpenOrca-GGUF"
filename = "mistral-7b-openorca.Q4_K_M.gguf"  # 4-bit quantized, ~4GB

print(f"Downloading {filename}...")
model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    local_dir="./models",
    resume_download=True
)
print(f"Model downloaded to: {model_path}")
EOF

# Create the miner script
echo -e "${YELLOW}Creating MIA miner with GGUF model...${NC}"
cat > mia_miner_unified.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Unified GPU Miner - GGUF Version (Vast.ai)
Uses llama-cpp-python for reliable GPU inference
"""
import os
import sys
import time
import json
import socket
import logging
import requests
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify
from waitress import serve
from llama_cpp import Llama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-unified')

# Global model variable
model = None
app = Flask(__name__)

class ModelServer:
    """Embedded model server"""
    
    def __init__(self):
        self.server_thread = None
        self.model_loaded = False
    
    def load_model(self):
        """Load the GGUF model"""
        global model
        
        logger.info("Loading GGUF model with llama.cpp...")
        
        try:
            # Model path
            model_path = "./models/mistral-7b-openorca.Q4_K_M.gguf"
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return False
            
            # Load model with GPU acceleration
            logger.info("Initializing model with GPU layers...")
            model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Use all layers on GPU
                n_ctx=2048,       # Context size
                n_batch=512,      # Batch size for prompt processing
                verbose=True      # Show loading progress
            )
            
            self.model_loaded = True
            logger.info("✓ GGUF Model loaded successfully!")
            
            # Quick performance test
            logger.info("Testing inference speed...")
            start = time.time()
            response = model("Hello", max_tokens=20)
            elapsed = time.time() - start
            tokens = len(response['choices'][0]['text'].split())
            logger.info(f"Test generation: {elapsed:.2f}s for ~{tokens} tokens ({tokens/elapsed:.1f} tok/s)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Make sure CUDA is properly installed on your Vast.ai instance")
            return False
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        def run_server():
            logger.info("Starting inference server on port 8000...")
            serve(app, host="0.0.0.0", port=8000, threads=4)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(5)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "model": "Mistral-7B-OpenOrca-GGUF"
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        
        # Format prompt in ChatML format
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        # Generate response
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        start_time = time.time()
        
        response = model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"]
        )
        
        # Extract text
        generated_text = response['choices'][0]['text'].strip()
        tokens_generated = response['usage']['completion_tokens']
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.1f} tok/s)")
        
        return jsonify({
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "model": "Mistral-7B-OpenOrca-GGUF"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

class MinerClient:
    """Miner client that connects to backend"""
    
    def __init__(self, model_server):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}-gguf"
        self.miner_id = None
        self.model_server = model_server
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            # Try nvidia-smi first
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
        return {'name': 'Vast.ai GPU', 'memory_mb': 8192}
    
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
                ip = "vastai"
            
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
    logger.info("MIA Unified GPU Miner - GGUF Version (Vast.ai)")
    logger.info("=" * 60)
    
    # Set environment for GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
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

chmod +x mia_miner_unified.py

# Create run script
cat > run_miner.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python mia_miner_unified.py
EOF
chmod +x run_miner.sh

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ MIA GPU Miner (GGUF) installed for Vast.ai!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}GGUF Benefits:${NC}"
echo "• More reliable GPU detection"
echo "• Fast inference with llama.cpp"
echo "• ~4GB model size"
echo "• Works even with CUDA detection issues"
echo ""
echo -e "${YELLOW}To start the miner:${NC}"
echo "  cd $INSTALL_DIR"
echo "  ./run_miner.sh"
echo ""
echo -e "${GREEN}Your miner is ready to mine for MIA!${NC}"