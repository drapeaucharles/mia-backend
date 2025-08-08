#!/bin/bash

# MIA GPU Miner Installer - For /data Volume Mount
# Specifically for Vast.ai with limited root but /data volume

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - /data Volume Install    ║${NC}"
echo -e "${GREEN}║        Optimized for Vast.ai              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Check if /data exists
if [ ! -d "/data" ]; then
    echo -e "${RED}Error: /data volume not found!${NC}"
    echo "Please ensure your Vast.ai instance has a volume mounted at /data"
    exit 1
fi

# Check available space
AVAILABLE_SPACE=$(df -BG /data | tail -1 | awk '{print $4}' | sed 's/G//')
echo -e "${YELLOW}Available space on /data: ${AVAILABLE_SPACE}GB${NC}"

if [ "$AVAILABLE_SPACE" -lt "10" ]; then
    echo -e "${RED}Warning: Less than 10GB available on /data${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to /data
cd /data

# Create project directory
INSTALL_DIR="/data/mia-gpu-miner"
echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check Python version and install venv if needed
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${YELLOW}Python version: $PYTHON_VERSION${NC}"

# Install python3-venv if needed (for root access)
if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
    apt-get update && apt-get install -y python3-venv python${PYTHON_VERSION}-venv 2>/dev/null || true
fi

# Create virtual environment in /data
echo -e "${YELLOW}Creating virtual environment in /data/venv...${NC}"
if ! python3 -m venv /data/venv; then
    echo -e "${RED}Failed to create venv. Trying without...${NC}"
    USE_VENV=false
else
    USE_VENV=true
    source /data/venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment created and activated${NC}"
fi

# Add auto-activation to .bashrc
if [ "$USE_VENV" = true ]; then
    if ! grep -q "source /data/venv/bin/activate" ~/.bashrc; then
        echo 'source /data/venv/bin/activate' >> ~/.bashrc
        echo -e "${GREEN}✓ Added venv activation to .bashrc${NC}"
    fi
fi

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip

# Install PyTorch based on Python version
echo -e "${YELLOW}Installing PyTorch...${NC}"
if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    # Python 3.8 compatible
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
else
    # Latest PyTorch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Test CUDA
echo -e "${YELLOW}Testing CUDA availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Not detected\"}')"

# Install transformers and dependencies
echo -e "${YELLOW}Installing AI libraries...${NC}"
if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    # Python 3.8 compatible versions
    pip install transformers==4.30.0 accelerate==0.20.0
else
    # Latest versions
    pip install transformers accelerate
fi

pip install flask waitress requests sentencepiece protobuf huggingface-hub

# Try to install auto-gptq (may fail, that's ok)
echo -e "${YELLOW}Attempting to install auto-gptq...${NC}"
pip install auto-gptq || echo -e "${YELLOW}auto-gptq installation failed, will use standard transformers${NC}"

# Download model to /data
echo -e "${YELLOW}Downloading model to /data...${NC}"
cat > download_model.py << 'EOF'
import os
os.environ["HF_HOME"] = "/data/huggingface"

from huggingface_hub import snapshot_download

print("Downloading Mistral-7B-OpenOrca-GPTQ to /data...")
model_path = snapshot_download(
    repo_id="TheBloke/Mistral-7B-OpenOrca-GPTQ",
    local_dir="/data/models/mistral-gptq",
    cache_dir="/data/huggingface/hub",
    resume_download=True,
    ignore_patterns=["*.md", "*.txt", ".gitattributes"]
)
print(f"Model downloaded to: {model_path}")
EOF

python3 download_model.py

# Create the miner script
echo -e "${YELLOW}Creating MIA miner...${NC}"
cat > mia_miner_unified.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - /data Volume Version
"""
import os

# Set all cache directories to /data
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/mia-miner.log')
    ]
)
logger = logging.getLogger('mia-unified')

# Reduce transformers logging
logging.getLogger("transformers").setLevel(logging.WARNING)

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
        
        logger.info("Loading model from /data/models...")
        
        try:
            # Check CUDA
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                device = "cuda:0"
            else:
                logger.warning("CUDA not detected! Will be very slow on CPU")
                device = "cpu"
            
            # Load tokenizer
            tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
            logger.info(f"Loading tokenizer from {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                cache_dir="/data/huggingface"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Try to load with auto-gptq first
            model_path = "/data/models/mistral-gptq"
            logger.info(f"Loading model from {model_path}")
            
            try:
                from auto_gptq import AutoGPTQForCausalLM
                logger.info("Using AutoGPTQ for faster inference...")
                model = AutoGPTQForCausalLM.from_quantized(
                    model_path,
                    device="cuda:0",
                    use_triton=False,
                    use_safetensors=True,
                    trust_remote_code=True,
                    inject_fused_attention=False
                )
            except Exception as e:
                logger.info(f"AutoGPTQ not available ({e}), using standard transformers...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            model.eval()
            self.model_loaded = True
            logger.info("✓ Model loaded successfully!")
            
            # Quick performance test
            self._test_speed()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _test_speed(self):
        """Test inference speed"""
        logger.info("Testing inference speed...")
        test_prompts = ["Hello", "Bonjour", "Hola", "你好"]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            elapsed = time.time() - start
            
            tokens = 20
            speed = tokens / elapsed
            logger.info(f"'{prompt}': {elapsed:.2f}s ({speed:.1f} tok/s)")
    
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
        
        # Use proper ChatML format
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses in multiple languages."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
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
        
        # Decode only generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>", "").strip()
        
        tokens_generated = len(generated_ids)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_generated/generation_time:.1f} tok/s)")
        
        return jsonify({
            "text": response,
            "tokens_generated": int(tokens_generated),
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
            if torch.cuda.is_available():
                return {
                    'name': torch.cuda.get_device_name(0),
                    'memory_mb': torch.cuda.get_device_properties(0).total_memory // (1024*1024)
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
    logger.info("MIA Unified GPU Miner - /data Volume Version")
    logger.info("=" * 60)
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: /data/models/mistral-gptq")
    logger.info(f"Cache path: /data/huggingface")
    logger.info(f"Log path: /data/mia-miner.log")
    
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

# Activate virtual environment if it exists
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
fi

# Set cache directories to /data
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"

# Run the miner
python3 mia_miner_unified.py
EOF
chmod +x run_miner.sh

# Create a systemd-style start script
cat > start_miner.sh << 'EOF'
#!/bin/bash
cd /data/mia-gpu-miner

# Activate virtual environment if it exists
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
fi

# Set cache directories to /data
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"

# Start miner in background
nohup python3 mia_miner_unified.py > /data/miner.log 2>&1 &
echo $! > /data/miner.pid
echo "Miner started with PID $(cat /data/miner.pid)"
echo "Logs: tail -f /data/miner.log"
EOF
chmod +x start_miner.sh

# Create stop script
cat > stop_miner.sh << 'EOF'
#!/bin/bash
if [ -f "/data/miner.pid" ]; then
    PID=$(cat /data/miner.pid)
    kill $PID 2>/dev/null && echo "Miner stopped (PID $PID)" || echo "Miner not running"
    rm -f /data/miner.pid
else
    echo "No miner PID file found"
fi
EOF
chmod +x stop_miner.sh

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ MIA GPU Miner installed to /data volume!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Installation Summary:${NC}"
echo "• Virtual environment: /data/venv"
echo "• Miner location: /data/mia-gpu-miner"
echo "• Model location: /data/models/mistral-gptq"
echo "• Cache location: /data/huggingface"
echo "• Log file: /data/miner.log"
echo ""
echo -e "${YELLOW}To start the miner:${NC}"
echo ""
echo "Option 1 - Interactive mode:"
echo "  cd /data/mia-gpu-miner"
echo "  ./run_miner.sh"
echo ""
echo "Option 2 - Background mode:"
echo "  cd /data/mia-gpu-miner"
echo "  ./start_miner.sh"
echo ""
echo "Option 3 - View logs:"
echo "  tail -f /data/miner.log"
echo ""
echo "Option 4 - Stop miner:"
echo "  cd /data/mia-gpu-miner"
echo "  ./stop_miner.sh"
echo ""
echo -e "${GREEN}The virtual environment will auto-activate on login!${NC}"
echo -e "${GREEN}All data is stored on the /data volume, not the root partition.${NC}"