#!/bin/bash

# Fix ALL issues in existing MIA miners
# This combines: PyTorch install, tokenizer fix, and response extraction fix

echo "==================================="
echo "MIA Miner - Complete Fix Script"
echo "==================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect installation directory
if [ -d "/opt/mia-gpu-miner" ]; then
    INSTALL_DIR="/opt/mia-gpu-miner"
    echo "Found installation at /opt/mia-gpu-miner"
elif [ -d "$HOME/mia-gpu-miner" ]; then
    INSTALL_DIR="$HOME/mia-gpu-miner"
    echo "Found installation at ~/mia-gpu-miner"
else
    echo -e "${RED}Error: No miner installation found${NC}"
    echo "Please run the installer first:"
    echo "  bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner-unified.sh)"
    exit 1
fi

cd "$INSTALL_DIR"

# Stop any running miners
echo -e "\n${YELLOW}Stopping existing miners...${NC}"
sudo systemctl stop mia-miner 2>/dev/null || true
sudo systemctl stop mia-gpu-miner 2>/dev/null || true
pkill -f "mia_miner_unified.py" 2>/dev/null || true
sleep 2

# Activate venv
source venv/bin/activate

# STEP 1: Fix PyTorch Installation
echo -e "\n${GREEN}=== STEP 1: Fixing PyTorch Installation ===${NC}"
echo -e "${YELLOW}Installing PyTorch with CUDA 11.8...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
python -c "
import torch
print(f'✓ PyTorch {torch.__version__} installed')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
" || exit 1

# Install auto-gptq
echo -e "${YELLOW}Installing auto-gptq...${NC}"
pip uninstall -y auto-gptq 2>/dev/null || true
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# STEP 2: Update all packages
echo -e "\n${GREEN}=== STEP 2: Updating AI packages ===${NC}"
pip install --upgrade transformers accelerate optimum sentencepiece protobuf
pip install --upgrade flask waitress requests

# STEP 3: Fix the miner code
echo -e "\n${GREEN}=== STEP 3: Fixing miner code ===${NC}"

# Create the fixed unified miner
cat > "$INSTALL_DIR/mia_miner_unified.py" << 'EOF'
#!/usr/bin/env python3
"""
MIA Unified GPU Miner - Fixed Version
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
        # IMPORTANT: Use the original model's tokenizer, not the GPTQ one
        tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
        model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
        
        try:
            logger.info(f"Loading tokenizer from {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Loading GPTQ model from {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                revision="main"  # main branch is 4-bit-128g-actorder
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
        
        # Generate response
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
        
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        
        logger.info(f"Generated {tokens_generated} tokens")
        
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
    
    def test_generation(self):
        """Test the generation with a simple prompt"""
        logger.info("Testing generation with 'Hello'...")
        try:
            r = requests.post(
                f"{self.local_url}/generate",
                json={"prompt": "Hello", "max_tokens": 50},
                timeout=30
            )
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Test response: {result.get('text', 'No response')}")
                return True
            else:
                logger.error(f"Test failed: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Test error: {e}")
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
    
    # Test generation
    logger.info("Testing model generation...")
    miner.test_generation()
    
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

# Create/update start script
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
exec python mia_miner_unified.py
EOF

chmod +x "$INSTALL_DIR/start.sh"

# Update run_miner.sh if it exists
if [ -f "$INSTALL_DIR/run_miner.sh" ]; then
    cp "$INSTALL_DIR/start.sh" "$INSTALL_DIR/run_miner.sh"
fi

# STEP 4: Test the fixes
echo -e "\n${GREEN}=== STEP 4: Testing fixes ===${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

python -c "
print('Testing imports...')
import torch
import auto_gptq
from transformers import AutoTokenizer, AutoModelForCausalLM
print('✓ All imports successful')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"

# STEP 5: Restart the miner
echo -e "\n${GREEN}=== STEP 5: Starting miner ===${NC}"

# Check if systemd service exists
if [ -f "/etc/systemd/system/mia-miner.service" ]; then
    # Update service file to use new start script
    sudo tee /etc/systemd/system/mia-miner.service > /dev/null << EOF
[Unit]
Description=MIA Unified GPU Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$INSTALL_DIR/start.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl restart mia-miner
    
    echo -e "${GREEN}✓ Miner service restarted${NC}"
    echo ""
    echo "Check status: sudo systemctl status mia-miner"
    echo "View logs:    sudo journalctl -u mia-miner -f"
else
    echo -e "${GREEN}✓ All fixes applied!${NC}"
    echo ""
    echo "To start the miner:"
    echo "  cd $INSTALL_DIR"
    echo "  ./start.sh"
    echo ""
    echo "Or run in background:"
    echo "  screen -dmS mia-miner $INSTALL_DIR/start.sh"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ ALL FIXES APPLIED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Your miner now has:"
echo "  ✓ PyTorch with CUDA support"
echo "  ✓ Proper tokenizer (Open-Orca/Mistral-7B-OpenOrca)"
echo "  ✓ Correct ChatML prompt format"
echo "  ✓ Fixed response extraction"
echo "  ✓ Better generation parameters"
echo ""
echo "The model should now respond naturally:"
echo "  'Hello' → 'Hello! How can I assist you today?'"