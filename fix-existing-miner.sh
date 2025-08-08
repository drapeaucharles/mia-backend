#!/bin/bash

# Fix for existing miner installation
# This updates an already installed miner to use the unified approach

echo "Fixing existing MIA miner installation..."

# Check if miner is already installed
if [ ! -d "/opt/mia-gpu-miner" ]; then
    echo "Error: Miner not found at /opt/mia-gpu-miner"
    echo "Please run the full installer first"
    exit 1
fi

cd /opt/mia-gpu-miner

# Stop existing services
echo "Stopping existing services..."
sudo systemctl stop mia-gpu-miner 2>/dev/null || true
sudo systemctl stop mia-miner 2>/dev/null || true

# Kill any running processes
pkill -f "vllm_server.py" 2>/dev/null || true
pkill -f "gpu_miner.py" 2>/dev/null || true
pkill -f "mia_miner_unified.py" 2>/dev/null || true

# Create the unified miner script
echo "Creating unified miner..."
cat > /opt/mia-gpu-miner/mia_miner_unified.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Unified GPU Miner - Fixed Version
Combines model server and client in one process
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-unified')

# Flask app for model server
app = Flask(__name__)

# Global model variables
model = None
tokenizer = None
model_ready = False

def load_model():
    """Load the AI model"""
    global model, tokenizer, model_ready
    
    logger.info("Loading AI model...")
    
    try:
        # First try vLLM if available
        try:
            from vllm import LLM, SamplingParams
            logger.info("Using vLLM for inference")
            
            model = LLM(
                model="NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
            )
            model_ready = True
            logger.info("✓ vLLM model loaded successfully!")
            return True
            
        except ImportError:
            logger.info("vLLM not available, using transformers")
            
        # Fallback to transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "teknium/OpenHermes-2.5-Mistral-7B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        model_ready = True
        logger.info("✓ Transformers model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model_ready else "loading",
        "model": "Mistral-7B"
    })

@app.route("/generate", methods=["POST"])
def generate():
    if not model_ready:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        
        # Check if using vLLM
        if hasattr(model, 'generate') and hasattr(model, 'get_tokenizer'):
            # vLLM path
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=max_tokens,
            )
            
            outputs = model.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            
        else:
            # Transformers path
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "<|im_start|>assistant" in response_text:
                response_text = response_text.split("<|im_start|>assistant")[-1].strip()
            
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        
        return jsonify({
            "text": response_text,
            "tokens_generated": tokens_generated,
            "model": "Mistral-7B"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

def start_server():
    """Start the model server in a thread"""
    logger.info("Starting inference server on port 8000...")
    serve(app, host="0.0.0.0", port=8000, threads=4)

class MinerClient:
    """Miner client"""
    
    def __init__(self):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        self.miner_id = None
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'name': gpu.name,
                    'memory_mb': int(gpu.memoryTotal)
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
                    'memory_mb': int(parts[1].strip().replace(' MiB', ''))
                }
        except:
            pass
            
        return {'name': 'Unknown GPU', 'memory_mb': 0}
    
    def register(self):
        """Register with backend"""
        try:
            gpu_info = self.get_gpu_info()
            
            # Get IP
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
            
            logger.info(f"Registering: {self.miner_name}")
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                resp = r.json()
                self.miner_id = resp.get('miner_id')
                logger.info(f"✓ Registered! ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def update_status(self, status):
        """Update status"""
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
        """Main mining loop"""
        # Wait for model
        logger.info("Waiting for model server...")
        for i in range(60):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200 and r.json().get("status") == "ready":
                    logger.info("✓ Model server ready")
                    break
            except:
                pass
            time.sleep(5)
        else:
            logger.error("Model server timeout")
            return
        
        # Register
        attempts = 0
        while not self.register():
            attempts += 1
            if attempts > 5:
                logger.error("Registration failed")
                return
            time.sleep(30)
        
        # Mining loop
        logger.info("Starting mining loop...")
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
                        logger.info(f"Processing: {work['request_id']}")
                        self.update_status("busy")
                        
                        # Generate
                        start = time.time()
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
                            elapsed = time.time() - start
                            
                            # Submit
                            submit_r = requests.post(
                                f"{self.backend_url}/submit_result",
                                json={
                                    "miner_id": self.miner_id,
                                    "request_id": work["request_id"],
                                    "result": {
                                        "response": result.get("text", ""),
                                        "tokens_generated": result.get("tokens_generated", 0),
                                        "processing_time": elapsed
                                    }
                                },
                                timeout=30
                            )
                            
                            if submit_r.status_code == 200:
                                logger.info(f"✓ Completed in {elapsed:.2f}s")
                                errors = 0
                        
                        self.update_status("idle")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                errors += 1
                logger.error(f"Error: {e}")
                if errors > 10:
                    break
                time.sleep(min(errors * 5, 60))

def main():
    """Main entry point"""
    logger.info("=" * 50)
    logger.info("MIA Unified GPU Miner")
    logger.info("=" * 50)
    
    # Load model
    if not load_model():
        logger.error("Model load failed")
        sys.exit(1)
    
    # Start server in thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(5)
    
    # Run client
    client = MinerClient()
    client.run()

if __name__ == "__main__":
    main()
EOF

chmod +x /opt/mia-gpu-miner/mia_miner_unified.py

# Create simple start script
cat > /opt/mia-gpu-miner/start.sh << 'EOF'
#!/bin/bash
cd /opt/mia-gpu-miner
source venv/bin/activate
exec python mia_miner_unified.py
EOF

chmod +x /opt/mia-gpu-miner/start.sh

# Update systemd service
sudo tee /etc/systemd/system/mia-miner.service > /dev/null << EOF
[Unit]
Description=MIA Unified GPU Miner
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/opt/mia-gpu-miner
Environment="PATH=/opt/mia-gpu-miner/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/mia-gpu-miner/start.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Reload and start
sudo systemctl daemon-reload
sudo systemctl enable mia-miner.service
sudo systemctl start mia-miner.service

echo "✓ Fixed! The unified miner is now running."
echo ""
echo "Check status: sudo systemctl status mia-miner"
echo "View logs:    sudo journalctl -u mia-miner -f"