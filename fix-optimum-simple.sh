#!/bin/bash

# Quick fix for missing optimum library

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Installing missing dependencies...${NC}"

cd /data/mia-gpu-miner

# Activate venv
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
fi

# Install optimum
echo -e "${YELLOW}Installing optimum library...${NC}"
pip install optimum

# Also install accelerate which is often needed
pip install accelerate

# Create a simple working miner that avoids GPTQ complexity
echo -e "${YELLOW}Creating simplified miner...${NC}"
cat > mia_miner_simple.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Simplified Version
Avoids GPTQ complexity issues
"""
import os
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
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/miner.log')
    ]
)
logger = logging.getLogger('mia')

# Global model
model = None
tokenizer = None
app = Flask(__name__)

def load_model():
    global model, tokenizer
    
    logger.info("Loading model (simplified approach)...")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
        device = "cuda:0"
    else:
        logger.error("No CUDA!")
        device = "cpu"
    
    try:
        # Option 1: Try loading GPTQ model directly
        logger.info("Attempting to load GPTQ model...")
        tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            "/data/models/mistral-gptq",
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        logger.info("✓ GPTQ model loaded!")
        
    except Exception as e:
        logger.warning(f"GPTQ failed: {e}")
        
        # Option 2: Load original model with 8-bit quantization
        logger.info("Falling back to 8-bit quantization...")
        try:
            # Install bitsandbytes if needed
            os.system("pip install bitsandbytes")
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                "Open-Orca/Mistral-7B-OpenOrca",
                quantization_config=bnb_config,
                device_map=device,
                cache_dir="/data/huggingface"
            )
            logger.info("✓ 8-bit model loaded!")
            
        except Exception as e2:
            logger.error(f"All loading methods failed: {e2}")
            return False
    
    model.eval()
    
    # Test
    logger.info("Testing...")
    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    elapsed = time.time() - start
    logger.info(f"Speed: {20/elapsed:.1f} tok/s")
    
    return True

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
        
        # Format
        formatted = f"""<|im_start|>system
You are MIA, a helpful multilingual assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        # Generate
        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        response = response.replace("<|im_end|>", "").strip()
        
        elapsed = time.time() - start
        tokens = len(generated_ids)
        
        return jsonify({
            "text": response,
            "tokens_generated": int(tokens),
            "time": round(elapsed, 2),
            "tokens_per_second": round(tokens/elapsed, 1)
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    logger.info("MIA Miner - Simplified Version")
    
    # Load model
    if not load_model():
        logger.error("Failed to load model")
        sys.exit(1)
    
    # Start server
    def run_server():
        serve(app, host="0.0.0.0", port=8000)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    logger.info("Server started on port 8000")
    
    # Simple mining loop
    backend_url = "https://mia-backend-production.up.railway.app"
    miner_name = f"gpu-miner-{socket.gethostname()}"
    miner_id = None
    
    # Register
    logger.info("Registering...")
    for attempt in range(5):
        try:
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            gpu_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0
            
            r = requests.post(
                f"{backend_url}/register_miner",
                json={
                    "name": miner_name,
                    "ip_address": "vastai",
                    "gpu_name": gpu_name,
                    "gpu_memory_mb": gpu_mb,
                    "status": "idle"
                },
                timeout=30
            )
            
            if r.status_code == 200:
                miner_id = r.json().get('miner_id')
                logger.info(f"✓ Registered! ID: {miner_id}")
                break
        except Exception as e:
            logger.error(f"Registration attempt {attempt+1} failed: {e}")
            time.sleep(30)
    
    if not miner_id:
        logger.error("Failed to register")
        sys.exit(1)
    
    # Mining loop
    logger.info("Starting mining loop...")
    while True:
        try:
            # Get work
            r = requests.get(
                f"{backend_url}/get_work",
                params={"miner_id": miner_id},
                timeout=10
            )
            
            if r.status_code == 200:
                work = r.json()
                if work and work.get("request_id"):
                    logger.info(f"Processing job {work['request_id']}")
                    
                    # Generate
                    gen_r = requests.post(
                        "http://localhost:8000/generate",
                        json={
                            "prompt": work.get("prompt", ""),
                            "max_tokens": work.get("max_tokens", 200)
                        },
                        timeout=60
                    )
                    
                    if gen_r.status_code == 200:
                        result = gen_r.json()
                        
                        # Submit
                        requests.post(
                            f"{backend_url}/submit_result",
                            json={
                                "miner_id": miner_id,
                                "request_id": work["request_id"],
                                "result": {
                                    "response": result.get("text", ""),
                                    "tokens_generated": result.get("tokens_generated", 0),
                                    "processing_time": result.get("time", 0)
                                }
                            },
                            timeout=30
                        )
                        
                        logger.info(f"✓ Job complete ({result.get('tokens_per_second', 0):.1f} tok/s)")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
EOF

chmod +x mia_miner_simple.py

echo ""
echo -e "${GREEN}✓ Dependencies installed!${NC}"
echo ""
echo "Now try running the simplified miner:"
echo "  ./stop_miner.sh"
echo "  python3 mia_miner_simple.py"
echo ""
echo "Or if that works, update the start script:"
echo "  sed -i 's/mia_miner_unified.py/mia_miner_simple.py/g' start_miner.sh"
echo "  ./start_miner.sh"