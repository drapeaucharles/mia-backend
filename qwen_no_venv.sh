#!/bin/bash

# MIA Qwen Miner - NO VIRTUAL ENVIRONMENT
# Direct installation for simplicity

echo "🚀 MIA Qwen Miner - Direct Installation (No venv)"
echo "==============================================="
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
apt update -qq && apt install -y python3 python3-pip git curl

# Install Python packages directly
echo ""
echo "📦 Installing Python packages..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers accelerate flask waitress requests bitsandbytes

# Create miner directory
INSTALL_DIR="/data/mia-qwen"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create the miner script
echo ""
echo "📝 Creating Qwen miner..."
cat > qwen_miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Qwen2.5-7B-Instruct"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

import torch
import time
import logging
import requests
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-qwen')

app = Flask(__name__)
model = None
tokenizer = None

# Backend URL
backend_url = "https://mia-backend-production.up.railway.app"

def load_model():
    global model, tokenizer
    
    logger.info("Loading Qwen2.5-7B-Instruct...")
    
    # 4-bit quantization for speed
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info("✓ Model loaded successfully!")
    return True

@app.route("/api/generate", methods=["POST"])
def generate():
    """MIA Backend compatible endpoint"""
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
        
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 500)
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new text
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return jsonify({
        "text": response,
        "response": response,
        "answer": response
    })

@app.route("/generate", methods=["POST"])
def generate_local():
    """Local testing endpoint"""
    return generate()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "loading",
        "model": "Qwen2.5-7B-4bit"
    })

# Auto-register miner
def register_miner():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    else:
        gpu_name = "CPU"
        vram = 0
    
    try:
        response = requests.post(
            f"{backend_url}/register_miner",
            json={
                "address": "http://$(hostname -I | awk '{print $1}'):8000",
                "gpu_model": gpu_name,
                "vram_gb": vram
            }
        )
        if response.status_code == 200:
            miner_id = response.json()["miner_id"]
            logger.info(f"✓ Registered as miner: {miner_id}")
            return miner_id
    except:
        logger.warning("Could not register with backend")
    return None

# Job fetching loop
def job_loop(miner_id):
    while True:
        try:
            # Get job
            job_resp = requests.get(f"{backend_url}/job/next?miner_id={miner_id}")
            if job_resp.status_code != 200:
                time.sleep(5)
                continue
                
            job = job_resp.json()
            job_id = job["job_id"]
            prompt = job["prompt"]
            
            logger.info(f"Processing job {job_id}")
            
            # Generate response
            result = generate_response(prompt)
            
            # Submit result
            requests.post(
                f"{backend_url}/submit_result",
                json={
                    "job_id": job_id,
                    "result": result,
                    "miner_id": miner_id,
                    "success": True
                }
            )
            
        except Exception as e:
            logger.error(f"Job loop error: {e}")
            time.sleep(10)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response

if __name__ == "__main__":
    # Load model
    if not load_model():
        exit(1)
    
    # Register miner
    miner_id = register_miner()
    
    # Start job fetching in background
    if miner_id:
        import threading
        threading.Thread(target=job_loop, args=(miner_id,), daemon=True).start()
    
    # Start server
    logger.info("Starting server on port 8000...")
    serve(app, host="0.0.0.0", port=8000, threads=8)
EOF

# Make executable
chmod +x qwen_miner.py

# Stop any existing miners
echo ""
echo "🛑 Stopping existing miners..."
pkill -f "miner" || true
pkill -f "8000" || true

# Start the miner
echo ""
echo "🚀 Starting Qwen miner..."
nohup python3 qwen_miner.py > miner.log 2>&1 &
PID=$!

echo ""
echo "✅ Qwen miner started!"
echo ""
echo "PID: $PID"
echo "Logs: tail -f $INSTALL_DIR/miner.log"
echo "Stop: kill $PID"
echo ""
echo "Testing in 20 seconds..."
sleep 20

# Test
curl -X POST http://localhost:8000/health || echo "Still loading model..."