#!/bin/bash

# MIA GPU Miner - Qwen2.5-7B-Instruct
# Modern multilingual model with excellent performance

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MIA GPU Miner - Qwen2.5-7B Setup        ║${NC}"
echo -e "${BLUE}║   29 Languages + Business Optimized       ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system deps if needed
echo -e "${YELLOW}Checking system dependencies...${NC}"
which python3 >/dev/null || apt install -y python3
which pip3 >/dev/null || apt install -y python3-pip
apt install -y python3-venv python3-dev 2>/dev/null || true

# Kill previous miners
pkill -f miner 2>/dev/null || true

# Setup directories
INSTALL_DIR="/data/mia-qwen"
VENV_DIR="/data/venv-qwen"

echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
rm -rf "$VENV_DIR"  # Clean install
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create fresh virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch (auto-detect CUDA)
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers with optimization libraries
echo -e "${YELLOW}Installing inference stack...${NC}"
pip install transformers>=4.37.0 accelerate bitsandbytes optimum
pip install flask waitress requests aiohttp
pip install sentencepiece protobuf  # Required for Qwen tokenizer

# Create Qwen miner
echo -e "${YELLOW}Creating Qwen2.5 miner...${NC}"
cat > qwen_miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Qwen2.5-7B-Instruct"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-qwen')

app = Flask(__name__)

# Load Qwen2.5 with optimization
logger.info("Loading Qwen2.5-7B-Instruct...")
model_id = "Qwen/Qwen2.5-7B-Instruct"

# Use 4-bit quantization for speed + quality
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    logger.info("✓ Qwen2.5-7B loaded successfully!")
    logger.info("  - 29 language support")
    logger.info("  - 4-bit quantization for 40+ tok/s")
    logger.info("  - Optimized for business queries")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Ensure tokenizer has padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.7)
        
        # Detect language and format appropriately
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1
            )
        gen_time = time.time() - start
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        else:
            response = response[len(text):].strip()
        
        tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        tps = tokens / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generated {tokens} tokens at {tps:.1f} tok/s")
        
        return jsonify({
            'text': response,
            'tokens_generated': tokens,
            'tokens_per_second': tps
        })
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

def worker():
    """Background worker for MIA backend"""
    backend_url = "https://mia-backend-production.up.railway.app"
    miner_id = None
    
    # Register with backend
    try:
        resp = requests.post(
            f"{backend_url}/register_miner",
            json={"name": f"qwen2.5-{os.getpid()}"},
            timeout=10
        )
        if resp.status_code == 200:
            miner_id = resp.json()['miner_id']
            logger.info(f"✓ Registered with MIA backend as miner {miner_id}")
    except Exception as e:
        logger.warning(f"Registration failed: {e}, using fallback ID")
        miner_id = 1
    
    # Work loop
    while True:
        try:
            # Get work from backend
            work_resp = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if work_resp.status_code == 200:
                work = work_resp.json()
                
                if work and 'request_id' in work:
                    logger.info(f"Processing job {work['request_id']}")
                    
                    # Generate response
                    prompt = work.get('prompt', '')
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = tokenizer(text, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=work.get('max_tokens', 150),
                            temperature=work.get('temperature', 0.7),
                            do_sample=True
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "assistant" in response:
                        response = response.split("assistant")[-1].strip()
                    else:
                        response = response[len(text):].strip()
                    
                    # Submit result
                    submit_resp = requests.post(
                        f"{backend_url}/submit_result",
                        json={
                            'miner_id': int(miner_id),
                            'request_id': work['request_id'],
                            'result': {'response': response}
                        },
                        timeout=10
                    )
                    
                    if submit_resp.status_code == 200:
                        logger.info("✓ Result submitted successfully")
                    else:
                        logger.error(f"Failed to submit result: {submit_resp.status_code}")
                        
        except requests.exceptions.Timeout:
            pass  # Normal - no work available
        except Exception as e:
            logger.error(f"Worker error: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    # Start background worker
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    logger.info("✓ Worker thread started")
    
    # Start Flask server
    logger.info("Starting Flask server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

chmod +x qwen_miner.py

# Create auto-restart script
cat > start_qwen.sh << 'EOF'
#!/bin/bash
cd /data/mia-qwen
source /data/venv-qwen/bin/activate
export HF_HOME="/data/huggingface"

while true; do
    echo "Starting Qwen miner..."
    python3 qwen_miner.py
    echo "Miner stopped, restarting in 5 seconds..."
    sleep 5
done
EOF
chmod +x start_qwen.sh

# Start the miner
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Starting Qwen2.5-7B miner...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

# Kill any existing
pkill -f qwen_miner 2>/dev/null || true

# Start new miner
nohup ./start_qwen.sh > /data/qwen_miner.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo ""
echo -e "${GREEN}✓ Qwen miner started with PID $PID${NC}"
echo ""
echo -e "${BLUE}Features:${NC}"
echo "  • 29 language support (EN/ES/FR/DE/IT/PT/NL/RU/JA/KO/ZH/AR...)"
echo "  • Optimized for business/professional queries"
echo "  • 4-bit quantization for 40+ tokens/second"
echo "  • Superior to Mistral-7B on benchmarks"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo "  View logs:  tail -f /data/qwen_miner.log"
echo "  Check GPU:  nvidia-smi"
echo "  Stop:       kill \$(cat /data/miner.pid)"
echo ""
echo -e "${GREEN}Your miner is now earning with Qwen2.5!${NC}"