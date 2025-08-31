#!/bin/bash

# MIA GPU Miner - Final Qwen2.5 Installer
# Uses only the new /submit_result endpoint

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MIA GPU Miner - Qwen2.5 Final Setup     ║${NC}"
echo -e "${BLUE}║   One-Line Install & Start                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system deps if missing
if ! command -v python3 &> /dev/null; then
    apt update && apt install -y python3 python3-pip python3-venv
fi

# Kill any existing miners
pkill -f miner 2>/dev/null || true

# Setup directories
INSTALL_DIR="/data/mia-final"
VENV_DIR="/data/venv-final"

echo -e "${YELLOW}Setting up in: $INSTALL_DIR${NC}"
rm -rf "$INSTALL_DIR" "$VENV_DIR"  # Clean install
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes optimum
pip install flask waitress requests aiohttp
pip install sentencepiece protobuf

# Create the final miner
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Qwen2.5 with new endpoint"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-final')

app = Flask(__name__)
backend_url = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")

# Load Qwen2.5-7B
logger.info("Loading Qwen2.5-7B-Instruct...")
model_id = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logger.info("✓ Model loaded successfully!")

def generate_response(prompt, max_tokens=150, temperature=0.7):
    """Generate response from prompt"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Store input length for proper extraction
    input_length = inputs['input_ids'].shape[1]
    
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
    
    # Extract only the generated tokens
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    tokens_generated = len(generated_tokens)
    tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0
    
    return {
        'text': response,
        'tokens_generated': tokens_generated,
        'tokens_per_second': tokens_per_second,
        'generation_time': gen_time
    }

@app.route('/generate', methods=['POST'])
def api_generate():
    """Local generation endpoint"""
    try:
        data = request.json
        result = generate_response(
            data.get('prompt', ''),
            data.get('max_tokens', 150),
            data.get('temperature', 0.7)
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"API generation error: {e}")
        return jsonify({'error': str(e)}), 500

def worker_loop():
    """Main worker loop using new endpoint format"""
    miner_id = None
    
    # Register miner
    logger.info(f"Connecting to {backend_url}")
    while miner_id is None:
        try:
            resp = requests.post(
                f"{backend_url}/register_miner",
                json={"name": f"qwen2.5-final-{os.getpid()}"},
                timeout=10
            )
            if resp.status_code == 200:
                miner_id = resp.json()['miner_id']
                logger.info(f"✓ Registered as miner {miner_id}")
            else:
                logger.warning(f"Registration failed: {resp.status_code}")
                time.sleep(5)
        except Exception as e:
            logger.warning(f"Cannot reach backend: {e}, retrying...")
            time.sleep(5)
    
    # Work loop
    while True:
        try:
            # Get work using miner_id
            work_resp = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if work_resp.status_code == 200:
                work = work_resp.json()
                
                # Check if we got actual work
                if work and work.get('request_id'):
                    request_id = work['request_id']
                    prompt = work.get('prompt', '')
                    max_tokens = work.get('max_tokens', 150)
                    temperature = work.get('temperature', 0.7)
                    
                    logger.info(f"Processing job {request_id}: {prompt[:50]}...")
                    
                    # Generate response
                    result = generate_response(prompt, max_tokens, temperature)
                    logger.info(f"Generated response: {result['text'][:50]}...")
                    
                    logger.info(f"Generated {result['tokens_generated']} tokens at {result['tokens_per_second']:.1f} tok/s")
                    
                    # Submit result using NEW endpoint format
                    submit_data = {
                        'miner_id': int(miner_id),
                        'request_id': request_id,
                        'result': {
                            'response': result['text'],
                            'tokens_generated': result['tokens_generated'],
                            'processing_time': result['generation_time']
                        }
                    }
                    
                    submit_resp = requests.post(
                        f"{backend_url}/submit_result",
                        json=submit_data,
                        timeout=10
                    )
                    
                    if submit_resp.status_code == 200:
                        logger.info("✓ Result submitted successfully")
                    else:
                        logger.error(f"Submit failed: {submit_resp.status_code} - {submit_resp.text[:200]}")
                        
        except requests.exceptions.Timeout:
            # Normal - no work available
            pass
        except requests.exceptions.ConnectionError:
            logger.warning("Backend connection lost, retrying...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(2)
            
        # Brief pause between checks
        time.sleep(1)

if __name__ == "__main__":
    # Start worker thread
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    
    # Start Flask server
    logger.info("Starting API server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

chmod +x miner.py

# Create launcher script
cat > start.sh << 'EOF'
#!/bin/bash
cd /data/mia-final
source /data/venv-final/bin/activate
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"
export CUDA_VISIBLE_DEVICES=0

while true; do
    echo "Starting Qwen2.5 miner..."
    python3 miner.py
    echo "Miner stopped, restarting in 5 seconds..."
    sleep 5
done
EOF
chmod +x start.sh

# Start the miner
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Starting Qwen2.5 miner...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

nohup ./start.sh > /data/final_miner.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo ""
echo -e "${GREEN}✓ Miner started with PID $PID${NC}"
echo ""
echo -e "${BLUE}Status:${NC}"
echo "  • Model: Qwen2.5-7B-Instruct (29 languages)"
echo "  • Quantization: 4-bit for 40+ tok/s"
echo "  • Endpoint: Using new /submit_result format"
echo "  • Auto-restart: Enabled"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo "  tail -f /data/final_miner.log    # View logs"
echo "  curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Hello!\"}'  # Test locally"
echo ""
echo -e "${GREEN}Your miner is now running and earning!${NC}"