#!/bin/bash

# MIA GPU Miner - CUDA Fix
# Works with any CUDA version

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Fixing CUDA issues...${NC}"

# Kill previous attempts
pkill -f miner 2>/dev/null || true

# Check CUDA version
echo -e "${YELLOW}Checking CUDA version...${NC}"
nvidia-smi

# Try to fix CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ldconfig 2>/dev/null || true

# Create simple transformers-based miner (no vLLM)
cd /data/mia-gpu-miner

# Use existing venv or create new
if [ -d "/data/venv-simple" ]; then
    source /data/venv-simple/bin/activate
else
    python3 -m venv /data/venv-simple
    source /data/venv-simple/bin/activate
    pip install --upgrade pip
    
    # Install PyTorch with auto-detected CUDA
    pip install torch torchvision torchaudio
    
    # Install transformers and deps
    pip install transformers accelerate bitsandbytes flask waitress requests
fi

# Create working miner without vLLM
cat > working_miner.py << 'EOF'
#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/huggingface"
import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-working')

app = Flask(__name__)

# Use 8-bit quantization for speed
logger.info("Loading model with 8-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

# Use TheBloke model (ungated)
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("GPTQ model loaded!")
except:
    # Fallback to simpler model
    model_id = "teknium/OpenHermes-2.5-Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )
    logger.info("OpenHermes model loaded!")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        gen_time = time.time() - start
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
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
    backend_url = "https://mia-backend-production.up.railway.app"
    
    try:
        resp = requests.post(f"{backend_url}/register_miner", 
                           json={"name": f"transformers-{os.getpid()}"}, timeout=10)
        miner_id = resp.json()['miner_id'] if resp.status_code == 200 else 1
        logger.info(f"Registered as miner {miner_id}")
    except:
        miner_id = 1
    
    while True:
        try:
            work = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30).json()
            if work and 'request_id' in work:
                logger.info(f"Processing job {work['request_id']}")
                
                inputs = tokenizer(work['prompt'], return_tensors="pt", padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=work.get('max_tokens', 150))
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(work['prompt']):].strip()
                
                requests.post(f"{backend_url}/submit_result", json={
                    'miner_id': int(miner_id),
                    'request_id': work['request_id'],
                    'result': {'response': response}
                })
        except Exception as e:
            logger.error(f"Worker error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting server on port 8000...")
    serve(app, host='0.0.0.0', port=8000)
EOF

chmod +x working_miner.py

# Start it
echo -e "${GREEN}Starting CUDA-compatible miner...${NC}"
nohup python3 working_miner.py > /data/working.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo -e "${GREEN}✓ Miner started with PID $PID${NC}"
echo ""
echo "View logs: tail -f /data/working.log"
echo ""
echo "This miner uses transformers instead of vLLM to avoid CUDA issues"
echo "Expected performance: 15-30 tokens/second with 8-bit quantization"