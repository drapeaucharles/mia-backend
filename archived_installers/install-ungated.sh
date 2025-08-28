#!/bin/bash

# MIA GPU Miner - Using Ungated Models
# No HuggingFace login required

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - Ungated Models          ║${NC}"
echo -e "${GREEN}║   No HuggingFace Token Required           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Quick fix - kill previous attempt
pkill -f mia_miner 2>/dev/null || true

# Go to existing installation
cd /data/mia-gpu-miner

# Use existing venv
source /data/venv-proven/bin/activate || source /data/venv/bin/activate || {
    echo "Creating new venv..."
    python3 -m venv /data/venv-ungated
    source /data/venv-ungated/bin/activate
    pip install --upgrade pip
    pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    pip install vllm==0.1.7 flask waitress requests aiohttp
}

# Create miner with ungated model
cat > ungated_miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Using Ungated Models"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-ungated')

app = Flask(__name__)

# Use TheBloke models - they're not gated
logger.info("Loading ungated model...")
try:
    # Try AWQ first for speed
    llm = LLM(
        model="TheBloke/Mistral-7B-OpenOrca-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.95,
        trust_remote_code=True
    )
    logger.info("AWQ model loaded - expecting 60+ tok/s!")
except Exception as e:
    logger.warning(f"AWQ failed: {e}, trying GPTQ...")
    try:
        # Try GPTQ as fallback
        llm = LLM(
            model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            quantization="gptq",
            gpu_memory_utilization=0.95
        )
        logger.info("GPTQ model loaded - expecting 40+ tok/s")
    except Exception as e2:
        logger.warning(f"GPTQ failed: {e2}, using unquantized...")
        # Final fallback - unquantized but ungated
        llm = LLM(
            model="teknium/OpenHermes-2.5-Mistral-7B",
            gpu_memory_utilization=0.95
        )
        logger.info("Unquantized model loaded - expecting 20+ tok/s")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens
        )
        
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        gen_time = time.time() - start
        
        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)
        tps = tokens / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generated {tokens} tokens at {tps:.1f} tok/s")
        
        return jsonify({
            'text': text,
            'tokens_generated': tokens,
            'tokens_per_second': tps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def worker():
    backend_url = "https://mia-backend-production.up.railway.app"
    
    # Register
    try:
        resp = requests.post(f"{backend_url}/register_miner", 
                           json={"name": f"ungated-{os.getpid()}"})
        miner_id = resp.json()['miner_id'] if resp.status_code == 200 else 1
        logger.info(f"Registered as miner {miner_id}")
    except:
        miner_id = 1
    
    while True:
        try:
            work_resp = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30)
            if work_resp.status_code == 200:
                work = work_resp.json()
                if work and 'request_id' in work:
                    logger.info(f"Got job {work['request_id']}")
                    
                    sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=work.get('max_tokens', 150)
                    )
                    outputs = llm.generate([work['prompt']], sampling_params)
                    
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': int(miner_id),
                        'request_id': work['request_id'],
                        'result': {'response': outputs[0].outputs[0].text}
                    })
                    logger.info("Result submitted")
        except Exception as e:
            logger.error(f"Worker error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting Flask server on port 8000...")
    serve(app, host='0.0.0.0', port=8000)
EOF

chmod +x ungated_miner.py

# Start it
echo ""
echo -e "${GREEN}Starting ungated model miner...${NC}"

nohup python3 ungated_miner.py > /data/ungated.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo -e "${GREEN}✓ Miner started with PID $PID${NC}"
echo ""
echo "View logs: tail -f /data/ungated.log"
echo ""
echo "This miner uses ungated models that don't require HuggingFace login:"
echo "- TheBloke/Mistral-7B-OpenOrca-AWQ (60+ tok/s)"
echo "- TheBloke/Mistral-7B-Instruct-v0.2-GPTQ (40+ tok/s)"
echo "- teknium/OpenHermes-2.5-Mistral-7B (20+ tok/s)"