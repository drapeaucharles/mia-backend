#!/bin/bash

# MIA GPU Miner - vLLM AWQ Installation (Proven 60+ tok/s)
# Uses the exact configuration that works

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA GPU Miner - vLLM AWQ Setup         ║${NC}"
echo -e "${GREEN}║      Proven 60+ tokens/second            ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Use /data for Vast.ai
INSTALL_DIR="/data/mia-gpu-miner"
VENV_DIR="/data/venv"

echo -e "${YELLOW}Installing to: $INSTALL_DIR${NC}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 11.8
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install specific vLLM version that works
echo -e "${YELLOW}Installing vLLM 0.2.2 (stable version)...${NC}"
pip install vllm==0.2.2

# Install other dependencies
echo -e "${YELLOW}Installing other dependencies...${NC}"
pip install flask waitress requests aiohttp

# Create the miner script optimized for AWQ
cat > mia_miner_vllm_awq.py << 'EOF'
#!/usr/bin/env python3
import os
import asyncio
import json
import logging
import requests
import time
from datetime import datetime
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Global variables
llm = None
BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")

def initialize_model():
    """Initialize vLLM with AWQ model"""
    global llm
    logger.info("Initializing vLLM with AWQ model...")
    
    # Use the proven AWQ model
    llm = LLM(
        model="TheBloke/Mistral-7B-OpenOrca-AWQ",
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=4096
    )
    
    logger.info("Model initialized successfully!")
    return True

@app.route('/generate', methods=['POST'])
def generate():
    """Local generation endpoint"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.7)
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            top_k=50
        )
        
        # Generate
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # Extract text
        generated_text = outputs[0].outputs[0].text
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return jsonify({
            'text': generated_text,
            'tokens_generated': tokens_generated,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second
        })
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

async def main_worker():
    """Main worker loop to get jobs from MIA backend"""
    logger.info(f"Starting worker, connecting to {BACKEND_URL}")
    
    while True:
        try:
            # Register with backend
            register_response = requests.post(
                f"{BACKEND_URL}/register_miner",
                json={"name": f"vllm-awq-{os.getpid()}"},
                timeout=10
            )
            
            if register_response.status_code == 200:
                miner_data = register_response.json()
                miner_id = miner_data['miner_id']
                logger.info(f"Registered as miner {miner_id}")
                
                # Work loop
                while True:
                    # Get work
                    work_response = requests.get(
                        f"{BACKEND_URL}/get_work?miner_id={miner_id}",
                        timeout=30
                    )
                    
                    if work_response.status_code == 200:
                        work = work_response.json()
                        
                        if work and 'request_id' in work:
                            logger.info(f"Got job {work['request_id']}")
                            
                            # Generate response
                            prompt = work.get('prompt', '')
                            max_tokens = work.get('max_tokens', 150)
                            
                            sampling_params = SamplingParams(
                                temperature=work.get('temperature', 0.7),
                                max_tokens=max_tokens,
                                top_p=0.95
                            )
                            
                            start_time = time.time()
                            outputs = llm.generate([prompt], sampling_params)
                            generation_time = time.time() - start_time
                            
                            generated_text = outputs[0].outputs[0].text
                            tokens_generated = len(outputs[0].outputs[0].token_ids)
                            tokens_per_second = tokens_generated / generation_time
                            
                            logger.info(f"Generated {tokens_generated} tokens at {tokens_per_second:.1f} tok/s")
                            
                            # Submit result
                            result_response = requests.post(
                                f"{BACKEND_URL}/submit_result",
                                json={
                                    'miner_id': int(miner_id),
                                    'request_id': work['request_id'],
                                    'result': {
                                        'response': generated_text,
                                        'tokens_generated': tokens_generated,
                                        'processing_time': generation_time
                                    }
                                },
                                timeout=10
                            )
                            
                            if result_response.status_code == 200:
                                logger.info("Result submitted successfully")
                            else:
                                logger.error(f"Failed to submit result: {result_response.status_code}")
                    
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(5)

def run_flask():
    """Run Flask server"""
    logger.info("Starting Flask server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)

if __name__ == "__main__":
    # Initialize model first
    if initialize_model():
        # Start Flask in a thread
        import threading
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        
        # Run main worker
        asyncio.run(main_worker())
    else:
        logger.error("Failed to initialize model")
EOF

chmod +x mia_miner_vllm_awq.py

# Create run script
cat > run_miner.sh << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source "$VENV_DIR/bin/activate"
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"
export CUDA_VISIBLE_DEVICES=0
python3 mia_miner_vllm_awq.py
EOF
chmod +x run_miner.sh

# Create start script
cat > start_miner.sh << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source "$VENV_DIR/bin/activate"
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"
export CUDA_VISIBLE_DEVICES=0

# Stop existing miner
if [ -f "/data/miner.pid" ]; then
    kill \$(cat /data/miner.pid) 2>/dev/null || true
    sleep 2
fi

# Start new miner
nohup python3 mia_miner_vllm_awq.py > /data/miner.log 2>&1 &
echo \$! > /data/miner.pid
echo "Miner started with PID \$(cat /data/miner.pid)"
echo "Logs: tail -f /data/miner.log"
EOF
chmod +x start_miner.sh

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ vLLM AWQ Miner installed successfully!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}This setup uses:${NC}"
echo "• vLLM 0.2.2 (stable version without puccinialin issue)"
echo "• Mistral-7B-OpenOrca-AWQ (proven 60+ tok/s)"
echo "• Optimized for maximum performance"
echo ""
echo -e "${YELLOW}To start:${NC}"
echo "  cd $INSTALL_DIR"
echo "  ./run_miner.sh"
echo ""
echo -e "${YELLOW}Or in background:${NC}"
echo "  ./start_miner.sh"
echo ""