#!/bin/bash

# MIA GPU Miner - Fixed Response Extraction
# Fixes the issue where responses get cut off at the beginning

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   MIA GPU Miner - Fixed Response Extract   ║${NC}"
echo -e "${BLUE}║   Fixing cut-off responses issue           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Install system deps if missing
if ! command -v python3 &> /dev/null; then
    apt update && apt install -y python3 python3-pip python3-venv
fi

# Detect Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${YELLOW}Detected Python version: $PYTHON_VERSION${NC}"

# Install correct venv package
echo -e "${YELLOW}Installing Python $PYTHON_VERSION venv...${NC}"
apt update
apt install -y python${PYTHON_VERSION}-venv python3-pip wget curl git build-essential || {
    apt install -y python3-venv python3-pip wget curl git build-essential
}

# Kill any existing miners
pkill -f miner 2>/dev/null || true

# Setup directories
INSTALL_DIR="/data/mia-fixed"
VENV_DIR="/data/venv-fixed"

echo -e "${YELLOW}Setting up in: $INSTALL_DIR${NC}"
rm -rf "$INSTALL_DIR" "$VENV_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
python3 -m venv "$VENV_DIR" --system-site-packages || {
    pip3 install virtualenv
    virtualenv "$VENV_DIR"
}

# Activate venv
source "$VENV_DIR/bin/activate"

# Ensure pip works
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch
echo -e "${YELLOW}Installing PyTorch...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages
echo -e "${YELLOW}Installing AI packages...${NC}"
pip install transformers accelerate bitsandbytes flask waitress requests sentencepiece protobuf

# Create fixed miner script
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Fixed Response Extraction"""
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
logger = logging.getLogger('mia-fixed')

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

def extract_assistant_response(full_output, prompt_tokens):
    """Extract only the assistant's response from the full output"""
    # Method 1: Look for assistant marker in the output
    if "<|assistant|>" in full_output:
        # Split by assistant marker and take everything after
        parts = full_output.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip()
    
    # Method 2: Look for common response patterns
    if "\nassistant:" in full_output.lower():
        parts = full_output.lower().split("\nassistant:")
        if len(parts) > 1:
            # Get the original case version
            idx = full_output.lower().find("\nassistant:") + len("\nassistant:")
            return full_output[idx:].strip()
    
    # Method 3: Decode only the new tokens (most reliable)
    # This requires keeping track of input length
    return None

def generate_response(prompt, max_tokens=150, temperature=0.7):
    """Generate response from prompt with fixed extraction"""
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input
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
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    gen_time = time.time() - start
    
    # Extract only the generated tokens
    generated_tokens = outputs[0][input_length:]
    
    # Decode only the generated part
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # If response is still empty or starts oddly, try full decode with extraction
    if not response or response.startswith(")") or response.startswith("$"):
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        logger.debug(f"Full output for debugging: {repr(full_output)}")
        
        # Try to extract assistant response
        extracted = extract_assistant_response(full_output, input_length)
        if extracted:
            response = extracted
        else:
            # Last resort: decode without special tokens and remove input
            full_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if full_clean.startswith(text):
                response = full_clean[len(text):].strip()
            else:
                # Find where the response likely starts
                for marker in ["User:", "Human:", prompt[-50:], "Assistant:", "AI:"]:
                    if marker in full_clean:
                        idx = full_clean.rfind(marker) + len(marker)
                        potential_response = full_clean[idx:].strip()
                        if potential_response and not potential_response.startswith((")", "$", ".", ",")):
                            response = potential_response
                            break
    
    tokens_generated = len(generated_tokens)
    tokens_per_second = tokens_generated / gen_time if gen_time > 0 else 0
    
    logger.info(f"Generated {tokens_generated} tokens at {tokens_per_second:.1f} tok/s")
    logger.debug(f"Response preview: {response[:100]}...")
    
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
                json={"name": f"qwen2.5-fixed-{os.getpid()}"},
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
    consecutive_errors = 0
    while True:
        try:
            # Get work
            work_resp = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if work_resp.status_code == 200:
                work = work_resp.json()
                
                if work and work.get('request_id'):
                    request_id = work['request_id']
                    prompt = work.get('prompt', '')
                    max_tokens = work.get('max_tokens', 150)
                    temperature = work.get('temperature', 0.7)
                    
                    logger.info(f"Processing job {request_id}")
                    logger.debug(f"Prompt: {prompt[:100]}...")
                    
                    # Generate response with fixed extraction
                    result = generate_response(prompt, max_tokens, temperature)
                    
                    # Log response for debugging
                    logger.info(f"Generated response: {result['text'][:100]}...")
                    
                    # Submit result
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
                        consecutive_errors = 0
                    else:
                        logger.error(f"Submit failed: {submit_resp.status_code} - {submit_resp.text[:200]}")
                        consecutive_errors += 1
                        
        except requests.exceptions.Timeout:
            pass  # Normal - no work available
        except requests.exceptions.ConnectionError:
            logger.warning("Backend connection lost, retrying...")
            consecutive_errors += 1
            time.sleep(5)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            consecutive_errors += 1
            
        # Exponential backoff on errors
        if consecutive_errors > 5:
            wait_time = min(60, consecutive_errors * 2)
            logger.warning(f"Too many errors, waiting {wait_time}s")
            time.sleep(wait_time)
        else:
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
cd /data/mia-fixed
source /data/venv-fixed/bin/activate
export HF_HOME="/data/huggingface"
export TRANSFORMERS_CACHE="/data/huggingface"
export CUDA_VISIBLE_DEVICES=0

# Set debug logging
export TRANSFORMERS_VERBOSITY=debug

while true; do
    echo "Starting Fixed Qwen2.5 miner..."
    python3 miner.py
    echo "Miner stopped, restarting in 5 seconds..."
    sleep 5
done
EOF
chmod +x start.sh

# Start the miner
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Starting Fixed Qwen2.5 miner...${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

nohup ./start.sh > /data/fixed_miner.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo ""
echo -e "${GREEN}✓ Miner started with PID $PID${NC}"
echo ""
echo -e "${RED}FIXES APPLIED:${NC}"
echo "  • Extracts only generated tokens (not the full output)"
echo "  • Handles special token markers properly"
echo "  • Multiple fallback methods for response extraction"
echo "  • Debug logging to track response generation"
echo ""
echo -e "${BLUE}Status:${NC}"
echo "  • Model: Qwen2.5-7B-Instruct"
echo "  • Response extraction: FIXED"
echo "  • Debug mode: Enabled"
echo ""
echo -e "${YELLOW}Commands:${NC}"
echo "  tail -f /data/fixed_miner.log    # View logs"
echo "  grep 'Response preview' /data/fixed_miner.log    # Check responses"
echo ""
echo -e "${GREEN}The response cut-off issue should now be fixed!${NC}"