#!/bin/bash

# Quick fix for Qwen miner submission

echo "Updating Qwen miner with better error handling..."

# Update the miner in place
cat > /data/mia-qwen/qwen_miner_fixed.py << 'EOF'
#!/usr/bin/env python3
"""MIA GPU Miner - Qwen2.5-7B-Instruct with dual endpoint support"""
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

# Check backend health
backend_url = "https://mia-backend-production.up.railway.app"
try:
    health = requests.get(f"{backend_url}/health", timeout=5)
    if health.status_code == 200:
        logger.info(f"✓ MIA backend is healthy: {health.json()}")
    else:
        logger.warning(f"MIA backend returned {health.status_code}")
except Exception as e:
    logger.warning(f"Cannot reach MIA backend: {e}")

# Load model (reuse existing if already loaded)
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

logger.info("✓ Model loaded and ready!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 150)
        temperature = data.get('temperature', 0.7)
        
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

def submit_result_with_fallback(miner_id, job_id, response_text, session_id=None):
    """Try both endpoints for result submission"""
    
    # First try new /submit_result endpoint
    try:
        resp = requests.post(
            f"{backend_url}/submit_result",
            json={
                'miner_id': int(miner_id),
                'request_id': job_id,
                'result': {'response': response_text}
            },
            timeout=10
        )
        if resp.status_code == 200:
            logger.info("✓ Result submitted via /submit_result")
            return True
        else:
            logger.warning(f"/submit_result returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"/submit_result failed: {e}")
    
    # Try old /job/result endpoint
    if session_id:
        try:
            resp = requests.post(
                f"{backend_url}/job/result",
                json={
                    'job_id': job_id,
                    'session_id': session_id,
                    'output': response_text,
                    'miner_id': str(miner_id)
                },
                timeout=10
            )
            if resp.status_code == 200:
                logger.info("✓ Result submitted via /job/result")
                return True
            else:
                logger.warning(f"/job/result returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"/job/result failed: {e}")
    
    return False

def worker():
    """Background worker for MIA backend"""
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
    
    consecutive_errors = 0
    
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
                    consecutive_errors = 0  # Reset error counter
                    
                    # Generate response
                    prompt = work.get('prompt', '')
                    messages = [{"role": "user", "content": prompt}]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = tokenizer(text, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    start = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=work.get('max_tokens', 150),
                            temperature=work.get('temperature', 0.7),
                            do_sample=True
                        )
                    gen_time = time.time() - start
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if "assistant" in response:
                        response = response.split("assistant")[-1].strip()
                    else:
                        response = response[len(text):].strip()
                    
                    tokens = len(outputs[0]) - len(inputs['input_ids'][0])
                    logger.info(f"Generated {tokens} tokens in {gen_time:.2f}s")
                    
                    # Submit with fallback
                    success = submit_result_with_fallback(
                        miner_id, 
                        work['request_id'], 
                        response,
                        work.get('session_id')
                    )
                    
                    if not success:
                        logger.error("Failed to submit result via both endpoints")
                        consecutive_errors += 1
                        
            elif work_resp.status_code == 502:
                logger.warning("Backend returned 502 - may be restarting")
                consecutive_errors += 1
                time.sleep(10)  # Wait longer if backend is down
                
        except requests.exceptions.Timeout:
            pass  # Normal - no work available
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Cannot connect to backend: {e}")
            consecutive_errors += 1
            time.sleep(10)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            consecutive_errors += 1
            
        # Exponential backoff if errors persist
        if consecutive_errors > 5:
            wait_time = min(60, consecutive_errors * 2)
            logger.warning(f"Too many errors, waiting {wait_time}s")
            time.sleep(wait_time)
        else:
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

# Kill old miner
pkill -f qwen_miner.py 2>/dev/null || true

# Start fixed miner
cd /data/mia-qwen
source /data/venv-qwen/bin/activate

echo "Starting fixed Qwen miner..."
nohup python3 qwen_miner_fixed.py > /data/qwen_fixed.log 2>&1 &
PID=$!
echo $PID > /data/miner.pid

echo "✓ Fixed miner started with PID $PID"
echo ""
echo "This version:"
echo "  • Tries both /submit_result and /job/result endpoints"
echo "  • Better error handling for 502 errors"
echo "  • Exponential backoff when backend is down"
echo ""
echo "View logs: tail -f /data/qwen_fixed.log"