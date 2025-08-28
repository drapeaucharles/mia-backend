#!/usr/bin/env python3
# Quick fix for speed issues - apply to running miner

import os

# Create fixed miner configuration
fixed_miner = """
import os
os.environ['HF_HOME'] = '/data/hf'

import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('mia-miner')

app = Flask(__name__)

logger.info("Loading Qwen2.5-7B...")
model_id = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

tokenizer.pad_token = tokenizer.eos_token
logger.info("✓ Model loaded!")

def generate(prompt, max_tokens=150, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    input_length = len(inputs['input_ids'][0])
    
    # FIXED: Proper generation config
    if temperature == 0:
        # Greedy decoding - FASTEST
        generation_args = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id
        }
    else:
        # Sampling
        generation_args = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id
        }
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_args)
    gen_time = time.time() - start
    
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    tokens = len(generated_ids)
    speed = tokens / gen_time
    logger.info(f"Generated {tokens} tokens in {gen_time:.2f}s = {speed:.1f} tok/s")
    
    return response, tokens, gen_time

@app.route('/generate', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    response, tokens, gen_time = generate(
        data.get('prompt', ''), 
        data.get('max_tokens', 150),
        data.get('temperature', 0.7)
    )
    return jsonify({
        'text': response, 
        'tokens_generated': tokens, 
        'tokens_per_second': round(tokens/gen_time, 1)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'model': 'Qwen2.5-7B-Fixed'})

backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    miner_id = 1
    while True:
        try:
            work = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30).json()
            if work and work.get('request_id'):
                response, tokens, gen_time = generate(
                    work.get('prompt', ''), 
                    work.get('max_tokens', 150),
                    work.get('temperature', 0.7)
                )
                
                requests.post(f"{backend_url}/submit_result", json={
                    'miner_id': int(miner_id),
                    'request_id': work['request_id'],
                    'result': {'response': response, 'tokens_generated': tokens, 'processing_time': gen_time}
                })
                logger.info(f"✓ Job completed: {tokens} @ {tokens/gen_time:.1f} tok/s")
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    # Test with greedy decoding
    logger.info("Testing greedy decoding speed (fastest)...")
    test_response, test_tokens, test_time = generate("Hello, how are you?", 50, 0)
    logger.info(f"Greedy speed: {test_tokens} @ {test_tokens/test_time:.1f} tok/s")
    
    # Test with sampling
    logger.info("Testing sampling speed...")
    test_response, test_tokens, test_time = generate("Hello, how are you?", 50, 0.7)
    logger.info(f"Sampling speed: {test_tokens} @ {test_tokens/test_time:.1f} tok/s")
    
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting server on port 8000 (single thread)...")
    serve(app, host='0.0.0.0', port=8000, threads=1)
"""

# Write fixed miner
with open('/data/miner_fixed.py', 'w') as f:
    f.write(fixed_miner)

print("Fixed miner created at /data/miner_fixed.py")
print("\nTo use it:")
print("1. Stop current miner: pkill -f miner.py")
print("2. Start fixed miner: cd /data && python3 miner_fixed.py")
print("\nOr run these commands:")
print("pkill -f miner.py && cd /data && nohup python3 miner_fixed.py > miner.log 2>&1 &")