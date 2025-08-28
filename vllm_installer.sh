#!/bin/bash
# vLLM Installer for 60+ tokens/sec

echo "🚀 Installing vLLM for Maximum Speed"
echo "==================================="
echo "Target: 60+ tokens/sec on RTX 3090"
echo ""

# Install vLLM
echo "📦 Installing vLLM..."
pip3 install vllm==0.4.2 flask waitress requests

# Create vLLM miner
cat > /data/vllm_miner.py << 'EOF'
import os
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('vllm-miner')

app = Flask(__name__)

# Load model with vLLM
logger.info("Loading Qwen2.5-7B with vLLM...")
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="half",  # FP16 for speed
    gpu_memory_utilization=0.95,  # Use 95% of GPU memory
    max_model_len=4096,
    trust_remote_code=True,
    enforce_eager=True,  # Disable CUDA graphs for stability
    disable_log_requests=True
)

logger.info("✓ vLLM model loaded!")

def generate(prompt, max_tokens=150, temperature=0.7):
    # vLLM handles chat templates internally
    messages = [{"role": "user", "content": prompt}]
    
    # Create sampling params
    if temperature == 0:
        # Greedy decoding - fastest
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            top_p=1.0
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
    
    # Format prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    start = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    gen_time = time.time() - start
    
    response = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
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
    return jsonify({
        'status': 'ready', 
        'backend': 'vLLM',
        'model': 'Qwen2.5-7B',
        'expected_speed': '60+ tokens/sec'
    })

backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    miner_id = "vllm-1"
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
                    'miner_id': miner_id,
                    'request_id': work['request_id'],
                    'result': {'response': response, 'tokens_generated': tokens, 'processing_time': gen_time}
                })
                logger.info(f"✓ Job done: {tokens} @ {tokens/gen_time:.1f} tok/s")
        except: pass
        time.sleep(1)

if __name__ == "__main__":
    # Speed test
    logger.info("Running speed test...")
    
    # Test 1: Greedy (fastest)
    test_response, test_tokens, test_time = generate("What is AI?", 100, 0)
    logger.info(f"Greedy test: {test_tokens} @ {test_tokens/test_time:.1f} tok/s")
    
    # Test 2: Sampling
    test_response, test_tokens, test_time = generate("Explain quantum computing", 100, 0.7)
    logger.info(f"Sampling test: {test_tokens} @ {test_tokens/test_time:.1f} tok/s")
    
    threading.Thread(target=worker, daemon=True).start()
    logger.info("Starting vLLM server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=1)
EOF

# Stop old miners
echo "🛑 Stopping old miners..."
pkill -f miner || true
sleep 2

# Start vLLM miner
echo "🚀 Starting vLLM miner..."
cd /data && nohup python3 vllm_miner.py > vllm.log 2>&1 &
echo $! > /data/vllm.pid

echo ""
echo "✅ vLLM miner starting..."
echo ""
echo "Expected performance:"
echo "- Greedy decoding: 70-90 tokens/sec"
echo "- With sampling: 60-80 tokens/sec"
echo ""
echo "Note: First run will download/compile kernels (5-10 min)"
echo ""
echo "Monitor: tail -f /data/vllm.log"
echo "Test: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Hello\",\"max_tokens\":100,\"temperature\":0}'"