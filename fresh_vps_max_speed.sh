#!/bin/bash
# Fresh VPS Maximum Speed Setup - Python 3.10 + vLLM
# Target: 60+ tokens/sec on RTX 3090

echo "🚀 Fresh VPS Maximum Speed Setup"
echo "================================"
echo "This will:"
echo "1. Install Python 3.10"
echo "2. Install CUDA drivers"
echo "3. Install vLLM for 60+ tokens/sec"
echo ""

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.10 and dependencies
echo "🐍 Installing Python 3.10..."
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-pip python3.10-dev python3.10-venv

# Make Python 3.10 the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
update-alternatives --config python3

# Install CUDA toolkit if not present
echo "🎮 Checking CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    apt install -y nvidia-driver-535
fi

# Install build tools
echo "🔧 Installing build tools..."
apt install -y build-essential git curl

# Create working directory
mkdir -p /data
cd /data

# Install PyTorch and vLLM with Python 3.10
echo "📦 Installing PyTorch and vLLM..."
python3.10 -m pip install --upgrade pip
python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3.10 -m pip install vllm flask waitress requests

# Create vLLM miner
echo "📝 Creating vLLM miner..."
cat > /data/vllm_max_speed.py << 'EOF'
import os
import torch
import time
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

print(f"Python: {os.sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('vllm-max-speed')

app = Flask(__name__)

# Load with maximum performance settings
logger.info("Loading Qwen2.5-7B with vLLM for maximum speed...")
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="half",  # FP16 for speed
    gpu_memory_utilization=0.95,  # Use 95% GPU
    max_model_len=8192,  # Larger context
    tensor_parallel_size=1,
    trust_remote_code=True,
    enforce_eager=True,
    disable_log_requests=True
)

logger.info("✓ Model loaded!")

def generate(prompt, max_tokens=150, temperature=0.7):
    if temperature == 0:
        params = SamplingParams(temperature=0, max_tokens=max_tokens, top_p=1.0)
    else:
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.95)
    
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    start = time.time()
    outputs = model.generate([formatted], params)
    elapsed = time.time() - start
    
    text = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / elapsed
    
    logger.info(f"Generated {tokens} tokens @ {speed:.1f} tok/s")
    return text, tokens, elapsed

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
        'response': response,
        'answer': response,
        'tokens_generated': tokens,
        'tokens_per_second': round(tokens/gen_time, 1)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready',
        'backend': 'vLLM',
        'expected_speed': '60-90 tokens/sec'
    })

# MIA backend integration
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    miner_id = f"vllm-speed-{os.getpid()}"
    logger.info(f"Worker started as {miner_id}")
    
    # Register
    try:
        resp = requests.post(f"{backend_url}/register_miner", 
                           json={"name": miner_id, "gpu_model": "RTX 3090", "vram_gb": 24})
        if resp.status_code == 200:
            logger.info(f"✓ Registered with MIA backend")
    except:
        pass
    
    while True:
        try:
            # Get work
            work = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30).json()
            if work and work.get('request_id'):
                response, tokens, gen_time = generate(
                    work.get('prompt', ''),
                    work.get('max_tokens', 150),
                    work.get('temperature', 0.7)
                )
                
                # Submit result
                requests.post(f"{backend_url}/submit_result", json={
                    'miner_id': miner_id,
                    'request_id': work['request_id'],
                    'result': {
                        'response': response,
                        'tokens_generated': tokens,
                        'processing_time': gen_time,
                        'tokens_per_second': round(tokens/gen_time, 1)
                    }
                })
                logger.info(f"✓ Job complete: {tokens} @ {tokens/gen_time:.1f} tok/s")
        except:
            pass
        time.sleep(1)

if __name__ == "__main__":
    # Speed test
    print("\n🏃 Running speed test...")
    
    # Test 1: Greedy
    resp, tokens, elapsed = generate("What is the meaning of life?", 100, 0)
    print(f"Greedy: {tokens} tokens @ {tokens/elapsed:.1f} tok/s")
    
    # Test 2: Sampling
    resp, tokens, elapsed = generate("Explain quantum computing", 100, 0.7)
    print(f"Sampling: {tokens} tokens @ {tokens/elapsed:.1f} tok/s")
    
    print("\n✓ Starting server...")
    threading.Thread(target=worker, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000, threads=1)
EOF

# Start the miner
echo ""
echo "🚀 Starting maximum speed miner..."
cd /data && nohup python3.10 vllm_max_speed.py > vllm.log 2>&1 &
echo $! > vllm.pid

echo ""
echo "✅ Setup complete!"
echo ""
echo "Expected performance on RTX 3090:"
echo "- Greedy decoding: 70-90 tokens/sec"
echo "- With sampling: 60-80 tokens/sec" 
echo ""
echo "Note: First run compiles CUDA kernels (5-10 min)"
echo ""
echo "Commands:"
echo "  Monitor: tail -f /data/vllm.log"
echo "  Test: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Hello\",\"max_tokens\":100,\"temperature\":0}'"
echo "  Stop: kill \$(cat /data/vllm.pid)"