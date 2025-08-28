#!/bin/bash
# Robust installation and deployment for MIA miner

echo "🚀 MIA Miner Installation & Deployment"
echo "====================================="
echo ""

# First, ensure pip is updated
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "📦 Installing PyTorch with CUDA support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Verify PyTorch installation
echo ""
echo "✓ Verifying PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "❌ PyTorch installation failed!"
    echo "Trying alternative installation..."
    python3 -m pip install torch --no-cache-dir
}

# Install other dependencies
echo ""
echo "📦 Installing vLLM and dependencies..."
python3 -m pip install vllm flask waitress --no-cache-dir

# Create the miner file
echo ""
echo "📝 Creating optimized miner..."
cat > mia_miner_production.py << 'MINER_EOF'
"""
MIA Optimized Miner - 60+ tokens/sec on RTX 3090
"""
import os
import sys

# Verify imports before starting
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"ERROR: {e}")
    print("Please install: pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
from waitress import serve
import logging
import time

# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None

def load_model():
    global model
    logger.info("Loading Qwen2.5-7B optimized for 60+ tokens/sec...")
    
    model = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        max_num_seqs=16,
        trust_remote_code=True,
        enforce_eager=True
    )
    
    # Test speed
    params = SamplingParams(temperature=0, max_tokens=50)
    start = time.time()
    out = model.generate(["Test"], params)
    speed = len(out[0].outputs[0].token_ids) / (time.time() - start)
    logger.info(f"✓ Model ready! Test speed: {speed:.1f} tokens/sec")
    return True

@app.route("/api/generate", methods=["POST"])
def generate():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 500)
    temp = data.get("temperature", 0.7)
    
    params = SamplingParams(
        temperature=temp,
        max_tokens=max_tokens,
        top_p=0.95 if temp > 0 else 1.0,
        repetition_penalty=1.1 if temp > 0 else 1.0
    )
    
    start = time.time()
    outputs = model.generate([prompt], params)
    elapsed = time.time() - start
    
    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / elapsed
    
    logger.info(f"Generated {tokens} tokens @ {speed:.1f} tokens/sec")
    
    return jsonify({
        "text": text,
        "tokens": tokens,
        "tokens_per_second": round(speed, 1)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "loading",
        "backend": "vLLM-Qwen2.5-7B-Optimized"
    })

if __name__ == "__main__":
    if load_model():
        logger.info("Starting server on :8000")
        serve(app, host="0.0.0.0", port=8000, threads=1)
MINER_EOF

# Stop existing
echo ""
echo "🛑 Stopping existing miners..."
pkill -f mia_miner || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

# Start new miner
echo ""
echo "🚀 Starting optimized miner..."
nohup python3 mia_miner_production.py > miner.log 2>&1 &
PID=$!
echo "Started PID: $PID"

# Wait for startup
echo "⏳ Waiting for model to load..."
sleep 15

# Test
echo ""
echo "📊 Testing performance..."
curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_tokens":100,"temperature":0}' | python3 -m json.tool

echo ""
echo "✅ Complete!"
echo "Monitor: tail -f miner.log"
echo "Health: curl http://localhost:8000/health"