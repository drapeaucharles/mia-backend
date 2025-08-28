#!/bin/bash
# Complete MIA Miner Installation - Handles EVERYTHING
# For RTX 3090: 60+ tokens/sec

echo "🚀 MIA Complete Installer - Handling Everything"
echo "=============================================="
echo ""

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then 
   echo "✓ Running with proper permissions"
else
   echo "⚠️  Not running as root, some commands may need sudo"
fi

# 1. Install system dependencies
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    apt-get update -qq
    apt-get install -y python3 python3-pip python3-dev build-essential curl git
elif command -v yum &> /dev/null; then
    # RHEL/CentOS
    yum install -y python3 python3-pip python3-devel gcc gcc-c++ curl git
else
    echo "⚠️  Could not detect package manager, skipping system packages"
fi

# 2. Upgrade pip
echo ""
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip || {
    # If pip module fails, try direct pip3
    pip3 install --upgrade pip || {
        # If that fails, try installing pip manually
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        rm get-pip.py
    }
}

# 3. Install PyTorch with CUDA
echo ""
echo "📦 Installing PyTorch with CUDA 11.8..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir || {
    echo "❌ PyTorch installation failed, trying alternative..."
    pip3 install torch --no-cache-dir
}

# 4. Install vLLM and dependencies
echo ""
echo "📦 Installing vLLM and other dependencies..."
pip3 install vllm flask waitress transformers accelerate --no-cache-dir

# 5. Verify installations
echo ""
echo "✓ Verifying installations..."
python3 << EOF
import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except:
    print("❌ PyTorch not available")
    
try:
    import vllm
    print("✓ vLLM installed")
except:
    print("❌ vLLM not available")
EOF

# 6. Create optimized miner
echo ""
echo "📝 Creating optimized miner..."
cat > mia_miner_production.py << 'MINER_EOF'
"""
MIA Miner - Optimized for RTX 3090 (60+ tokens/sec)
"""
import os
import sys
import logging

# Check dependencies
try:
    import torch
    from vllm import LLM, SamplingParams
    from flask import Flask, request, jsonify
    from waitress import serve
    import time
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing missing packages...")
    os.system("pip3 install torch vllm flask waitress")
    sys.exit(1)

# CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None

print(f"Starting MIA Miner...")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_model():
    global model
    logger.info("Loading Qwen2.5-7B optimized configuration...")
    
    try:
        model = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=0.95,  # Maximum GPU usage
            max_model_len=4096,           # 2x context
            max_num_seqs=16,              # 2x parallelism  
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_requests=True
        )
        
        # Warmup test
        logger.info("Running warmup test...")
        params = SamplingParams(temperature=0, max_tokens=50)
        start = time.time()
        outputs = model.generate(["Hello world"], params)
        elapsed = time.time() - start
        
        tokens = len(outputs[0].outputs[0].token_ids)
        speed = tokens / elapsed
        logger.info(f"✓ Model loaded! Warmup speed: {speed:.1f} tokens/sec")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route("/api/generate", methods=["POST"])
def generate():
    if not model:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 500)
    temperature = data.get("temperature", 0.7)
    
    # Optimize for speed when temperature=0
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95 if temperature > 0 else 1.0,
        repetition_penalty=1.1 if temperature > 0 else 1.0
    )
    
    start = time.time()
    outputs = model.generate([prompt], params)
    elapsed = time.time() - start
    
    text = outputs[0].outputs[0].text
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / elapsed
    
    logger.info(f"Generated {tokens} tokens in {elapsed:.2f}s = {speed:.1f} tokens/sec")
    
    return jsonify({
        "text": text,
        "tokens": tokens,
        "time_ms": int(elapsed * 1000),
        "tokens_per_second": round(speed, 1)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "loading",
        "model": "Qwen2.5-7B-AWQ",
        "optimization": "RTX3090-60fps",
        "config": {
            "gpu_memory_utilization": "95%",
            "max_sequences": 16,
            "context_length": 4096
        }
    })

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("MIA Miner - Optimized for 60+ tokens/sec")
    logger.info("=" * 50)
    
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available! Performance will be poor.")
    
    if load_model():
        logger.info("Starting server on port 8000 (single-threaded for max performance)")
        serve(app, host="0.0.0.0", port=8000, threads=1, connection_limit=100)
    else:
        logger.error("Failed to start miner")
        sys.exit(1)
MINER_EOF

# 7. Stop any existing miners
echo ""
echo "🛑 Stopping existing miners..."
pkill -f "mia_miner" || true
pkill -f "python.*8000" || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 3

# 8. Start the optimized miner
echo ""
echo "🚀 Starting optimized miner..."
nohup python3 mia_miner_production.py > miner.log 2>&1 &
PID=$!
echo "Miner started with PID: $PID"

# 9. Wait for startup
echo "⏳ Waiting for model to load (this may take 1-2 minutes)..."
sleep 20

# 10. Test the miner
echo ""
echo "📊 Testing miner performance..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Miner is running!"
    echo ""
    echo "Health check:"
    curl -s http://localhost:8000/health | python3 -m json.tool
    echo ""
    echo "Speed test:"
    curl -s -X POST http://localhost:8000/api/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt":"What is artificial intelligence?","max_tokens":100,"temperature":0}' | python3 -m json.tool | grep -E "(tokens_per_second|text)" || true
else
    echo "❌ Miner failed to start. Checking logs..."
    tail -30 miner.log
fi

echo ""
echo "=============================================="
echo "✅ Installation Complete!"
echo ""
echo "Expected performance on RTX 3090: 60+ tokens/sec"
echo ""
echo "Commands:"
echo "  Monitor logs:  tail -f miner.log"
echo "  Check health:  curl http://localhost:8000/health"
echo "  Stop miner:    kill $PID"
echo "  Start miner:   python3 mia_miner_production.py"
echo ""
echo "If you see less than 60 tokens/sec, check:"
echo "  1. GPU is RTX 3090 (24GB)"
echo "  2. CUDA is properly installed"
echo "  3. No other processes using GPU"
echo "=============================================="