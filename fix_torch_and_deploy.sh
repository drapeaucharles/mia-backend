#!/bin/bash
# Fix missing torch and deploy optimized miner

echo "🔧 Fixing PyTorch installation and deploying optimized miner"
echo "=========================================================="

# Check Python version
echo "Python version:"
python3 --version

# Install PyTorch and required packages
echo ""
echo "📦 Installing PyTorch and dependencies..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install vllm flask waitress

# Verify installation
echo ""
echo "✓ Verifying installations..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}')"
python3 -c "import vllm; print(f'vLLM installed successfully')"

# Now deploy the optimized miner
echo ""
echo "🚀 Deploying optimized miner..."

# Create the optimized miner
cat > mia_miner_production.py << 'EOF'
"""
Optimized MIA Miner for RTX 3090 with Qwen2.5-7B
Target: 60+ tokens/second
"""
import os
import sys

# Check if torch is available
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} loaded")
except ImportError:
    print("ERROR: PyTorch not found. Installing...")
    os.system("pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
    import torch

from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
from waitress import serve
import logging
import time

# Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Environment optimizations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None

def load_optimized_model():
    """Load Qwen2.5-7B with optimal settings for RTX 3090"""
    global model
    
    logger.info("Loading optimized Qwen2.5-7B for RTX 3090...")
    
    try:
        # Optimal configuration for RTX 3090 (24GB VRAM)
        model = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            max_num_seqs=16,
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            disable_log_requests=True
        )
        
        # Warmup
        logger.info("Warming up model...")
        warmup_params = SamplingParams(temperature=0, max_tokens=50)
        start = time.time()
        outputs = model.generate(["Hello, how are you?"], warmup_params)
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
    """Optimized generation endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 500)
    temperature = data.get("temperature", 0.7)
    
    # For maximum speed, use greedy decoding
    if temperature == 0:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            repetition_penalty=1.0
        )
    else:
        # Normal sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            repetition_penalty=1.1
        )
    
    try:
        start = time.time()
        outputs = model.generate([prompt], sampling_params)
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
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "loading"}), 503
    
    # Get GPU stats
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        used_memory = torch.cuda.memory_allocated(0) / (1024**3)
        utilization = (used_memory / gpu_memory) * 100
    else:
        gpu_memory = used_memory = utilization = 0
    
    return jsonify({
        "status": "ready",
        "model": "Qwen2.5-7B-Optimized",
        "backend": "vLLM-AWQ",
        "gpu": {
            "total_gb": round(gpu_memory, 2),
            "used_gb": round(used_memory, 2),
            "utilization_percent": round(utilization, 1)
        }
    })

def main():
    """Start optimized server"""
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logger.error("❌ CUDA not available!")
        
    # Load model
    if not load_optimized_model():
        logger.error("Failed to load model, exiting...")
        return
    
    # Start server with single thread for consistency
    logger.info("Starting optimized inference server on port 8000...")
    serve(app, host="0.0.0.0", port=8000, threads=1, connection_limit=100)

if __name__ == "__main__":
    main()
EOF

# Kill any existing processes
echo ""
echo "🛑 Stopping existing processes..."
pkill -f "mia_miner" || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

# Start the optimized miner
echo ""
echo "🚀 Starting optimized miner..."
nohup python3 mia_miner_production.py > miner.log 2>&1 &
echo "Miner started with PID: $!"

# Wait for startup
sleep 10

# Check if it's running
echo ""
echo "📊 Checking status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Miner is running!"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "❌ Miner failed to start. Check logs:"
    tail -20 miner.log
fi

echo ""
echo "To monitor: tail -f miner.log"
echo "To test speed: curl -X POST http://localhost:8000/api/generate -H 'Content-Type: application/json' -d '{\"prompt\":\"Hello\",\"max_tokens\":100,\"temperature\":0}'"