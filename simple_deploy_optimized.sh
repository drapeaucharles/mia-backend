#!/bin/bash
# Simple deployment for optimized MIA miner - NO VENV
# For decentralized GPU network participants

echo "🚀 MIA Optimized Miner - Simple Deployment"
echo "=========================================="
echo "RTX 3090 Target: 60+ tokens/sec"
echo ""

# Install required packages directly
echo "📦 Installing required packages..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install vllm flask waitress

# Create optimized miner
echo ""
echo "📝 Creating optimized miner..."
cat > mia_miner_production.py << 'EOF'
"""
Optimized MIA Miner for RTX 3090
Decentralized GPU Network - 60+ tokens/sec
"""
import os
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
    
    logger.info("Loading Qwen2.5-7B Optimized for 60+ tokens/sec...")
    
    try:
        model = LLM(
            model="Qwen/Qwen2.5-7B-Instruct",
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=0.95,  # Use 95% GPU memory
            max_model_len=4096,           # 2x larger context
            max_num_seqs=16,              # 2x more parallelism
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=1,
            disable_log_requests=True
        )
        
        # Warmup
        logger.info("Warming up...")
        warmup_params = SamplingParams(temperature=0, max_tokens=50)
        start = time.time()
        outputs = model.generate(["Hello"], warmup_params)
        elapsed = time.time() - start
        
        tokens = len(outputs[0].outputs[0].token_ids)
        speed = tokens / elapsed
        logger.info(f"✓ Ready! Warmup: {speed:.1f} tokens/sec")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate endpoint"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.json
    prompt = data.get("prompt", "")
    max_tokens = min(data.get("max_tokens", 200), 500)
    temperature = data.get("temperature", 0.7)
    
    # Greedy decoding for max speed when temp=0
    if temperature == 0:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens,
            repetition_penalty=1.0
        )
    else:
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
        
        logger.info(f"Generated {tokens} tokens @ {speed:.1f} tokens/sec")
        
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
    """Health check"""
    if model is None:
        return jsonify({"status": "loading"}), 503
    
    return jsonify({
        "status": "ready",
        "model": "Qwen2.5-7B-Optimized",
        "config": {
            "gpu_memory": "95%",
            "max_seqs": 16,
            "max_len": 4096
        }
    })

def main():
    """Start server"""
    if not load_optimized_model():
        logger.error("Model load failed")
        return
    
    logger.info("Starting server on :8000 (single thread for max speed)")
    serve(app, host="0.0.0.0", port=8000, threads=1)

if __name__ == "__main__":
    main()
EOF

# Stop any existing miners
echo ""
echo "🛑 Stopping existing miners..."
pkill -f "mia_miner" || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

# Start optimized miner
echo ""
echo "🚀 Starting optimized miner..."
nohup python3 mia_miner_production.py > miner.log 2>&1 &
PID=$!
echo "Started with PID: $PID"

# Wait and test
sleep 10
echo ""
echo "📊 Testing..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Miner running!"
    echo ""
    curl -s -X POST http://localhost:8000/api/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Test","max_tokens":50,"temperature":0}' | python3 -m json.tool | grep tokens_per_second
else
    echo "❌ Failed to start. Logs:"
    tail -20 miner.log
fi

echo ""
echo "Monitor: tail -f miner.log"
echo "Stop: kill $PID"
EOF

chmod +x simple_deploy_optimized.sh