#!/bin/bash
# Simple Maximum Speed Setup - Works with any Python 3.8+

echo "🚀 Simple Maximum Speed Setup"
echo "============================="

# Check Python version
PYTHON_CMD="python3"
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.9 &> /dev/null; then
    PYTHON_CMD="python3.9"
fi

echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Update pip and install packages
echo "📦 Installing packages..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Try vLLM, fallback to optimized transformers if it fails
echo "📦 Attempting vLLM install..."
if $PYTHON_CMD -m pip install vllm==0.3.0; then
    echo "✓ vLLM installed successfully"
    USE_VLLM=true
else
    echo "⚠️  vLLM failed, using optimized transformers"
    $PYTHON_CMD -m pip install transformers accelerate flask waitress requests
    USE_VLLM=false
fi

# Create the appropriate miner
mkdir -p /data
cd /data

if [ "$USE_VLLM" = true ]; then
    # Create vLLM miner
    cat > /data/miner.py << 'EOF'
# vLLM Version - 60+ tokens/sec
import os
print("Using vLLM backend")
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify
from waitress import serve
import time

app = Flask(__name__)

model = LLM("Qwen/Qwen2.5-7B-Instruct", dtype="half", gpu_memory_utilization=0.95)

def generate(prompt, max_tokens=150, temperature=0.7):
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = model.generate([f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"], params)
    return outputs[0].outputs[0].text.strip()

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    start = time.time()
    text = generate(data.get('prompt', ''), data.get('max_tokens', 150))
    elapsed = time.time() - start
    return jsonify({'text': text, 'time': elapsed, 'tokens_per_second': len(text.split())/(elapsed+0.001)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'backend': 'vLLM'})

if __name__ == "__main__":
    print("Testing speed...")
    test = generate("Hello world", 50, 0)
    print(f"Test output: {test}")
    serve(app, host='0.0.0.0', port=8000, threads=1)
EOF
else
    # Create optimized transformers miner
    cat > /data/miner.py << 'EOF'
# Optimized Transformers - 35-45 tokens/sec
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

print("Using optimized transformers")
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
from waitress import serve
import time

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Try to compile
try:
    model = torch.compile(model)
    print("Model compiled for extra speed")
except:
    print("Model compilation not available")

def generate(prompt, max_tokens=150, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=temperature>0, temperature=temperature)
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    start = time.time()
    text = generate(data.get('prompt', ''), data.get('max_tokens', 150))
    elapsed = time.time() - start
    tokens = len(tokenizer.encode(text))
    return jsonify({'text': text, 'tokens': tokens, 'tokens_per_second': tokens/(elapsed+0.001)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'backend': 'transformers-optimized'})

if __name__ == "__main__":
    print("Testing speed...")
    start = time.time()
    test = generate("Hello world", 50, 0)
    elapsed = time.time() - start
    print(f"Test: {len(tokenizer.encode(test))} tokens in {elapsed:.2f}s")
    serve(app, host='0.0.0.0', port=8000, threads=1)
EOF
fi

# Kill old processes
pkill -f miner.py || true

# Start the miner
echo "🚀 Starting miner..."
nohup $PYTHON_CMD /data/miner.py > /data/miner.log 2>&1 &
echo $! > /data/miner.pid

echo ""
echo "✅ Miner started!"
echo ""
echo "Check status: tail -f /data/miner.log"
echo "Test: curl http://localhost:8000/health"
echo ""

# Wait and show initial logs
sleep 10
echo "Initial logs:"
tail -20 /data/miner.log