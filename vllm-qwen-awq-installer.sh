#!/bin/bash
# vLLM Qwen AWQ Installer - No bitsandbytes needed

echo "🚀 Installing vLLM with Qwen AWQ"
echo "================================"

# Base directory
BASE_DIR="/data"
MINER_DIR="$BASE_DIR/qwen-awq-miner"

# Stop existing processes
pkill -f miner.py || true
pkill -f vllm || true
sleep 2

# Create directory
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3-pip git wget curl

# Create virtual environment
echo "🐍 Creating Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.1
echo "🔥 Installing PyTorch..."
pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Install vLLM
echo "⚡ Installing vLLM..."
pip install vllm==0.2.7

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install transformers accelerate flask waitress requests

# Create the miner using vLLM
echo "✍️ Creating miner.py..."
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Qwen AWQ Miner using vLLM
"""
import os
import sys
import json
import time
import logging
import requests
import socket
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

# Global model
model = None
model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"  # Using the AWQ version

def load_model():
    """Load Qwen AWQ model with vLLM"""
    global model
    logger.info(f"Loading {model_name}...")
    
    try:
        # vLLM will handle AWQ quantization automatically
        model = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.95,
            max_model_len=12000,
            quantization="awq"
        )
        logger.info("✓ Model loaded successfully with vLLM!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Trying fallback model...")
        try:
            # Fallback to smaller model if 32B fails
            model_name_fallback = "Qwen/Qwen2.5-7B-Instruct-AWQ"
            model = LLM(
                model=model_name_fallback,
                trust_remote_code=True,
                dtype="half",
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                quantization="awq"
            )
            logger.info(f"✓ Loaded fallback model: {model_name_fallback}")
            return True
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            return False

def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    """Format prompt with tool definitions"""
    if not tools:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Build tool descriptions
    tool_descriptions = []
    for tool in tools:
        params_str = json.dumps(tool.get("parameters", {}), indent=2)
        tool_descriptions.append(f"""Function: {tool['name']}
Description: {tool.get('description', '')}
Parameters: {params_str}""")
    
    tools_text = "\n\n".join(tool_descriptions)
    
    # System prompt with tools
    system_prompt = f"""You are a helpful assistant with access to these functions:

{tools_text}

To use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>"""
    
    if context and context.get("business_name"):
        system_prompt += f"\n\nYou are helping customers at {context['business_name']}."
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response"""
    import re
    match = re.search(r'<tool_call>\s*({[^}]+})\s*</tool_call>', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None

# Flask API
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate endpoint"""
    data = request.json
    prompt = data.get('prompt', '')
    tools = data.get('tools')
    context = data.get('context')
    max_tokens = data.get('max_tokens', 150)
    temperature = data.get('temperature', 0.7)
    
    # Format prompt
    formatted_prompt = format_prompt_with_tools(prompt, tools, context)
    
    # Generate with vLLM
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    start_time = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    gen_time = time.time() - start_time
    
    response = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    
    # Check for tool call
    tool_call = extract_tool_call(response) if tools else None
    
    return jsonify({
        'text': response,
        'tool_call': tool_call,
        'tokens_generated': tokens,
        'tokens_per_second': round(tokens / gen_time, 1) if gen_time > 0 else 0
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'model': model_name if model else 'not loaded'})

# MIA Backend integration
def register_with_backend():
    """Register with MIA backend"""
    backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
    hostname = socket.gethostname()
    
    try:
        response = requests.post(f"{backend_url}/register_miner", json={
            "miner_id": f"vllm-qwen-awq-{hostname}",
            "model": model_name,
            "endpoint": "http://localhost:8000"
        }, timeout=10)
        
        if response.status_code == 200:
            logger.info("✓ Registered with MIA backend")
            return True
    except Exception as e:
        logger.error(f"Registration failed: {e}")
    return False

if __name__ == "__main__":
    # Load model
    if not load_model():
        logger.error("Failed to load model, exiting")
        sys.exit(1)
    
    # Register with backend
    register_with_backend()
    
    # Start API server
    logger.info("Starting API server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

# Create startup script
echo "📝 Creating startup script..."
cat > start_miner.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/huggingface
export TRANSFORMERS_CACHE=/data/huggingface
python miner.py 2>&1 | tee miner.log
EOF
chmod +x start_miner.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "This uses vLLM with Qwen AWQ models (no bitsandbytes needed)"
echo ""
echo "To start the miner:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_miner.sh"