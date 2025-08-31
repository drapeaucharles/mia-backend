#!/bin/bash
# Simple Qwen AWQ Miner Installer - No fallbacks

echo "🚀 Installing Qwen AWQ Miner"
echo "=========================="

# Base directory
BASE_DIR="/data"
MINER_DIR="$BASE_DIR/qwen-awq-miner"

# Stop existing processes
pkill -f miner.py || true
sleep 2

# Create directory
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Install dependencies
echo "📦 Installing dependencies..."
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3-pip git wget curl > /dev/null 2>&1

# Create virtual environment
echo "🐍 Creating Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and wheel
pip install -U pip wheel setuptools

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch==2.1.0
transformers==4.35.0
accelerate==0.24.1
sentencepiece==0.1.99
protobuf==3.20.3
vllm==0.2.7
flask==3.0.0
waitress==2.1.2
requests==2.31.0
autoawq==0.1.7
EOF

# Install Python packages
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Create the miner.py
echo "✍️ Creating miner.py..."
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Qwen AWQ Miner - Simple and Direct
"""
import os
import sys
import json
import time
import logging
import requests
import threading
import re
import socket
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model
model = None
model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"

def load_model():
    """Load Qwen AWQ model with optimized settings"""
    global model
    logger.info(f"Loading {model_name} with xFormers backend...")
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available!")
            sys.exit(1)
            
        # Load with optimal settings
        model = LLM(
            model=model_name,
            dtype="half",
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            backend="xformers",
            quantization="awq"
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

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
    
    # Generate
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
        'tokens_per_second': round(tokens / gen_time, 1) if gen_time > 0 else 0,
        'model': model_name
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ready', 'model': model_name})

# MIA Backend Worker
def register_with_backend():
    """Register with MIA backend"""
    backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
    hostname = socket.gethostname()
    
    try:
        response = requests.post(f"{backend_url}/register_miner", json={
            "miner_id": f"qwen-awq-{hostname}",
            "model": model_name,
            "endpoint": "http://localhost:8000"
        }, timeout=10)
        
        if response.status_code == 200:
            logger.info("✓ Registered with MIA backend")
            return True
    except Exception as e:
        logger.error(f"Registration failed: {e}")
    return False

def worker_thread():
    """Background worker for MIA jobs"""
    backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
    miner_id = f"qwen-awq-{socket.gethostname()}"
    
    # Register first
    register_with_backend()
    
    while True:
        try:
            # Get work
            response = requests.get(
                f"{backend_url}/get_work",
                params={"miner_id": miner_id},
                timeout=30
            )
            
            if response.status_code == 200:
                work = response.json()
                
                if work and work.get('request_id'):
                    logger.info(f"Processing job: {work['request_id']}")
                    
                    # Format and generate
                    formatted_prompt = format_prompt_with_tools(
                        work.get('prompt', ''),
                        work.get('tools'),
                        work.get('context')
                    )
                    
                    sampling_params = SamplingParams(
                        temperature=work.get('temperature', 0.7),
                        max_tokens=work.get('max_tokens', 150),
                        top_p=0.95
                    )
                    
                    start_time = time.time()
                    outputs = model.generate([formatted_prompt], sampling_params)
                    gen_time = time.time() - start_time
                    
                    response = outputs[0].outputs[0].text.strip()
                    tool_call = extract_tool_call(response) if work.get('tools') else None
                    
                    # Submit result
                    result = {
                        'response': response,
                        'tokens_generated': len(outputs[0].outputs[0].token_ids),
                        'processing_time': gen_time,
                        'tool_call': tool_call
                    }
                    
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': miner_id,
                        'request_id': work['request_id'],
                        'result': result
                    })
                    
                    logger.info(f"✓ Completed job: {work['request_id']}")
                    
        except Exception as e:
            logger.error(f"Worker error: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    # Load model
    load_model()
    
    # Start worker thread
    worker = threading.Thread(target=worker_thread, daemon=True)
    worker.start()
    
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
python miner.py 2>&1 | tee miner.log
EOF
chmod +x start_miner.sh

# Create run script for nohup
cat > run_miner.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
nohup python miner.py > miner.log 2>&1 &
echo "Miner started with PID: $!"
EOF
chmod +x run_miner.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "To start the miner:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_miner.sh"
echo ""
echo "Or run in background:"
echo "  ./run_miner.sh"
echo ""
echo "Monitor logs:"
echo "  tail -f miner.log"