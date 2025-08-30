#!/bin/bash
# Command-R Installer with Proper vLLM Setup

echo "🚀 Command-R-7B Installer (vLLM Fixed)"
echo "===================================="

# Environment setup
[ -d "/data" ] && BASE_DIR="/data" || BASE_DIR="$HOME"
MINER_DIR="$BASE_DIR/command-r-miner"

# Stop existing
pkill -f "miner.py" || true
sleep 2

# Setup directory
mkdir -p "$MINER_DIR" && cd "$MINER_DIR"

# System deps
apt-get update -qq && apt-get install -y python3-pip python3-venv git wget > /dev/null 2>&1

# Fresh venv with proper name
echo "🐍 Creating fresh environment..."
python3 -m venv vllm_env
source vllm_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Clean pip config & force official PyPI
pip config unset global.index-url || true
pip config unset global.extra-index-url || true
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL

# Install PyTorch first
echo "📦 Installing PyTorch..."
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.3.0

# Install vLLM from official PyPI
echo "📦 Installing vLLM..."
pip install --index-url https://pypi.org/simple vllm

# Other dependencies
echo "📦 Installing dependencies..."
pip install transformers sentencepiece flask waitress requests psutil

# Create Command-R miner
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""Command-R Miner with Tool Calling"""
import os
import sys
import json
import time
import logging
import requests
import threading
import re
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('command-r')

app = Flask(__name__)
model = None

def load_model():
    global model
    models = [
        "CohereForAI/c4ai-command-r-v01",
        "Qwen/Qwen2.5-7B-Instruct"  # Fallback
    ]
    
    for model_name in models:
        try:
            logger.info(f"Loading {model_name}...")
            model = LLM(
                model=model_name,
                dtype="half",
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                trust_remote_code=True
            )
            logger.info(f"✓ Loaded {model_name}")
            return model_name
        except Exception as e:
            logger.error(f"Failed: {e}")
    
    logger.error("No model loaded!")
    sys.exit(1)

def format_prompt(prompt, tools=None, context=None):
    if tools:
        tools_text = json.dumps(tools, indent=2)
        return f"""You are a helpful assistant with access to tools.

Available tools:
{tools_text}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>

User: {prompt}
Assistant:"""
    return f"User: {prompt}\nAssistant:"

def extract_tool_call(response):
    match = re.search(r'<tool_call>\s*({[^}]+})\s*</tool_call>', response)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None

def generate(prompt, tools=None, **kwargs):
    formatted = format_prompt(prompt, tools, kwargs.get('context'))
    
    params = SamplingParams(
        temperature=kwargs.get('temperature', 0.7),
        max_tokens=kwargs.get('max_tokens', 150),
        top_p=0.95
    )
    
    start = time.time()
    output = model.generate([formatted], params)[0]
    elapsed = time.time() - start
    
    response = output.outputs[0].text.strip()
    tokens = len(output.outputs[0].token_ids)
    
    return {
        'response': response,
        'tool_call': extract_tool_call(response) if tools else None,
        'tokens': tokens,
        'time': elapsed,
        'speed': tokens / elapsed
    }

@app.route('/generate', methods=['POST'])
def api_generate():
    data = request.json
    result = generate(
        data.get('prompt', ''),
        data.get('tools'),
        **data
    )
    return jsonify({
        'text': result['response'],
        'tool_call': result.get('tool_call'),
        'tokens_generated': result['tokens'],
        'tokens_per_second': round(result['speed'], 1)
    })

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ready', 'model': 'Command-R/Qwen'}

# MIA worker
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    miner_id = f"command-r-{os.uname().nodename}"
    logger.info(f"Worker: {miner_id}")
    
    while True:
        try:
            resp = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30)
            if resp.status_code == 200:
                work = resp.json()
                if work and work.get('request_id'):
                    result = generate(work.get('prompt', ''), work.get('tools'), **work)
                    
                    submission = {
                        'response': result['response'],
                        'tokens_generated': result['tokens'],
                        'processing_time': result['time']
                    }
                    
                    if result.get('tool_call'):
                        submission['tool_call'] = result['tool_call']
                    
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': miner_id,
                        'request_id': work['request_id'],
                        'result': submission
                    })
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(5)
        time.sleep(1)

if __name__ == "__main__":
    model_name = load_model()
    threading.Thread(target=worker, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)
EOF

# Create start/stop scripts
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source vllm_env/bin/activate
nohup python miner.py > miner.log 2>&1 &
echo $! > miner.pid
echo "Started. PID: $(cat miner.pid)"
echo "Logs: tail -f miner.log"
EOF
chmod +x start.sh

cat > stop.sh << 'EOF'
#!/bin/bash
[ -f miner.pid ] && kill $(cat miner.pid) && rm miner.pid || pkill -f miner.py
echo "Stopped"
EOF
chmod +x stop.sh

echo "✅ Installation complete!"
echo "📂 Location: $MINER_DIR"
echo "🚀 Start: ./start.sh"

read -p "Start now? (y/n) " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] && ./start.sh