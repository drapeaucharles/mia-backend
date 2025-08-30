#!/bin/bash
# Command-R-7B Only Installer - Lightweight, No Fallbacks
# For dedicated Command-R deployment

echo "🚀 Command-R-7B Dedicated Installer"
echo "==================================="
echo "Lightweight installation for Command-R only"
echo "No fallbacks, minimal dependencies"
echo ""

# Detect environment
if [ -d "/data" ]; then
    echo "✓ Detected Vast.ai environment"
    BASE_DIR="/data"
    export HF_HOME="/data/huggingface"
    export TRANSFORMERS_CACHE="/data/huggingface"
else
    echo "✓ Detected regular VPS"
    BASE_DIR="$HOME"
    export HF_HOME="$HOME/.cache/huggingface"
fi

MINER_DIR="$BASE_DIR/command-r-miner"

# Stop existing miners
echo "🛑 Stopping existing miners..."
pkill -f "miner.py" || true
pkill -f "command_r" || true
systemctl stop mia-miner 2>/dev/null || true
sleep 2

# Create directory
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Check GPU
echo "🔍 Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ No GPU detected! Command-R requires GPU"
    exit 1
fi
nvidia-smi

# Install minimal system deps
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y python3-pip python3-venv git wget > /dev/null 2>&1

# Create venv
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install only what we need
echo "📦 Installing minimal dependencies..."
pip install --upgrade pip wheel setuptools

# Install PyTorch for CUDA 11.8 (most compatible)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# Install vLLM and minimal deps
pip install vllm==0.4.2
pip install transformers sentencepiece  # Minimal for Command-R
pip install flask waitress requests psutil

# Try AWQ support
pip install autoawq || echo "⚠️ AWQ not available"

# Create Command-R only miner
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""
Command-R-7B Miner - Tool Calling for MIA
No fallbacks, Command-R only
"""
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
from typing import Dict, List, Optional

# Environment
if os.path.exists("/data"):
    os.environ["HF_HOME"] = "/data/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('command-r-miner')

app = Flask(__name__)
model = None

def load_model():
    """Load Command-R model only"""
    global model
    
    # Try AWQ first, then full model
    models = [
        ("TheBloke/c4ai-command-r-v01-AWQ", {"quantization": "awq"}),
        ("CohereForAI/c4ai-command-r-v01", {})
    ]
    
    for model_name, kwargs in models:
        try:
            logger.info(f"Loading {model_name}...")
            model = LLM(
                model=model_name,
                dtype="half",
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                trust_remote_code=True,
                **kwargs
            )
            logger.info(f"✓ Loaded {model_name}")
            return model_name
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
    
    logger.error("❌ Could not load Command-R! This miner requires Command-R.")
    sys.exit(1)

# Command-R specific formatting
def format_with_tools(prompt: str, tools: List[Dict], context: Dict = None) -> str:
    """Format for Command-R with tools"""
    tools_text = json.dumps(tools, indent=2)
    
    ctx = ""
    if context and context.get("business_name"):
        ctx = f"\nContext: You are helping customers at {context['business_name']}.\n"
    
    return f"""<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant.{ctx}

Available tools:
{tools_text}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call><|END_OF_TURN_TOKEN|>
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|>
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""

def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response"""
    match = re.search(r'<tool_call>\s*({[^}]+})\s*</tool_call>', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None

def generate(prompt: str, tools: List[Dict] = None, **kwargs) -> Dict:
    """Generate with optional tools"""
    if tools:
        formatted = format_with_tools(prompt, tools, kwargs.get('context'))
    else:
        formatted = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    
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
    return jsonify({
        'status': 'ready',
        'model': 'Command-R-7B',
        'features': ['tool_calling']
    })

# MIA worker
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    miner_id = f"command-r-{os.uname().nodename}"
    logger.info(f"Worker started: {miner_id}")
    
    while True:
        try:
            resp = requests.get(f"{backend_url}/get_work?miner_id={miner_id}", timeout=30)
            if resp.status_code == 200:
                work = resp.json()
                if work and work.get('request_id'):
                    result = generate(
                        work.get('prompt', ''),
                        work.get('tools'),
                        **work
                    )
                    
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
            logger.error(f"Worker error: {e}")
            time.sleep(5)
        time.sleep(1)

if __name__ == "__main__":
    model_name = load_model()
    logger.info(f"Starting Command-R miner with {model_name}")
    threading.Thread(target=worker, daemon=True).start()
    serve(app, host='0.0.0.0', port=8000)
EOF

chmod +x miner.py

# Create start script
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
nohup python miner.py > miner.log 2>&1 &
echo $! > miner.pid
echo "Command-R miner started. PID: $(cat miner.pid)"
echo "Logs: tail -f miner.log"
EOF
chmod +x start.sh

# Create stop script  
cat > stop.sh << 'EOF'
#!/bin/bash
[ -f miner.pid ] && kill $(cat miner.pid) && rm miner.pid || pkill -f miner.py
echo "Miner stopped"
EOF
chmod +x stop.sh

echo ""
echo "✅ Command-R Installation Complete!"
echo ""
echo "📦 Installed:"
echo "   - PyTorch 2.3.0 (CUDA 11.8)"
echo "   - vLLM 0.4.2"
echo "   - Minimal dependencies (no OpenAI, no extras)"
echo ""
echo "🚀 Usage:"
echo "   ./start.sh - Start miner"
echo "   ./stop.sh  - Stop miner"
echo "   tail -f miner.log - View logs"
echo ""
echo "⚠️  Note: This installer is Command-R only!"
echo "   No fallbacks. If Command-R can't load, miner will exit."
echo ""

read -p "Start Command-R miner now? (y/n) " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] && ./start.sh