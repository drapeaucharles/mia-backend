#!/bin/bash
# Universal MIA Miner Installer v2 - Works with any GPU/system
# Supports Command-R with tool calling or falls back to Qwen

echo "🚀 MIA Universal Miner Installer v2"
echo "=================================="
echo "Features:"
echo "- Auto-detects GPU and CUDA"
echo "- Installs Command-R-7B with tool calling"
echo "- Falls back to Qwen2.5-7B if needed"
echo "- Works on any VPS/GPU setup"
echo ""

# Detect if running on Vast.ai or regular VPS
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

MINER_DIR="$BASE_DIR/mia-gpu-miner"

# Stop any existing miners
echo "🛑 Stopping existing miners..."
pkill -f "miner.py" || true
pkill -f "command_r" || true
pkill -f "vllm_worker" || true
systemctl stop mia-miner 2>/dev/null || true
sleep 2

# Create directory
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y python3-pip python3-venv git wget curl > /dev/null 2>&1

# Check CUDA
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    echo "✓ CUDA $CUDA_VERSION detected"
else
    echo "⚠️ No GPU detected - CPU mode only"
    CUDA_VERSION="cpu"
fi

# Create Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch based on CUDA version
echo "📦 Installing PyTorch..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ "$CUDA_VERSION" -ge "12" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install transformers accelerate sentencepiece protobuf
pip install flask waitress requests psutil gputil
pip install vllm==0.4.2  # For fast inference
pip install auto-awq  # For AWQ model support

# Create the universal miner with Command-R + fallback
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Universal Miner v2 - Command-R with Tool Calling
Falls back to Qwen if Command-R not available
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
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams
from typing import Dict, List, Optional

# Configure environment
if os.path.exists("/data"):
    os.environ["HF_HOME"] = "/data/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging
log_file = "/data/miner.log" if os.path.exists("/data") else "miner.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('mia-miner-v2')

app = Flask(__name__)

# Global model
model = None
model_name = None

def load_model():
    """Load best available model with tool calling support"""
    global model, model_name
    
    models_to_try = [
        # Command-R variants (with tool calling)
        ("TheBloke/c4ai-command-r-v01-AWQ", {"quantization": "awq"}),
        ("CohereForAI/c4ai-command-r-v01", {}),
        # Fallback to Qwen (tool calling via prompts)
        ("Qwen/Qwen2.5-7B-Instruct", {}),
        ("Qwen/Qwen2.5-7B-Instruct-AWQ", {"quantization": "awq"}),
    ]
    
    for name, kwargs in models_to_try:
        try:
            logger.info(f"Attempting to load {name}...")
            model = LLM(
                model=name,
                dtype="half",
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                trust_remote_code=True,
                enforce_eager=True,
                **kwargs
            )
            model_name = name
            logger.info(f"✓ Successfully loaded {name}")
            return
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            continue
    
    logger.error("Failed to load any model!")
    sys.exit(1)

# Tool calling patterns
TOOL_PATTERNS = {
    "command-r": {
        "call": r'<tool_call>\s*({[^}]+})\s*</tool_call>',
        "template": """<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a helpful assistant with access to tools.

Available tools:
{tools}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call><|END_OF_TURN_TOKEN|>
<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|>
<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"""
    },
    "qwen": {
        "call": r'(?:<tool_call>|<function_call>)\s*({[^}]+})\s*(?:</tool_call>|</function_call>)',
        "template": """You are a helpful assistant with access to functions.

Available functions:
{tools}

When you need information, use functions by responding with:
<function_call>
{{"name": "function_name", "parameters": {{"param": "value"}}}}
</function_call>

User: {prompt}
Assistant:"""
    }
}

def get_model_type():
    """Determine model type for prompting"""
    if "command" in model_name.lower():
        return "command-r"
    else:
        return "qwen"

def format_tools_prompt(prompt: str, tools: List[Dict], context: Dict = None) -> str:
    """Format prompt based on model type"""
    model_type = get_model_type()
    template = TOOL_PATTERNS[model_type]["template"]
    
    # Format tools
    tools_text = json.dumps(tools, indent=2)
    
    # Add context if provided
    if context and context.get("business_name"):
        prompt = f"[Context: You are helping customers at {context['business_name']}]\n{prompt}"
    
    return template.format(tools=tools_text, prompt=prompt)

def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response"""
    model_type = get_model_type()
    pattern = TOOL_PATTERNS[model_type]["call"]
    
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool call: {match.group(1)}")
    return None

def generate(prompt: str, tools: List[Dict] = None, context: Dict = None,
            max_tokens: int = 150, temperature: float = 0.7) -> Dict:
    """Generate response with optional tool calling"""
    
    # Format prompt
    if tools:
        formatted_prompt = format_tools_prompt(prompt, tools, context)
    else:
        # Simple format
        if "command" in model_name.lower():
            formatted_prompt = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        else:
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    start = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    gen_time = time.time() - start
    
    response = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / gen_time if gen_time > 0 else 0
    
    logger.info(f"Generated {tokens} tokens in {gen_time:.2f}s = {speed:.1f} tok/s")
    
    # Check for tool call
    tool_call = extract_tool_call(response) if tools else None
    
    return {
        'response': response,
        'tool_call': tool_call,
        'tokens': tokens,
        'time': gen_time,
        'speed': speed
    }

@app.route('/generate', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for generation"""
    data = request.json
    result = generate(
        prompt=data.get('prompt', ''),
        tools=data.get('tools'),
        context=data.get('context'),
        max_tokens=data.get('max_tokens', 150),
        temperature=data.get('temperature', 0.7)
    )
    
    return jsonify({
        'text': result['response'],
        'tool_call': result.get('tool_call'),
        'tokens_generated': result['tokens'],
        'tokens_per_second': round(result['speed'], 1),
        'model': model_name
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready',
        'model': model_name,
        'type': get_model_type(),
        'features': ['tool_calling', 'multilingual'],
        'version': '2.0'
    })

# MIA Backend Integration
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def register_miner():
    """Register with MIA backend"""
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Register
        response = requests.post(
            f"{backend_url}/register_miner",
            json={
                "miner_id": f"universal-v2-{hostname}",
                "model": model_name,
                "features": ["tool_calling", "multilingual"],
                "version": "2.0"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✓ Registered with MIA backend")
        else:
            logger.warning(f"Registration failed: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to register: {e}")

def worker():
    """Background worker for MIA jobs"""
    miner_id = f"universal-v2-{socket.gethostname()}"
    logger.info(f"Starting worker: {miner_id}")
    
    while True:
        try:
            # Get work
            response = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                work = response.json()
                
                if work and work.get('request_id'):
                    logger.info(f"Processing job: {work['request_id']}")
                    
                    # Generate response
                    result = generate(
                        prompt=work.get('prompt', ''),
                        tools=work.get('tools'),
                        context=work.get('context'),
                        max_tokens=work.get('max_tokens', 150),
                        temperature=work.get('temperature', 0.7)
                    )
                    
                    # Submit result
                    submission = {
                        'response': result['response'],
                        'tokens_generated': result['tokens'],
                        'processing_time': result['time']
                    }
                    
                    if result.get('tool_call'):
                        submission['tool_call'] = result['tool_call']
                        submission['requires_tool_execution'] = True
                    
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': miner_id,
                        'request_id': work['request_id'],
                        'result': submission
                    })
                    
                    logger.info(f"✓ Completed: {work['request_id']}")
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
        
        time.sleep(1)

if __name__ == "__main__":
    # Load model
    load_model()
    
    # Register with backend
    register_miner()
    
    # Test tool calling
    logger.info("Testing tool calling...")
    test_result = generate(
        "What ingredients are in the pasta?",
        tools=[{
            "name": "get_dish_details",
            "description": "Get dish details",
            "parameters": {"dish_name": {"type": "string"}}
        }]
    )
    logger.info(f"Tool call test: {test_result.get('tool_call')}")
    
    # Start worker
    threading.Thread(target=worker, daemon=True).start()
    
    # Start server
    logger.info(f"Starting server with {model_name}...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

# Make executable
chmod +x miner.py

# Create runner script
cat > run_miner.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python miner.py
EOF
chmod +x run_miner.sh

# Create systemd service
cat > mia-miner.service << EOF
[Unit]
Description=MIA GPU Miner v2
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$MINER_DIR
Environment="PATH=$MINER_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$MINER_DIR/venv/bin/python $MINER_DIR/miner.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install service if systemd available
if command -v systemctl &> /dev/null; then
    echo "📦 Installing systemd service..."
    sudo cp mia-miner.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable mia-miner
fi

# Create start/stop scripts
cat > start_miner.sh << 'EOF'
#!/bin/bash
if command -v systemctl &> /dev/null; then
    sudo systemctl start mia-miner
    echo "Miner started via systemd"
    echo "View logs: sudo journalctl -u mia-miner -f"
else
    cd "$(dirname "$0")"
    source venv/bin/activate
    nohup python miner.py > miner.log 2>&1 &
    echo $! > miner.pid
    echo "Miner started. PID: $(cat miner.pid)"
    echo "View logs: tail -f miner.log"
fi
EOF
chmod +x start_miner.sh

cat > stop_miner.sh << 'EOF'
#!/bin/bash
if command -v systemctl &> /dev/null; then
    sudo systemctl stop mia-miner
    echo "Miner stopped"
else
    if [ -f miner.pid ]; then
        kill $(cat miner.pid) 2>/dev/null
        rm miner.pid
        echo "Miner stopped"
    else
        pkill -f "miner.py"
        echo "All miner processes stopped"
    fi
fi
EOF
chmod +x stop_miner.sh

# Create test script
cat > test_miner.sh << 'EOF'
#!/bin/bash
echo "Testing miner API..."
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}' | python3 -m json.tool
EOF
chmod +x test_miner.sh

echo ""
echo "✅ MIA Miner v2 Installation Complete!"
echo ""
echo "🚀 Quick Start:"
echo "   ./start_miner.sh  - Start the miner"
echo "   ./stop_miner.sh   - Stop the miner"
echo "   ./test_miner.sh   - Test the API"
echo ""
echo "📊 Monitor:"
if command -v systemctl &> /dev/null; then
    echo "   sudo journalctl -u mia-miner -f"
else
    echo "   tail -f miner.log"
fi
echo ""
echo "🔧 Features:"
echo "   - Model: $model_name (auto-selected)"
echo "   - Tool calling support"
echo "   - Multilingual (EN/FR/RU/ID)"
echo "   - Auto-restarts on failure"
echo ""

# Offer to start now
read -p "Start miner now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./start_miner.sh
fi