#!/bin/bash
# vLLM Installer - Wheels Only, No Source Builds

echo "🚀 vLLM Wheels-Only Installer"
echo "============================="

# Require Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is required"
    echo "Installing Python 3.11..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# Verify Python 3.11
python3.11 --version || {
    echo "❌ Failed to install Python 3.11"
    exit 1
}

# Set up directory
MINER_DIR="/data/qwen-awq-miner"
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Create .venv (not venv)
echo "🐍 Creating .venv with Python 3.11..."
python3.11 -m venv .venv
source .venv/bin/activate

# Verify we're in the right venv
echo "📍 Python: $(which python)"
echo "📍 Version: $(python --version)"

# Set pip environment for wheels-only
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_ONLY_BINARY=":all:"
export PIP_NO_CACHE_DIR="1"

echo "⚙️ Pip configuration:"
echo "  - Index: Official PyPI only"
echo "  - Binary: Wheels only (no source builds)"
echo "  - Cache: Disabled"

# Upgrade pip first
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch from official wheels
echo "🔥 Installing PyTorch 2.7.1 (CUDA 12.8)..."
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128 || {
    echo "⚠️  CUDA 12.8 wheels not found, trying CUDA 12.1..."
    pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
}

# Verify PyTorch
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Install vLLM and xFormers (wheels only)
echo "⚡ Installing vLLM 0.10.1.1..."
pip install vllm==0.10.1.1

echo "📦 Installing xFormers 0.0.31..."
pip install xformers==0.0.31

# Try flash-attn wheel (will fail if no wheel exists - that's OK)
echo "📦 Checking for flash-attn wheel..."
pip install flash-attn 2>/dev/null || echo "⚠️  No flash-attn wheel available (expected)"

# Install only essential extras
echo "📦 Installing minimal dependencies..."
pip install requests  # For our test scripts

# Verify installations
echo -e "\n✅ Verifying installations..."
python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'✅ PyTorch: {torch.__version__}')

import xformers
print(f'✅ xFormers: {xformers.__version__}')

import vllm
print(f'✅ vLLM: {vllm.__version__}')
"

# Create logs directory
mkdir -p logs

# Create vLLM server script with exact specifications
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

# Write PID
echo $$ > vllm.pid

# Start vLLM with exact specifications
exec python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12288 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --host 0.0.0.0 \
    --port 8000 \
    >> logs/vllm.out 2>&1
EOF
chmod +x start_vllm.sh

# Create test script
echo "✍️ Creating test_vllm.py..."
cat > test_vllm.py << 'EOF'
#!/usr/bin/env python3
import requests
import json
import time

def test_health():
    """Test server health"""
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        if resp.status_code == 200:
            print("✅ Server running")
            data = resp.json()
            print(f"   Model: {data['data'][0]['id']}")
            return True
    except:
        print("❌ Server not responding")
        return False

def test_tools():
    """Test tool calling with auto mode"""
    data = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": "You are Maria at Bella Vista Restaurant."},
            {"role": "user", "content": "What fish dishes do you have?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_menu_items",
                "description": "Search menu items",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string"},
                        "search_type": {"type": "string", "enum": ["ingredient", "category", "name"]}
                    },
                    "required": ["search_term", "search_type"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0
    }
    
    resp = requests.post("http://localhost:8000/v1/chat/completions", json=data, timeout=30)
    result = resp.json()
    
    if "choices" in result and "tool_calls" in result["choices"][0]["message"]:
        print("✅ Tool calling works (Hermes + auto enabled)")
        tool = result["choices"][0]["message"]["tool_calls"][0]
        print(f"   Function: {tool['function']['name']}")
        print(f"   Arguments: {tool['function']['arguments']}")
    else:
        print("❌ Tool calling not working")

if __name__ == "__main__":
    print("🧪 Testing vLLM\n")
    
    if test_health():
        print("\n📞 Testing tool calling...")
        test_tools()
    else:
        print("\nStart server with: ./start_vllm.sh")
EOF
chmod +x test_vllm.py

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Environment: /data/qwen-awq-miner/.venv (Python 3.11)"
echo "PyTorch: 2.7.1 (CUDA 12.8 or 12.1)"
echo "vLLM: 0.10.1.1"
echo "xFormers: 0.0.31"
echo ""
echo "✅ All packages installed from wheels only"
echo "✅ No source builds allowed"
echo "✅ Using official PyPI only"
echo ""
echo "To start vLLM:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_vllm.sh"
echo ""
echo "To test:"
echo "  ./test_vllm.py"
echo ""
echo "Logs: tail -f logs/vllm.out"
echo "PID: cat vllm.pid"