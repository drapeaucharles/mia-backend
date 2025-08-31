#!/bin/bash
# vLLM Installer - Exact Specifications
# NO process killing, NO venv changes, exact versions only

echo "🚀 Installing vLLM with Exact Specifications"
echo "==========================================="

# Verify we're in the right place
if [ ! -d "/data/qwen-awq-miner/.venv" ]; then
    echo "❌ ERROR: Expected .venv at /data/qwen-awq-miner/.venv"
    echo "This installer uses your existing environment"
    exit 1
fi

cd /data/qwen-awq-miner

# Activate existing venv
echo "✅ Using existing .venv"
source .venv/bin/activate

# Verify Python 3.11
python_version=$(python --version | cut -d' ' -f2 | cut -d'.' -f1-2)
if [ "$python_version" != "3.11" ]; then
    echo "❌ ERROR: Expected Python 3.11, got $python_version"
    exit 1
fi

echo "✅ Python 3.11 confirmed"

# Install exact PyTorch versions
echo "📦 Installing PyTorch 2.7.1 with CUDA 12.8..."
pip install --only-binary :all: torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 -f https://download.pytorch.org/whl/torch_stable.html || {
    echo "⚠️  CUDA 12.8 wheels not available, falling back to CUDA 12.1..."
    pip install --only-binary :all: torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
}

# Install vLLM and dependencies (wheels only from PyPI)
echo "📦 Installing vLLM 0.10.1.1 and dependencies..."
pip install --only-binary :all: vllm==0.10.1.1
pip install --only-binary :all: xformers==0.0.31
pip install --only-binary :all: transformers accelerate

# Try flash-attn only if wheel exists
echo "📦 Checking for flash-attn wheel..."
pip install --only-binary :all: flash-attn 2>/dev/null || echo "⚠️  No flash-attn wheel available, skipping"

# Create logs directory
mkdir -p /data/qwen-awq-miner/logs

# Create vLLM startup script
echo "✍️ Creating vllm_server.py..."
cat > vllm_server.py << 'EOF'
#!/usr/bin/env python3
"""vLLM Server with Exact Specifications"""
import os
import sys

# Set environment before imports
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.entrypoints.openai.api_server import run_server
import uvicorn

def main():
    # Write PID
    with open("/data/qwen-awq-miner/vllm.pid", "w") as f:
        f.write(str(os.getpid()))
    
    # Model and engine args
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=12288,
        trust_remote_code=True,
        tool_call_parser="hermes"
    )
    
    # Start server
    uvicorn.run(
        "vllm.entrypoints.openai.api_server:app",
        host="0.0.0.0",
        port=8000,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": "/data/qwen-awq-miner/logs/vllm.out",
                    "mode": "a"
                }
            },
            "root": {
                "handlers": ["file"]
            }
        },
        factory=True,
        app_dir="",
        factory_kwargs={"engine_args": engine_args}
    )

if __name__ == "__main__":
    main()
EOF

# Create simple start script
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0

# Log to file
exec python vllm_server.py >> logs/vllm.out 2>&1
EOF
chmod +x start_vllm.sh

# Create management script
echo "✍️ Creating vllm_manage.sh..."
cat > vllm_manage.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

case "$1" in
    start)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "vLLM already running with PID $(cat vllm.pid)"
        else
            nohup ./start_vllm.sh > /dev/null 2>&1 &
            echo $! > vllm.pid
            echo "Started vLLM with PID $(cat vllm.pid)"
            echo "Logs: tail -f logs/vllm.out"
        fi
        ;;
    stop)
        if [ -f vllm.pid ]; then
            kill $(cat vllm.pid) 2>/dev/null && echo "Stopped vLLM"
            rm -f vllm.pid
        else
            echo "No PID file found"
        fi
        ;;
    status)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "vLLM running with PID $(cat vllm.pid)"
        else
            echo "vLLM not running"
        fi
        ;;
    logs)
        tail -f logs/vllm.out
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        ;;
esac
EOF
chmod +x vllm_manage.sh

# Create OpenAI-compatible client example
echo "✍️ Creating test_openai_tools.py..."
cat > test_openai_tools.py << 'EOF'
#!/usr/bin/env python3
"""Test OpenAI Tools interface"""
import requests
import json

url = "http://localhost:8000/v1/chat/completions"

# Test with tools
data = {
    "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    "messages": [
        {"role": "user", "content": "What fish dishes do you have?"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "search_menu_items",
            "description": "Search for menu items by ingredient, category, or name",
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
    "tool_choice": "auto"
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
EOF
chmod +x test_openai_tools.py

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Environment: /data/qwen-awq-miner/.venv (Python 3.11)"
echo "PyTorch: 2.7.1 with CUDA 12.8 (or 12.1 fallback)"
echo "vLLM: 0.10.1.1 with xFormers 0.0.31"
echo "Model: Qwen2.5-7B-Instruct-AWQ"
echo "Context: 12,288 tokens"
echo "Tool parser: hermes"
echo ""
echo "To manage vLLM:"
echo "  ./vllm_manage.sh start    # Start server"
echo "  ./vllm_manage.sh stop     # Stop server"
echo "  ./vllm_manage.sh status   # Check status"
echo "  ./vllm_manage.sh logs     # View logs"
echo ""
echo "Test OpenAI Tools:"
echo "  ./test_openai_tools.py"
echo ""
echo "API endpoint: http://localhost:8000/v1/*"