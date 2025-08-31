#!/bin/bash
# Universal vLLM MIA Installer - Works anywhere
# Creates environment if needed, uses existing if present

echo "🚀 Universal vLLM Installer with MIA Tool Calling"
echo "==============================================="

# Determine base directory
if [ -d "/data" ]; then
    BASE_DIR="/data"
else
    BASE_DIR="$HOME"
fi

MINER_DIR="$BASE_DIR/qwen-awq-miner"
echo "📁 Using directory: $MINER_DIR"

# Create directory if needed
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Install system dependencies if needed
if ! command -v python3.11 &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "📦 Installing Python and dependencies..."
    apt-get update -qq
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip git wget curl build-essential 2>/dev/null || \
    apt-get install -y python3 python3-venv python3-dev python3-pip git wget curl build-essential 2>/dev/null
fi

# Determine Python command
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check for existing venv
if [ -d ".venv" ]; then
    echo "✅ Found existing .venv"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "✅ Found existing venv"
    source venv/bin/activate
else
    echo "📦 Creating new virtual environment..."
    $PYTHON_CMD -m venv .venv
    source .venv/bin/activate
fi

# Upgrade pip, wheel, setuptools
echo "📦 Upgrading pip, wheel, setuptools..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA detection
echo "🔥 Installing PyTorch..."
# Try CUDA 12.8 first, fallback to 12.1, then CPU
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 -f https://download.pytorch.org/whl/torch_stable.html 2>/dev/null || \
pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html 2>/dev/null || \
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install vLLM and dependencies
echo "📦 Installing vLLM 0.10.1.1 and dependencies..."
pip install vllm==0.10.1.1 xformers==0.0.31 transformers accelerate

# Try flash-attn only if wheel exists
pip install flash-attn 2>/dev/null || echo "⚠️  No flash-attn wheel available, skipping"

# Create logs directory
mkdir -p logs

# Create vLLM server script
echo "✍️ Creating vllm_serve.sh..."
cat > vllm_serve.sh << 'EOF'
#!/bin/bash
# vLLM Server with MIA Tool Calling

# Find and activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0

# Write PID
echo $$ > vllm.pid

# Start vLLM with tool calling enabled
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
chmod +x vllm_serve.sh

# Create management script
echo "✍️ Creating vllm_manage.sh..."
cat > vllm_manage.sh << 'EOF'
#!/bin/bash

cd "$(dirname "$0")"

case "$1" in
    start)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "vLLM already running with PID $(cat vllm.pid)"
        else
            echo "Starting vLLM with Hermes tool parser..."
            nohup ./vllm_serve.sh > /dev/null 2>&1 &
            sleep 3
            if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
                echo "✅ Started vLLM with PID $(cat vllm.pid)"
                echo "📋 Logs: tail -f logs/vllm.out"
                echo "🌐 API: http://localhost:8000/v1/"
            else
                echo "❌ Failed to start - check logs/vllm.out"
            fi
        fi
        ;;
    stop)
        if [ -f vllm.pid ]; then
            PID=$(cat vllm.pid)
            if kill $PID 2>/dev/null; then
                echo "✅ Stopped vLLM (PID $PID)"
            fi
            rm -f vllm.pid
        else
            echo "No PID file found"
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "✅ vLLM running with PID $(cat vllm.pid)"
            echo "Checking features..."
            grep -q "hermes" logs/vllm.out 2>/dev/null && echo "✅ Hermes parser active" || echo "⏳ Waiting for Hermes confirmation"
            grep -q "auto_tool_choice" logs/vllm.out 2>/dev/null && echo "✅ Auto tool choice enabled" || echo "⏳ Waiting for auto tool confirmation"
        else
            echo "❌ vLLM not running"
        fi
        ;;
    logs)
        tail -f logs/vllm.out
        ;;
    test)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "Testing vLLM endpoint..."
            curl -s http://localhost:8000/v1/models | python -m json.tool || echo "❌ API not responding"
        else
            echo "❌ vLLM not running"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        ;;
esac
EOF
chmod +x vllm_manage.sh

# Create test script
echo "✍️ Creating test_tools.py..."
cat > test_tools.py << 'EOF'
#!/usr/bin/env python3
"""Test vLLM Tool Calling"""
import requests
import json
import sys

def test_health():
    """Test if server is running"""
    try:
        resp = requests.get("http://localhost:8000/v1/models", timeout=5)
        if resp.status_code == 200:
            print("✅ Server is running")
            return True
    except:
        print("❌ Server not responding")
        return False

def test_tool_calling():
    """Test tool calling functionality"""
    url = "http://localhost:8000/v1/chat/completions"
    
    # Test with tools
    data = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": "You are Maria at Bella Vista Restaurant. Use tools for menu questions."},
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
    
    try:
        resp = requests.post(url, json=data, timeout=30)
        result = resp.json()
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
            
        message = result["choices"][0]["message"]
        if "tool_calls" in message:
            print("✅ Tool calling works!")
            print(f"   Called: {message['tool_calls'][0]['function']['name']}")
            print(f"   Args: {message['tool_calls'][0]['function']['arguments']}")
        else:
            print("⚠️  No tool was called")
            print(f"   Response: {message.get('content', '')[:100]}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🧪 Testing vLLM Server\n")
    if test_health():
        print("\n📞 Testing tool calling...")
        test_tool_calling()
    else:
        print("\nStart the server with: ./vllm_manage.sh start")
EOF
chmod +x test_tools.py

# Create the MIA restaurant tools module
echo "✍️ Creating mia_restaurant_tools.py..."
cat > mia_restaurant_tools.py << 'EOF'
#!/usr/bin/env python3
"""MIA Restaurant Tools Definition"""

RESTAURANT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_menu_items",
        "description": "Search for menu items by ingredient, category, or name",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The term to search for"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["ingredient", "category", "name"],
                    "description": "Type of search to perform"
                }
            },
            "required": ["search_term", "search_type"]
        }
    }
}, {
    "type": "function", 
    "function": {
        "name": "get_dish_details",
        "description": "Get detailed information about a specific dish",
        "parameters": {
            "type": "object",
            "properties": {
                "dish": {
                    "type": "string",
                    "description": "Name of the dish"
                }
            },
            "required": ["dish"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "filter_by_dietary",
        "description": "Filter menu items by dietary restrictions",
        "parameters": {
            "type": "object",
            "properties": {
                "diet": {
                    "type": "string",
                    "description": "Dietary restriction (e.g., vegetarian, vegan, gluten-free)"
                }
            },
            "required": ["diet"]
        }
    }
}]

SYSTEM_PROMPT = """You are Maria, a friendly server at Bella Vista Restaurant. Be concise and helpful. Use tools for menu/food questions when available. If tools are not needed, answer directly. If information is missing, ask one short clarifying question."""

def get_tool_choice(user_message):
    """Determine tool choice based on intent"""
    message_lower = user_message.lower()
    
    # Clear menu/DB queries - force tool
    menu_keywords = ["menu", "dish", "food", "price", "ingredient", "vegetarian", "vegan", "fish", "meat", "pasta"]
    if any(keyword in message_lower for keyword in menu_keywords):
        return "auto"  # Let model decide which tool
    
    # Greetings/small talk - no tools
    greeting_keywords = ["hi", "hello", "how are you", "good morning", "good evening"]
    if any(keyword in message_lower for keyword in greeting_keywords):
        return "none"
    
    # Ambiguous - let model decide
    return "auto"

if __name__ == "__main__":
    print("MIA Restaurant Tools loaded")
    print(f"Tools available: {len(RESTAURANT_TOOLS)}")
    for tool in RESTAURANT_TOOLS:
        print(f"  - {tool['function']['name']}")
EOF

echo ""
echo "✅ Universal Installation Complete!"
echo "=================================="
echo ""
echo "📁 Installed at: $MINER_DIR"
echo "🐍 Python: $($PYTHON_CMD --version)"
echo "🔥 PyTorch: $(pip show torch | grep Version | cut -d' ' -f2)"
echo "⚡ vLLM: 0.10.1.1 with Hermes parser"
echo ""
echo "To start vLLM:"
echo "  cd $MINER_DIR"
echo "  ./vllm_manage.sh start"
echo ""
echo "To test:"
echo "  ./test_tools.py"
echo ""
echo "Management commands:"
echo "  ./vllm_manage.sh start    # Start server"
echo "  ./vllm_manage.sh stop     # Stop server"
echo "  ./vllm_manage.sh status   # Check status"
echo "  ./vllm_manage.sh logs     # View logs"
echo "  ./vllm_manage.sh test     # Test API"