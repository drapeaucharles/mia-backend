#!/bin/bash
# vLLM Installer - MIA Tool Calling with Hermes Parser
# Model decides when to call tools (auto mode)

echo "🚀 Installing vLLM with MIA Tool Calling (Hermes)"
echo "==============================================="

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

# Create vLLM server with Hermes parser and auto tool choice
echo "✍️ Creating vllm_server.py with tool calling enabled..."
cat > vllm_server.py << 'EOF'
#!/usr/bin/env python3
"""vLLM Server with MIA Tool Calling (Hermes)"""
import os
import sys
import logging

# Set environment before imports
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/qwen-awq-miner/logs/vllm.out'),
        logging.StreamHandler(sys.stdout)
    ]
)

from vllm import LLM, SamplingParams

def main():
    # Write PID
    with open("/data/qwen-awq-miner/vllm.pid", "w") as f:
        f.write(str(os.getpid()))
    
    # Initialize model with tool calling enabled
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=12288,
        tool_call_parser="hermes",
        enable_auto_tool_choice=True
    )
    
    # Start OpenAI-compatible server
    from vllm.entrypoints.openai.api_server import run_server
    import uvicorn
    
    uvicorn.run(
        app=run_server(llm),
        host="0.0.0.0",
        port=8000,
        log_config=None  # We handle logging above
    )

if __name__ == "__main__":
    main()
EOF

# Alternative server using CLI (more stable)
echo "✍️ Creating vllm_serve.sh..."
cat > vllm_serve.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0

# Write PID
echo $$ > vllm.pid

# Start vLLM with all required flags
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
cd /data/qwen-awq-miner

case "$1" in
    start)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "vLLM already running with PID $(cat vllm.pid)"
        else
            echo "Starting vLLM with Hermes tool parser..."
            nohup ./vllm_serve.sh > /dev/null 2>&1 &
            sleep 2
            if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
                echo "Started vLLM with PID $(cat vllm.pid)"
                echo "Logs: tail -f logs/vllm.out"
            else
                echo "Failed to start vLLM - check logs/vllm.out"
            fi
        fi
        ;;
    stop)
        if [ -f vllm.pid ]; then
            PID=$(cat vllm.pid)
            if kill $PID 2>/dev/null; then
                echo "Stopped vLLM (PID $PID)"
            fi
            rm -f vllm.pid
        else
            echo "No PID file found"
        fi
        ;;
    status)
        if [ -f vllm.pid ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
            echo "vLLM running with PID $(cat vllm.pid)"
            echo "Checking for Hermes parser..."
            grep -q "hermes" logs/vllm.out && echo "✓ Hermes tool parser active" || echo "⚠ Hermes not confirmed in logs"
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

# Create test script for auto tool choice
echo "✍️ Creating test_auto_tools.py..."
cat > test_auto_tools.py << 'EOF'
#!/usr/bin/env python3
"""Test MIA Tool Calling with Auto Mode"""
import requests
import json
import time

base_url = "http://localhost:8000/v1"

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "search_menu_items",
        "description": "Search for menu items by ingredient, category, or name",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {"type": "string", "description": "The term to search for"},
                "search_type": {
                    "type": "string", 
                    "enum": ["ingredient", "category", "name"],
                    "description": "Type of search"
                }
            },
            "required": ["search_term", "search_type"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "get_dish_details",
        "description": "Get details about a specific dish",
        "parameters": {
            "type": "object",
            "properties": {
                "dish": {"type": "string", "description": "Name of the dish"}
            },
            "required": ["dish"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "filter_by_dietary",
        "description": "Filter menu by dietary restrictions",
        "parameters": {
            "type": "object",
            "properties": {
                "diet": {"type": "string", "description": "Dietary restriction"}
            },
            "required": ["diet"]
        }
    }
}]

def test_request(messages, tool_choice="auto", include_tools=True):
    """Make a test request"""
    data = {
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": messages,
        "temperature": 0,  # Stable tool arguments
        "max_tokens": 150
    }
    
    if include_tools:
        data["tools"] = tools
        data["tool_choice"] = tool_choice
    
    response = requests.post(f"{base_url}/chat/completions", json=data)
    return response.json()

print("🧪 Testing MIA Tool Calling with Hermes Parser\n")

# Test 1: Auto mode - should call tool
print("1️⃣ Test AUTO mode: 'I want fish'")
messages = [
    {"role": "system", "content": "You are Maria, a friendly server at Bella Vista Restaurant. Be concise and helpful. Use tools for menu/food questions when available."},
    {"role": "user", "content": "I want fish"}
]
result = test_request(messages, tool_choice="auto")
if "choices" in result and result["choices"][0]["message"].get("tool_calls"):
    print("✅ Tool called:", result["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
    print("   Arguments:", result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
else:
    print("❌ No tool call in response")

time.sleep(1)

# Test 2: No tools - greeting
print("\n2️⃣ Test NO TOOLS: 'Hi'")
messages = [
    {"role": "system", "content": "You are Maria, a friendly server at Bella Vista Restaurant."},
    {"role": "user", "content": "Hi"}
]
result = test_request(messages, include_tools=False)
if "choices" in result:
    print("✅ Response:", result["choices"][0]["message"]["content"][:100])

time.sleep(1)

# Test 3: Force specific tool
print("\n3️⃣ Test FORCE mode: 'price of Sea Bass'")
messages = [
    {"role": "system", "content": "You are Maria, a friendly server at Bella Vista Restaurant."},
    {"role": "user", "content": "What's the price of Sea Bass?"}
]
forced_choice = {"type": "function", "function": {"name": "get_dish_details"}}
result = test_request(messages, tool_choice=forced_choice)
if "choices" in result and result["choices"][0]["message"].get("tool_calls"):
    print("✅ Forced tool:", result["choices"][0]["message"]["tool_calls"][0]["function"]["name"])
    print("   Arguments:", result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

print("\n📊 Check server status with: ./vllm_manage.sh status")
EOF
chmod +x test_auto_tools.py

# Create MIA integration example
echo "✍️ Creating mia_integration.py..."
cat > mia_integration.py << 'EOF'
#!/usr/bin/env python3
"""MIA Backend Integration Example"""
import requests
import json

def handle_tool_request(user_message):
    """Complete tool calling flow"""
    base_url = "http://localhost:8000/v1/chat/completions"
    
    messages = [
        {
            "role": "system", 
            "content": "You are Maria, a friendly server at Bella Vista Restaurant. Be concise and helpful. Use tools for menu/food questions when available. If tools are not needed, answer directly. If information is missing, ask one short clarifying question."
        },
        {"role": "user", "content": user_message}
    ]
    
    tools = [{
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
    }]
    
    # Step 1: Get tool call from model
    print("🤖 Step 1: Model deciding on tools...")
    response1 = requests.post(base_url, json={
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0
    })
    
    result1 = response1.json()
    assistant_msg = result1["choices"][0]["message"]
    
    if "tool_calls" in assistant_msg:
        tool_call = assistant_msg["tool_calls"][0]
        print(f"✅ Tool called: {tool_call['function']['name']}")
        print(f"   Arguments: {tool_call['function']['arguments']}")
        
        # Step 2: Execute tool (mock result)
        tool_result = {
            "items": [
                {"name": "Grilled Salmon", "price": 24.99, "description": "Fresh Atlantic salmon"},
                {"name": "Sea Bass", "price": 28.99, "description": "Pan-seared Chilean sea bass"},
                {"name": "Fish Tacos", "price": 16.99, "description": "Crispy mahi-mahi tacos"}
            ]
        }
        
        # Step 3: Send tool result back for final response
        messages.append(assistant_msg)  # Add assistant's tool call
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(tool_result)
        })
        
        print("\n🤖 Step 2: Generating final response with tool data...")
        response2 = requests.post(base_url, json={
            "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "messages": messages,
            "temperature": 0.7
        })
        
        final_response = response2.json()["choices"][0]["message"]["content"]
        print(f"\n💬 Final response: {final_response}")
        
    else:
        print("ℹ️  Model chose not to use tools")
        print(f"💬 Response: {assistant_msg['content']}")

# Test the flow
if __name__ == "__main__":
    print("🔄 MIA Tool Calling Flow Demo\n")
    handle_tool_request("I want fish dishes")
EOF
chmod +x mia_integration.py

echo ""
echo "✅ Installation Complete!"
echo "================================"
echo ""
echo "Key changes for MIA tool calling:"
echo "  • tool_call_parser = 'hermes'"
echo "  • enable_auto_tool_choice = True"
echo "  • Temperature 0 for tool calls (stable arguments)"
echo ""
echo "Environment: /data/qwen-awq-miner/.venv (Python 3.11)"
echo "PyTorch: 2.7.1 with CUDA 12.8 (or 12.1 fallback)"
echo "vLLM: 0.10.1.1 with xFormers 0.0.31"
echo "Model: Qwen2.5-7B-Instruct-AWQ"
echo "Context: 12,288 tokens"
echo ""
echo "To start vLLM with tool calling:"
echo "  ./vllm_manage.sh start"
echo ""
echo "To test:"
echo "  ./test_auto_tools.py      # Test auto/force/none modes"
echo "  ./mia_integration.py      # Full tool flow example"
echo ""
echo "API endpoint: http://localhost:8000/v1/*"
echo "Logs: tail -f logs/vllm.out"