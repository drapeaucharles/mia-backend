#!/bin/bash
# Start vLLM with OpenAI-compatible API server for tool calling

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting vLLM OpenAI-compatible server for Qwen2.5-7B-Instruct-AWQ${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run the installer first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Kill any existing vLLM processes
echo -e "${YELLOW}Stopping any existing vLLM servers...${NC}"
pkill -f "vllm.entrypoints.openai" || true
sleep 2

# Set environment variables
export VLLM_ATTENTION_BACKEND="XFORMERS"
export CUDA_VISIBLE_DEVICES="0"

# Create the vLLM OpenAI server script if it doesn't exist
cat > vllm_openai_server.py << 'EOF'
#!/usr/bin/env python3
"""
vLLM OpenAI-compatible API Server for Qwen2.5-7B-Instruct-AWQ
Supports tool/function calling
"""
import os
import sys

# Force xFormers backend for performance
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

# Import vLLM's OpenAI-compatible server
from vllm.entrypoints.openai import api_server

if __name__ == "__main__":
    # Server arguments optimized for tool calling
    args = [
        "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "--host", "0.0.0.0",
        "--port", "8000",
        
        # Context and performance settings
        "--max-model-len", "12288",  # 12k context as requested
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.95",
        
        # Tool calling support
        "--enable-prefix-caching",  # Important for tool calling efficiency
        "--max-num-seqs", "256",
        
        # AWQ quantization
        "--quantization", "awq",
        "--dtype", "float16",
        
        # Performance optimizations
        "--disable-log-requests",
        "--disable-log-stats",
        
        # Enable chat template for proper tool formatting
        "--chat-template", "/opt/mia-gpu-miner/chat_template.jinja"
    ]
    
    # Set argv for the API server
    sys.argv = ["vllm.entrypoints.openai.api_server"] + args
    
    print("\n🚀 Starting vLLM OpenAI-compatible API server...")
    print(f"   Model: Qwen/Qwen2.5-7B-Instruct-AWQ")
    print(f"   Context: 12k tokens")
    print(f"   Endpoint: http://localhost:8000/v1")
    print(f"   Tool calling: ENABLED")
    print("\n📝 API endpoints:")
    print("   - POST /v1/chat/completions (with tools parameter)")
    print("   - POST /v1/completions")
    print("   - GET /v1/models")
    print("\n")
    
    api_server.main()
EOF

# Create Qwen chat template for proper tool formatting
cat > chat_template.jinja << 'EOF'
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = "You are a helpful assistant." -%}
{%- endif -%}

{{ '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}

{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'tool' -%}
        {{ '<|im_start|>tool\n' + message['content'] + '<|im_end|>\n' }}
    {%- endif -%}
{%- endfor -%}

{%- if add_generation_prompt -%}
    {{ '<|im_start|>assistant\n' }}
{%- endif -%}
EOF

# Start the server
echo -e "${GREEN}Starting vLLM OpenAI API server...${NC}"
python vllm_openai_server.py 2>&1 | tee vllm_openai.log &
VLLM_PID=$!

# Wait for server to be ready
echo -e "${YELLOW}Waiting for server to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM OpenAI API server is ready!${NC}"
        echo -e "${GREEN}Server PID: $VLLM_PID${NC}"
        echo ""
        echo "Test with:"
        echo 'curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "messages": [{"role": "user", "content": "Hello"}]}"'
        echo ""
        echo "To stop: kill $VLLM_PID"
        exit 0
    fi
    sleep 2
done

echo -e "${RED}❌ Server failed to start. Check vllm_openai.log for errors.${NC}"
exit 1