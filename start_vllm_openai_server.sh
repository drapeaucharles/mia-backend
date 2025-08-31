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

# Use vLLM's direct CLI command
echo -e "${GREEN}Starting vLLM OpenAI API server...${NC}"
echo ""
echo "🚀 Starting vLLM OpenAI-compatible API server..."
echo "   Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "   Context: 12k tokens"
echo "   Endpoint: http://localhost:8000/v1"
echo "   Tool calling: ENABLED"
echo ""
echo "📝 API endpoints:"
echo "   - POST /v1/chat/completions (with tools parameter)"
echo "   - POST /v1/completions"
echo "   - GET /v1/models"
echo ""

# Start the server using vllm serve command
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 12288 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.95 \
    --quantization awq \
    --dtype float16 \
    --enable-prefix-caching \
    --max-num-seqs 256 \
    --disable-log-requests \
    --disable-log-stats \
    2>&1 | tee vllm_server.log &
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