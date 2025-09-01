#!/bin/bash
# Start vLLM with OpenAI-compatible API server for tool calling
# Uses the direct vLLM CLI command (no Python wrapper needed)

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
pkill -f "vllm serve" || true
sleep 2

# Set environment variables
export VLLM_ATTENTION_BACKEND="XFORMERS"
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="1"

# Start the server
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

# Run vLLM directly with the serve command
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.95 \
    --max-model-len 12288 \
    --trust-remote-code \
    --enforce-eager \
    --disable-custom-all-reduce \
    --max-num-seqs 8 \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests \
    --disable-log-stats \
    2>&1 | tee vllm_server.log &

VLLM_PID=$!
echo $VLLM_PID > vllm_server.pid

# Wait for server to be ready
echo -e "${YELLOW}Waiting for server to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✅ vLLM OpenAI API server is ready!${NC}"
        echo -e "${GREEN}Server PID: $VLLM_PID${NC}"
        echo ""
        echo "Test with:"
        echo 'curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '\''{
  "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
  "messages": [{"role": "user", "content": "Hello"}]
}'\'''
        echo ""
        echo "To stop: kill $VLLM_PID"
        echo "Logs: tail -f vllm_server.log"
        exit 0
    fi
    sleep 2
done

echo -e "${RED}❌ Server failed to start. Check vllm_server.log for errors.${NC}"
exit 1