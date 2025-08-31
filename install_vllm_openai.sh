#!/bin/bash
# Quick installer for vLLM OpenAI server with proper tool calling

cd /data/qwen-awq-miner

# Stop current miner
echo "Stopping current miner..."
pkill -f miner.py || true
pkill -f vllm || true
sleep 2

# Create the vLLM OpenAI server script
cat > start_vllm_openai.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Set environment
export VLLM_ATTENTION_BACKEND="XFORMERS"
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="1"

echo "🚀 Starting vLLM OpenAI-compatible API server..."
echo "   Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "   Context: 12k tokens"
echo "   Endpoint: http://localhost:8000/v1"
echo "   Tool calling: ENABLED"

# Run vLLM OpenAI API server directly
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
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --disable-log-requests \
    --disable-log-stats
EOF

chmod +x start_vllm_openai.sh

echo "✓ Created vLLM OpenAI server script"
echo "Starting server..."

# Run it
nohup ./start_vllm_openai.sh > vllm_openai.log 2>&1 &
VLLM_PID=$!

echo "vLLM OpenAI server starting with PID: $VLLM_PID"
echo ""
echo "Check if it's running:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "View logs:"
echo "  tail -f /data/qwen-awq-miner/vllm_openai.log"
echo ""
echo "To stop:"
echo "  kill $VLLM_PID"