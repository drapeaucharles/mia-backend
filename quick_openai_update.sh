#!/bin/bash
# Quick OpenAI tools update for qwen-awq-miner

cd /data/qwen-awq-miner

# Stop current miner
pkill -f miner.py || true
sleep 2

# Download vLLM OpenAI server script directly
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/start_vllm_openai_server.sh -o start_vllm_openai_server.sh
chmod +x start_vllm_openai_server.sh

# Download OpenAI tools miner (optional)
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/miner_openai_tools.py -o miner_openai_tools.py

echo "✓ Downloaded OpenAI tools support"
echo "Starting vLLM OpenAI server..."

# Start the server
./start_vllm_openai_server.sh