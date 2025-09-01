#!/bin/bash
# Fix miner to use 7B model instead of 32B

cd /data/qwen-awq-miner

echo "🔧 Fixing model to use 7B instead of 32B"
echo "========================================"

# Stop miner
pkill -f miner.py

# Update miner.py to use 7B model
sed -i 's/Qwen2.5-32B-Instruct-AWQ/Qwen2.5-7B-Instruct-AWQ/g' miner.py
sed -i 's/max_model_len=12000/max_model_len=4096/g' miner.py

echo "✅ Updated to Qwen2.5-7B-Instruct-AWQ"
echo ""
echo "The 7B model:"
echo "- Only needs ~4GB disk space (not 20GB)"
echo "- Runs faster"
echo "- Still handles tools properly"
echo ""
echo "Start the miner: ./start_miner.sh"