#!/bin/bash
# Clean everything and reinstall the working miner

echo "=== Clean and Reinstall Miner ==="

# 1. Stop all processes
echo "1. Stopping all processes..."
pkill -9 python
pkill -9 miner
sleep 2

# 2. Clean the data directory
echo "2. Cleaning /data directory..."
cd /data
rm -rf /data/qwen-awq-miner
rm -rf /data/venv
rm -rf /data/*.log

# 3. Clear caches
echo "3. Clearing caches..."
rm -rf ~/.cache/torch*
rm -rf ~/.cache/huggingface*
rm -rf /tmp/cuda*

# 4. Install the original working miner
echo "4. Installing original miner..."
curl -sSL https://gist.githubusercontent.com/drapeaucharles/79ba8bef4bb3794a50a5e5ef088630a5/raw/universal-miner-installer.sh | bash

echo -e "\n=== Installation Complete ==="
echo "The original miner is installed at: /data/qwen-awq-miner"
echo "Start with: cd /data/qwen-awq-miner && ./start_miner.sh"