#!/bin/bash
# Check current setup

echo "🔍 Checking current setup..."
echo ""

# Check what's in /data
echo "1. Contents of /data:"
ls -la /data/ 2>/dev/null || echo "Cannot access /data"

# Check qwen-awq-miner directory
echo -e "\n2. Contents of /data/qwen-awq-miner:"
ls -la /data/qwen-awq-miner/ 2>/dev/null || echo "Directory doesn't exist"

# Check for any venv
echo -e "\n3. Looking for Python environments:"
find /data -name "venv" -o -name ".venv" -o -name "*env*" -type d 2>/dev/null | head -10

# Check current directory
echo -e "\n4. Current directory:"
pwd
ls -la

echo -e "\n5. Python version:"
python3 --version 2>/dev/null || python --version