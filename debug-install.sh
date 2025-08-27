#!/bin/bash

# Debug installer to find the exact error

set -x  # Show all commands being executed

echo "=== System Info ==="
python3 --version
pip --version
nvcc --version || echo "CUDA not found in PATH"

echo "=== Creating test environment ==="
cd /data
mkdir -p test-install
cd test-install

echo "=== Creating venv ==="
python3 -m venv test_venv
source test_venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing packages one by one ==="

echo "1. Testing torch installation..."
pip install torch --index-url https://download.pytorch.org/whl/cu118 2>&1 | tee torch_install.log

echo "2. Testing torchvision installation..."
pip install torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tee torchvision_install.log

echo "3. Testing torchaudio installation..."
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tee torchaudio_install.log

echo "4. Testing vllm installation..."
pip install vllm 2>&1 | tee vllm_install.log

echo "5. Testing other packages..."
pip install flask waitress requests aiohttp 2>&1 | tee other_install.log

echo "=== Checking for errors ==="
grep -i "error\|fail\|puciialin" *.log

echo "=== Installed packages ==="
pip list

echo "Done. Check the .log files for detailed error messages."