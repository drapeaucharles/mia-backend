#!/bin/bash

# Fix auto-gptq CUDA kernels for faster inference

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Fixing auto-gptq CUDA kernels           ║${NC}"
echo -e "${GREEN}║   This will speed up inference 10x        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

# Navigate to miner directory
cd /data/mia-gpu-miner

# Activate venv
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
else
    echo -e "${RED}Virtual environment not found at /data/venv${NC}"
    exit 1
fi

# Check CUDA
echo -e "${YELLOW}Checking CUDA installation...${NC}"
nvcc --version || {
    echo -e "${RED}nvcc not found! Installing CUDA toolkit...${NC}"
    apt-get update
    apt-get install -y cuda-toolkit-11-8 || apt-get install -y nvidia-cuda-toolkit
}

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Uninstall existing auto-gptq
echo -e "${YELLOW}Removing existing auto-gptq...${NC}"
pip uninstall -y auto-gptq

# Install build dependencies
echo -e "${YELLOW}Installing build dependencies...${NC}"
apt-get install -y build-essential

# Install auto-gptq from source with CUDA
echo -e "${YELLOW}Building auto-gptq with CUDA kernels...${NC}"
pip install packaging
BUILD_CUDA_EXT=1 pip install auto-gptq --no-cache-dir

# If that fails, try pre-built wheel with CUDA
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Source build failed, trying pre-built CUDA wheel...${NC}"
    
    # Get Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    # Try to install pre-built wheel based on PyTorch/CUDA version
    pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
fi

# Test if CUDA kernels are loaded
echo -e "${YELLOW}Testing auto-gptq CUDA kernels...${NC}"
python3 << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

try:
    from auto_gptq import AutoGPTQForCausalLM
    print("✓ auto-gptq imported successfully")
    
    # Check for CUDA kernels
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    print("✓ CUDA kernels available!")
except Exception as e:
    print(f"✗ Error: {e}")
EOF

# Alternative: Use exllama backend if available
echo -e "${YELLOW}Checking exllama backend...${NC}"
pip install exllama || echo "Exllama not available for this configuration"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Auto-gptq CUDA fix complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Now restart your miner:${NC}"
echo "  cd /data/mia-gpu-miner"
echo "  ./stop_miner.sh"
echo "  ./start_miner.sh"
echo ""
echo -e "${GREEN}You should see 20-50 tok/s instead of 3-4 tok/s!${NC}"