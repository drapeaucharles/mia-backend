#!/bin/bash

# Quick script to continue setup after PyTorch is installed

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Continuing MIA Miner Setup (PyTorch already installed)${NC}"
echo "===================================================="

# Check if we're in the right place
if [ ! -d "/opt/mia-gpu-miner/venv" ]; then
    echo -e "${RED}Error: /opt/mia-gpu-miner/venv not found${NC}"
    echo "Please run the full installer first."
    exit 1
fi

cd /opt/mia-gpu-miner
source venv/bin/activate

# Test PyTorch
echo -e "${YELLOW}Checking PyTorch installation...${NC}"
python3 -c "import torch; print(f'PyTorch {torch.__version__} is installed')" || {
    echo -e "${RED}PyTorch not found! Please install it first.${NC}"
    exit 1
}

# Continue with the rest of the installation
echo -e "\n${YELLOW}Installing vLLM and other dependencies...${NC}"

# Install dependencies
pip install --no-cache-dir numpy scipy
pip install --no-cache-dir transformers>=4.36.0 accelerate sentencepiece protobuf
pip install --no-cache-dir requests psutil gpustat py-cpuinfo uvicorn fastapi

# Install vLLM
echo "Installing vLLM..."
pip install --no-cache-dir vllm==0.2.7 || {
    echo -e "${YELLOW}Trying latest vLLM version...${NC}"
    pip install --no-cache-dir vllm
}

# Download the setup script to get the embedded files
wget -q https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/setup-gpu-miner.sh -O temp_setup.sh

# Extract and run the rest of the setup starting from step 5
sed -n '/# Step 5: Download configuration files/,$p' temp_setup.sh > continue_setup.sh
chmod +x continue_setup.sh

# Update the step numbers in the extracted script
sed -i 's/\[5\/6\]/[1\/2]/g' continue_setup.sh
sed -i 's/\[6\/6\]/[2\/2]/g' continue_setup.sh

# Run the rest of the setup
bash continue_setup.sh

rm -f temp_setup.sh continue_setup.sh