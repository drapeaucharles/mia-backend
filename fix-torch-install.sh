#!/bin/bash

# Fix PyTorch installation for existing miners
echo "Fixing PyTorch and auto-gptq installation..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detect installation directory
if [ -d "/opt/mia-gpu-miner" ]; then
    INSTALL_DIR="/opt/mia-gpu-miner"
    echo "Found installation at /opt/mia-gpu-miner"
elif [ -d "$HOME/mia-gpu-miner" ]; then
    INSTALL_DIR="$HOME/mia-gpu-miner"
    echo "Found installation at ~/mia-gpu-miner"
else
    echo -e "${RED}Error: No miner installation found${NC}"
    exit 1
fi

cd "$INSTALL_DIR"

# Stop any running miners
echo "Stopping existing services..."
sudo systemctl stop mia-miner 2>/dev/null || true
sudo systemctl stop mia-gpu-miner 2>/dev/null || true
pkill -f "mia_miner_unified.py" 2>/dev/null || true

# Activate venv
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check current PyTorch status
echo -e "\n${YELLOW}Checking current PyTorch installation...${NC}"
python -c "import torch; print(f'Current PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"

# Reinstall PyTorch with CUDA
echo -e "\n${YELLOW}Installing PyTorch with CUDA 11.8...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
echo -e "\n${YELLOW}Verifying PyTorch installation...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || {
    echo -e "${RED}PyTorch installation failed!${NC}"
    exit 1
}

# Install auto-gptq with proper CUDA support
echo -e "\n${YELLOW}Installing auto-gptq...${NC}"
pip uninstall -y auto-gptq 2>/dev/null || true
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# Verify auto-gptq
echo -e "\n${YELLOW}Verifying auto-gptq installation...${NC}"
python -c "import auto_gptq; print('auto-gptq installed successfully')" || {
    echo -e "${RED}auto-gptq installation failed!${NC}"
    exit 1
}

# Reinstall other packages to ensure compatibility
echo -e "\n${YELLOW}Reinstalling other packages...${NC}"
pip install --upgrade transformers accelerate optimum

# Test model loading
echo -e "\n${YELLOW}Testing model loading...${NC}"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Testing tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Open-Orca/Mistral-7B-OpenOrca')
print('✓ Tokenizer loads successfully')
print('Model loading will be tested when miner starts')
"

# Restart service if it exists
if [ -f "/etc/systemd/system/mia-miner.service" ]; then
    echo -e "\n${YELLOW}Starting miner service...${NC}"
    sudo systemctl start mia-miner
    echo -e "${GREEN}✓ Service started${NC}"
    echo "Check status: sudo systemctl status mia-miner"
    echo "View logs: sudo journalctl -u mia-miner -f"
else
    echo -e "\n${GREEN}✓ PyTorch and auto-gptq fixed!${NC}"
    echo ""
    echo "To start your miner:"
    echo "  cd $INSTALL_DIR"
    echo "  ./run_miner.sh"
fi