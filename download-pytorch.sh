#!/bin/bash

# Script to download PyTorch with resume support

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}PyTorch Downloader with Resume Support${NC}"
echo "======================================"

# Create download directory
DOWNLOAD_DIR="/opt/pytorch-download"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Detect Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d, -f1 | cut -dV -f2)
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    echo -e "${GREEN}CUDA version: $CUDA_VERSION${NC}"
else
    CUDA_MAJOR="11"
    echo -e "${YELLOW}CUDA not detected, defaulting to CUDA 11.8${NC}"
fi

# Determine PyTorch URL
if [[ "$CUDA_MAJOR" == "12" ]]; then
    CUDA_TAG="cu121"
else
    CUDA_TAG="cu118"
fi

if [[ "$PYTHON_VERSION" == "3.8" ]]; then
    PY_TAG="cp38-cp38"
elif [[ "$PYTHON_VERSION" == "3.9" ]]; then
    PY_TAG="cp39-cp39"
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    PY_TAG="cp310-cp310"
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    PY_TAG="cp311-cp311"
else
    echo -e "${RED}Unsupported Python version: $PYTHON_VERSION${NC}"
    exit 1
fi

# PyTorch URLs
TORCH_FILE="torch-2.4.1+${CUDA_TAG}-${PY_TAG}-linux_x86_64.whl"
TORCH_URL="https://download.pytorch.org/whl/${CUDA_TAG}/${TORCH_FILE}"

TORCHVISION_FILE="torchvision-0.19.1+${CUDA_TAG}-${PY_TAG}-linux_x86_64.whl"
TORCHVISION_URL="https://download.pytorch.org/whl/${CUDA_TAG}/${TORCHVISION_FILE}"

TORCHAUDIO_FILE="torchaudio-2.4.1+${CUDA_TAG}-${PY_TAG}-linux_x86_64.whl"
TORCHAUDIO_URL="https://download.pytorch.org/whl/${CUDA_TAG}/${TORCHAUDIO_FILE}"

echo -e "\n${YELLOW}Downloading PyTorch packages...${NC}"
echo "This may take a while due to large file sizes."
echo "Downloads will resume if interrupted."

# Function to download with resume
download_with_resume() {
    local url=$1
    local filename=$2
    
    echo -e "\n${GREEN}Downloading $filename...${NC}"
    
    # Use wget with resume support
    wget -c --progress=bar:force --tries=10 --retry-connrefused "$url" -O "$filename" || {
        echo -e "${RED}Failed to download $filename${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ Downloaded $filename${NC}"
    return 0
}

# Download all packages
download_with_resume "$TORCH_URL" "$TORCH_FILE" || exit 1
download_with_resume "$TORCHVISION_URL" "$TORCHVISION_FILE" || exit 1
download_with_resume "$TORCHAUDIO_URL" "$TORCHAUDIO_FILE" || exit 1

echo -e "\n${GREEN}All packages downloaded successfully!${NC}"

# Now install from local files
echo -e "\n${YELLOW}Installing PyTorch from local files...${NC}"

# Make sure we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "/opt/mia-gpu-miner/venv" ]; then
        source /opt/mia-gpu-miner/venv/bin/activate
    else
        echo -e "${RED}Virtual environment not found!${NC}"
        echo "Please activate your virtual environment first."
        exit 1
    fi
fi

# Install the packages
pip install --no-deps "$TORCH_FILE" || exit 1
pip install --no-deps "$TORCHVISION_FILE" || exit 1
pip install --no-deps "$TORCHAUDIO_FILE" || exit 1

# Install dependencies
pip install numpy pillow typing-extensions

echo -e "\n${GREEN}✓ PyTorch installation complete!${NC}"

# Test installation
echo -e "\n${YELLOW}Testing PyTorch installation...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"

echo -e "\n${GREEN}Success! You can now continue with the miner setup.${NC}"
echo "Run: cd /opt/mia-gpu-miner && ./setup-gpu-miner.sh"

# Optionally clean up downloads
echo -e "\n${YELLOW}Downloaded files are in: $DOWNLOAD_DIR${NC}"
echo "You can delete them with: rm -rf $DOWNLOAD_DIR"