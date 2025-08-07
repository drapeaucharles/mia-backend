#!/bin/bash

# Quick fix for PyTorch installation issues

echo "Fixing PyTorch installation..."

# Clear pip cache
pip cache purge 2>/dev/null || true

# Remove any partial downloads
rm -rf ~/.cache/pip 2>/dev/null || true

# Try installing PyTorch with no-cache-dir flag
echo "Installing PyTorch for CUDA 12.1..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $? -eq 0 ]; then
    echo "PyTorch installed successfully!"
else
    echo "Trying CUDA 11.8 version..."
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo "Testing PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"