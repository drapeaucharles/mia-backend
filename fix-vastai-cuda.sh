#!/bin/bash

# Fix CUDA detection on Vast.ai
echo "Fixing CUDA detection on Vast.ai..."

# Set CUDA paths
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"

# Check if we're in the miner directory
cd ~/mia-gpu-miner || exit 1
source venv/bin/activate

# Test CUDA availability
echo "Testing CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'PyTorch version: {torch.__version__}')"

# If CUDA still not detected, try reinstalling PyTorch
read -p "Is CUDA detected? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Reinstalling PyTorch with CUDA support..."
    pip uninstall -y torch torchvision torchaudio
    
    # Install PyTorch with explicit CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Test again
    python3 -c "import torch; print(f'CUDA available after reinstall: {torch.cuda.is_available()}')"
fi

# Create a wrapper script that ensures CUDA paths
cat > run_with_cuda.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Ensure CUDA paths are set
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
export CUDA_VISIBLE_DEVICES=0

# Check CUDA before starting
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"

# Run the miner
python mia_miner_unified.py
EOF
chmod +x run_with_cuda.sh

echo ""
echo "Fix applied! Now run:"
echo "  cd ~/mia-gpu-miner"
echo "  ./run_with_cuda.sh"
echo ""
echo "If CUDA is still not detected, try:"
echo "1. Check Vast.ai uses a CUDA-enabled image"
echo "2. Run: nvidia-smi (should show your GPU)"
echo "3. Contact Vast.ai support"