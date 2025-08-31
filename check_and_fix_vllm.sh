#!/bin/bash
# Check and fix vLLM installation

cd /data/qwen-awq-miner

echo "🔍 Checking installation..."

# Activate venv
source .venv/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check what's installed
echo -e "\n📦 Installed packages:"
pip list | grep -E "(torch|vllm|xformers)" || echo "No relevant packages found"

# Check if vLLM failed to install
echo -e "\n🔍 Checking vLLM..."
python -c "import vllm; print('vLLM is installed')" 2>&1 || {
    echo "❌ vLLM not installed"
    
    echo -e "\n📦 Installing vLLM with wheels-only policy..."
    
    # Set strict pip policy
    export PIP_INDEX_URL="https://pypi.org/simple"
    export PIP_ONLY_BINARY=":all:"
    export PIP_NO_CACHE_DIR="1"
    
    # First ensure we have PyTorch
    echo "Checking PyTorch..."
    python -c "import torch" 2>/dev/null || {
        echo "Installing PyTorch first..."
        pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
            --index-url https://download.pytorch.org/whl/cu121
    }
    
    # Try different vLLM versions
    echo -e "\nTrying vLLM 0.2.7 (more stable)..."
    pip install vllm==0.2.7 || {
        echo "Failed with 0.2.7, trying 0.3.0..."
        pip install vllm==0.3.0 || {
            echo "Failed with 0.3.0, trying 0.4.0..."
            pip install vllm==0.4.0 || {
                echo "❌ All vLLM versions failed to install as wheels"
                echo "This likely means no prebuilt wheels exist for your platform"
            }
        }
    }
}

# Final check
echo -e "\n✅ Final status:"
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
except:
    print('PyTorch: NOT INSTALLED')
try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except:
    print('vLLM: NOT INSTALLED')
"

# If vLLM is installed, update the start script for the installed version
if python -c "import vllm" 2>/dev/null; then
    echo -e "\n✅ vLLM is now installed"
    echo "You can start the server with: ./start_vllm.sh"
else
    echo -e "\n❌ vLLM installation failed"
    echo "This usually means:"
    echo "1. No wheel exists for Python $(python --version | cut -d' ' -f2) on your platform"
    echo "2. The package requires building from source (which we've disabled)"
    echo ""
    echo "Options:"
    echo "1. Use a different Python version (3.9 or 3.10 often have more wheels)"
    echo "2. Use a Docker image with vLLM pre-installed"
    echo "3. Allow source builds (not recommended due to dependency issues)"
fi