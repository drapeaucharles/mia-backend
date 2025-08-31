#!/bin/bash
# Fix pip and install vLLM properly

cd /data/qwen-awq-miner
source .venv/bin/activate

echo "🔧 Fixing pip and vLLM installation"
echo "==================================="

# Upgrade pip first
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Show pip version
echo "Pip version: $(pip --version)"

# Set wheels-only policy
export PIP_ONLY_BINARY=":all:"
export PIP_NO_CACHE_DIR="1"

# Install setuptools and wheel
echo -e "\n📦 Installing setuptools and wheel..."
pip install --upgrade setuptools wheel

# Check Python version and architecture
echo -e "\n🐍 System info:"
python -c "
import sys
import platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.machine()}')
"

# Install PyTorch first if not present
echo -e "\n🔥 Checking PyTorch..."
python -c "import torch" 2>/dev/null || {
    echo "Installing PyTorch..."
    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
}

# Try to install vLLM
echo -e "\n⚡ Installing vLLM..."

# First check available vLLM versions
echo "Available vLLM versions:"
pip index versions vllm 2>/dev/null || echo "Cannot query versions"

# Try specific versions known to have good wheel coverage
for version in "0.2.7" "0.3.3" "0.4.3" "0.5.5"; do
    echo -e "\nTrying vLLM $version..."
    if pip install vllm==$version; then
        echo "✅ Successfully installed vLLM $version"
        break
    else
        echo "❌ vLLM $version failed (no wheel available)"
    fi
done

# If still no vLLM, try without version pin
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "\n⚠️  Trying latest vLLM..."
    pip install vllm || echo "❌ No vLLM wheel available for this platform"
fi

# Install other dependencies
echo -e "\n📦 Installing other dependencies..."
pip install transformers accelerate requests

# Final check
echo -e "\n📋 Installation summary:"
pip list | grep -E "(torch|vllm|transformers|pip)" | sort

# Test imports
echo -e "\n🧪 Testing imports:"
python << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import vllm
    print(f"✅ vLLM: {vllm.__version__}")
except Exception as e:
    print(f"❌ vLLM: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers: {e}")
EOF

# If vLLM is installed, create a working start script
if python -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
    echo -e "\n✅ Creating start script for vLLM $VLLM_VERSION..."
    
    cat > start_vllm_fixed.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

mkdir -p logs
echo $$ > vllm.pid

# Start vLLM with compatible settings
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --dtype half \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee logs/vllm.out
EOF
    chmod +x start_vllm_fixed.sh
    
    echo "✅ Start vLLM with: ./start_vllm_fixed.sh"
else
    echo -e "\n❌ vLLM could not be installed"
    echo "This platform may not have prebuilt wheels for vLLM"
    echo "Consider using:"
    echo "1. A different Python version (3.9 or 3.10)"
    echo "2. A Docker container with vLLM pre-installed"
    echo "3. The original miner.py instead of vLLM"
fi