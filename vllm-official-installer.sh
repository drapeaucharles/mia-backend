#!/bin/bash
# vLLM Official Installer - PyPI packages only

echo "🚀 vLLM Official Installer"
echo "========================="

# Ensure Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "📦 Installing Python..."
    apt-get update
    apt-get install -y python3 python3-pip python3-venv
    PYTHON_CMD="python3"
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

# Ensure pip
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "📦 Installing pip..."
    apt-get install -y python3-pip || {
        curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
    }
fi

# Setup directory
BASE_DIR="/data"
[ ! -d "$BASE_DIR" ] && BASE_DIR="$HOME"
MINER_DIR="$BASE_DIR/qwen-awq-miner"

mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Create virtual environment
echo "🐍 Creating virtual environment..."
$PYTHON_CMD -m venv venv || {
    $PYTHON_CMD -m pip install virtualenv
    $PYTHON_CMD -m virtualenv venv
}
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch from official index
echo "🔥 Installing PyTorch from official wheels..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM from PyPI
echo "⚡ Installing vLLM from PyPI..."
pip install vllm==0.2.7

# Install dependencies from PyPI
echo "📦 Installing dependencies from PyPI..."
pip install transformers accelerate sentencepiece protobuf

# Create minimal test
cat > test_install.py << 'EOF'
#!/usr/bin/env python3
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch error: {e}")

try:
    import vllm
    print("vLLM: Installed successfully")
except Exception as e:
    print(f"vLLM error: {e}")
EOF
chmod +x test_install.py

# Create start script
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting vLLM server..."
echo "Note: Model will be downloaded from Hugging Face on first run"
echo "Model: Qwen/Qwen2.5-32B-Instruct-AWQ"

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12000 \
    --host 0.0.0.0 \
    --port 8000
EOF
chmod +x start_vllm.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "All packages installed from official sources:"
echo "  - PyTorch: Official PyTorch index"
echo "  - vLLM: PyPI (official Python package index)"
echo "  - Dependencies: PyPI"
echo ""
echo "Test installation:"
echo "  cd $MINER_DIR"
echo "  python test_install.py"
echo ""
echo "Start vLLM:"
echo "  ./start_vllm.sh"
echo ""
echo "⚠️  Note: The model will download from Hugging Face on first run"
echo "   This is unavoidable - vLLM needs the model files to run"