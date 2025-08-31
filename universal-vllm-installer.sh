#!/bin/bash
# Universal vLLM Installer - Works with any Python version

echo "🚀 Universal vLLM Installer"
echo "=========================="

# First, ensure we have Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "📦 Installing Python..."
    apt-get update
    apt-get install -y python3 python3-pip python3-venv || {
        echo "❌ Failed to install Python. Please install Python manually."
        exit 1
    }
fi

# Find Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ No Python found!"
    exit 1
fi

echo "🐍 Found Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Ensure pip is installed
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "📦 Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD || {
        # Try apt if curl method fails
        apt-get update && apt-get install -y python3-pip
    }
fi

# Ensure venv module
if ! $PYTHON_CMD -m venv --help &> /dev/null; then
    echo "📦 Installing venv module..."
    apt-get install -y python3-venv || $PYTHON_CMD -m pip install virtualenv
fi

# Set up directory
if [ -d "/data" ]; then
    BASE_DIR="/data"
else
    BASE_DIR="$HOME"
fi

MINER_DIR="$BASE_DIR/qwen-awq-miner"
echo "📁 Installing to: $MINER_DIR"

mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Create virtual environment
echo "🐍 Creating virtual environment..."
if $PYTHON_CMD -m venv venv; then
    source venv/bin/activate
else
    # Fallback to virtualenv
    $PYTHON_CMD -m pip install virtualenv
    $PYTHON_CMD -m virtualenv venv
    source venv/bin/activate
fi

# Verify we're in venv
echo "📍 Python location: $(which python)"
echo "📍 Pip location: $(which pip)"

# Upgrade pip inside venv
echo "📦 Upgrading pip, setuptools, wheel..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (try different CUDA versions)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
pip install torch torchvision torchaudio

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install vLLM
echo "⚡ Installing vLLM..."
pip install vllm

# Install other dependencies
echo "📦 Installing additional dependencies..."
pip install transformers accelerate xformers

# Create startup script
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

echo "Starting vLLM..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --dtype auto \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee logs/vllm.log
EOF
chmod +x start_vllm.sh

# Create test script
echo "✍️ Creating test_env.sh..."
cat > test_env.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

echo "=== Environment Test ==="
python --version
echo ""
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
    print('vLLM: INSTALLED')
except:
    print('vLLM: NOT INSTALLED')
"
EOF
chmod +x test_env.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Installed at: $MINER_DIR"
echo ""
echo "Test environment:"
echo "  cd $MINER_DIR"
echo "  ./test_env.sh"
echo ""
echo "Start vLLM:"
echo "  ./start_vllm.sh"