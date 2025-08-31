#!/bin/bash
# Proper vLLM Installer - Install system deps first, then pip, then venv

echo "🚀 vLLM Installer with Proper Setup Order"
echo "========================================"

# Install system dependencies FIRST
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    build-essential \
    git \
    wget \
    curl \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install pip for Python 3.11 if not present
echo "📦 Ensuring pip is installed..."
if ! python3.11 -m pip --version 2>/dev/null; then
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
fi

# Verify pip works
python3.11 -m pip --version

# Set up directory
BASE_DIR="/data"
if [ ! -d "$BASE_DIR" ]; then
    BASE_DIR="$HOME"
fi

MINER_DIR="$BASE_DIR/qwen-awq-miner"
echo "📁 Installing to: $MINER_DIR"

mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Create virtual environment with Python 3.11
echo "🐍 Creating virtual environment..."
python3.11 -m venv venv

# Activate venv
source venv/bin/activate

# Verify we're in venv
which python
python --version

# NOW upgrade pip, setuptools, wheel INSIDE the venv
echo "📦 Upgrading pip, setuptools, wheel in venv..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch
echo "🔥 Installing PyTorch..."
pip install torch==2.7.1+cu121 torchvision==0.22.1+cu121 torchaudio==2.7.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# Install vLLM and dependencies
echo "⚡ Installing vLLM and dependencies..."
pip install vllm==0.5.0  # More stable version
pip install xformers==0.0.31
pip install transformers accelerate

# Create logs directory
mkdir -p logs

# Create vLLM startup script
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source venv/bin/activate

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0

# Write PID
echo $$ > vllm.pid

# Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12000 \
    --host 0.0.0.0 \
    --port 8000 \
    >> logs/vllm.out 2>&1
EOF
chmod +x start_vllm.sh

# Create simple test
echo "✍️ Creating test.sh..."
cat > test.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source venv/bin/activate

echo "Testing Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print('vLLM: OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
EOF
chmod +x test.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Installed at: $MINER_DIR"
echo ""
echo "To test environment:"
echo "  cd $MINER_DIR"
echo "  ./test.sh"
echo ""
echo "To start vLLM:"
echo "  ./start_vllm.sh"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/vllm.out"