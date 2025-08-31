#!/bin/bash
# vLLM Installer using Python 3.10 (better wheel support)

echo "🚀 vLLM Installer with Python 3.10"
echo "=================================="

# Install Python 3.10 if not present
if ! command -v python3.10 &> /dev/null; then
    echo "📦 Installing Python 3.10..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi

# Ensure pip for Python 3.10
if ! python3.10 -m pip --version &> /dev/null; then
    echo "📦 Installing pip for Python 3.10..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
fi

# Set up directory
MINER_DIR="/data/qwen-awq-miner"
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Remove old .venv if it exists
if [ -d ".venv" ]; then
    echo "🗑️ Removing old .venv (Python 3.11)..."
    rm -rf .venv
fi

# Create new .venv with Python 3.10
echo "🐍 Creating .venv with Python 3.10..."
python3.10 -m venv .venv
source .venv/bin/activate

# Verify Python version
echo "✅ Python version: $(python --version)"

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

# Set wheels-only policy
export PIP_ONLY_BINARY=":all:"
export PIP_NO_CACHE_DIR="1"

# Install PyTorch for Python 3.10
echo "🔥 Installing PyTorch for Python 3.10..."
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (has wheels for Python 3.10)
echo "⚡ Installing vLLM..."
pip install vllm==0.2.7

# Install other dependencies
echo "📦 Installing dependencies..."
pip install transformers accelerate xformers requests

# Verify installation
echo -e "\n✅ Verifying installation..."
python << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import vllm
    print(f"✅ vLLM: {vllm.__version__}")
except Exception as e:
    print(f"❌ vLLM: {e}")

try:
    import xformers
    print(f"✅ xFormers: installed")
except:
    print("⚠️  xFormers: not installed")
EOF

# Create logs directory
mkdir -p logs

# Create working start script
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS

mkdir -p logs
echo $$ > vllm.pid

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12288 \
    --host 0.0.0.0 \
    --port 8000 \
    2>&1 | tee logs/vllm.out
EOF
chmod +x start_vllm.sh

# Create test script
echo "✍️ Creating test_server.sh..."
cat > test_server.sh << 'EOF'
#!/bin/bash
echo "Testing vLLM server..."
curl -s http://localhost:8000/v1/models | python -m json.tool || echo "Server not running"
EOF
chmod +x test_server.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "Using Python 3.10 (better vLLM wheel support)"
echo "Environment: /data/qwen-awq-miner/.venv"
echo ""
echo "To start vLLM:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_vllm.sh"
echo ""
echo "To test:"
echo "  ./test_server.sh"
echo ""
echo "Logs: tail -f logs/vllm.out"