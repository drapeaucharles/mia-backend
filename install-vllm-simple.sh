#!/bin/bash
# Simple vLLM installer for containers

echo "🚀 Simple vLLM Installer"
echo "======================="

# Basic setup
cd /data
mkdir -p qwen-awq-miner
cd qwen-awq-miner

# Check Python
if command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "Installing Python..."
    apt-get update && apt-get install -y python3 python3-pip python3-venv
    PYTHON=python3
fi

echo "Using Python: $PYTHON"

# Create venv
echo "Creating virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Set pip to wheels-only
export PIP_ONLY_BINARY=":all:"
export PIP_NO_CACHE_DIR=1

# Install PyTorch
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
echo "Installing vLLM..."
pip install vllm==0.5.5  # More stable version

# Create simple start script
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

export HF_HOME=/data/cache/hf
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs

echo "Starting vLLM..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --max-model-len 4096 \
    2>&1 | tee logs/vllm.log
EOF
chmod +x start_vllm.sh

# Test script
cat > test.sh << 'EOF'
#!/bin/bash
echo "Testing server..."
curl -s http://localhost:8000/v1/models | python -m json.tool
EOF
chmod +x test.sh

echo ""
echo "✅ Installation complete!"
echo ""
echo "Start vLLM: ./start_vllm.sh"
echo "Test: ./test.sh"
echo "Logs: tail -f logs/vllm.log"