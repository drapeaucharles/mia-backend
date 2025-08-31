#!/bin/bash
# vLLM Installer - No Model Download

echo "🚀 vLLM Installer (No Download)"
echo "=============================="

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Installing Python..."
    apt-get update && apt-get install -y python3 python3-pip python3-venv
    PYTHON_CMD="python3"
fi

# Set up directory
cd /data 2>/dev/null || cd $HOME
mkdir -p qwen-awq-miner
cd qwen-awq-miner

# Create venv
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv 2>/dev/null || {
    $PYTHON_CMD -m pip install virtualenv
    $PYTHON_CMD -m virtualenv venv
}
source venv/bin/activate

# Install packages
echo "Installing packages..."
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers

# Check if model already exists locally
echo "Checking for local models..."
if [ -d "/data/models" ]; then
    MODEL_PATH=$(find /data/models -name "*.bin" -o -name "*.safetensors" | head -1 | xargs dirname)
    if [ -n "$MODEL_PATH" ]; then
        echo "✅ Found local model at: $MODEL_PATH"
    fi
fi

# Create startup script that uses local model or dummy
cat > start_vllm.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Look for local model
MODEL_PATH=""
if [ -d "/data/models" ]; then
    MODEL_PATH=$(find /data/models -name "config.json" | head -1 | xargs dirname)
fi

if [ -z "$MODEL_PATH" ]; then
    # Check common locations
    for path in /data/huggingface/hub/models--* /data/qwen* /data/Qwen*; do
        if [ -d "$path" ] && [ -f "$path/config.json" ]; then
            MODEL_PATH="$path"
            break
        fi
    done
fi

if [ -z "$MODEL_PATH" ]; then
    echo "❌ No local model found!"
    echo ""
    echo "To avoid downloading, you need a model already on disk."
    echo "Options:"
    echo "1. Copy a model to /data/models/"
    echo "2. Download manually: git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ /data/models/qwen"
    echo "3. Use a smaller test model: facebook/opt-125m"
    echo ""
    echo "Starting with small test model..."
    MODEL_PATH="facebook/opt-125m"
fi

echo "Using model: $MODEL_PATH"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048
EOF
chmod +x start_vllm.sh

# Create script to use existing model
cat > use_local_model.sh << 'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/model"
    exit 1
fi

MODEL_PATH="$1"
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ No config.json found at $MODEL_PATH"
    exit 1
fi

cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting vLLM with local model: $MODEL_PATH"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000
EOF
chmod +x use_local_model.sh

echo ""
echo "✅ Installation Complete!"
echo "========================"
echo ""
echo "To start vLLM:"
echo "  cd $(pwd)"
echo "  ./start_vllm.sh"
echo ""
echo "To use a specific local model:"
echo "  ./use_local_model.sh /path/to/model"
echo ""
echo "The server will:"
echo "1. Look for models in /data/models/"
echo "2. Check common Hugging Face cache locations"
echo "3. Use a small test model if nothing found"
echo ""
echo "To avoid ANY download, ensure you have a model at /data/models/"