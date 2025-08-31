#!/bin/bash
# Fix vLLM installation in existing environment

echo "🔧 Fixing vLLM Installation"
echo "=========================="

# Navigate to miner directory
if [ -d "/data/command-r-miner" ]; then
    cd /data/command-r-miner
elif [ -d "$HOME/command-r-miner" ]; then
    cd $HOME/command-r-miner
else
    echo "❌ Miner directory not found!"
    exit 1
fi

echo "📂 Working in: $(pwd)"

# Stop miner if running
./stop.sh 2>/dev/null || true

# Remove old venv
echo "🗑️ Removing old venv..."
rm -rf venv

# Create fresh venv
echo "🐍 Creating fresh venv..."
python3 -m venv vllm_env
source vllm_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Clean pip config
echo "🧹 Cleaning pip configuration..."
pip config unset global.index-url || true
pip config unset global.extra-index-url || true

# Force official PyPI
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL

# Install PyTorch for CUDA first
echo "📦 Installing PyTorch..."
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.3.0

# Install vLLM from official PyPI
echo "📦 Installing vLLM from official PyPI..."
pip install --index-url https://pypi.org/simple vllm

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install transformers sentencepiece flask waitress requests psutil

# Check installation
echo ""
echo "✅ Checking installation..."
pip install pipdeptree
pipdeptree -p vllm -fl

# Update start script to use new venv
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source vllm_env/bin/activate
nohup python miner.py > miner.log 2>&1 &
echo $! > miner.pid
echo "Command-R miner started. PID: $(cat miner.pid)"
echo "Logs: tail -f miner.log"
EOF
chmod +x start.sh

echo ""
echo "✅ vLLM installation fixed!"
echo ""
echo "🚀 Start miner with: ./start.sh"
echo "📊 Check logs with: tail -f miner.log"