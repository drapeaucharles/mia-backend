#!/bin/bash
# Check for existing venv and use it for deployment

echo "🔍 Checking for virtual environment..."
echo "====================================="

# Common venv locations
VENV_PATHS=(
    "venv"
    ".venv"
    "env"
    ".env"
    "mia-env"
    "miner-env"
)

FOUND_VENV=""

# Check each possible venv location
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -d "$venv_path" ] && [ -f "$venv_path/bin/activate" ]; then
        echo "✓ Found virtual environment: $venv_path"
        FOUND_VENV="$venv_path"
        break
    fi
done

if [ -z "$FOUND_VENV" ]; then
    echo "❌ No virtual environment found"
    echo ""
    echo "Creating new venv..."
    python3 -m venv venv
    FOUND_VENV="venv"
fi

echo ""
echo "📦 Activating virtual environment: $FOUND_VENV"
source "$FOUND_VENV/bin/activate"

echo ""
echo "Python location: $(which python3)"
echo "Pip location: $(which pip3)"

# Check if torch is installed in venv
echo ""
echo "🔍 Checking for PyTorch in venv..."
if python3 -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch found in venv!"
    python3 -c "import torch; print(f'Version: {torch.__version__}')"
else
    echo "❌ PyTorch not found, installing..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip3 install vllm flask waitress
fi

# Update the miner startup script to use venv
echo ""
echo "📝 Creating startup script with venv..."

cat > start_miner_with_venv.sh << EOF
#!/bin/bash
# Start miner with virtual environment

# Kill existing processes
pkill -f "mia_miner" || true
fuser -k 8000/tcp 2>/dev/null || true

# Activate venv and start miner
source $FOUND_VENV/bin/activate
nohup python3 mia_miner_production.py > miner.log 2>&1 &
echo "Miner started with PID: \$!"
EOF

chmod +x start_miner_with_venv.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the miner with venv:"
echo "./start_miner_with_venv.sh"
echo ""
echo "Current venv packages:"
pip3 list | grep -E "(torch|vllm|flask|waitress)"