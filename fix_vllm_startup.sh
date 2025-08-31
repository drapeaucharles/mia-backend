#!/bin/bash
# Fix vLLM startup without assuming virtualenv path

cd /data/qwen-awq-miner

echo "=== Fixing vLLM Startup ==="

# Find the correct Python
echo "1. Finding Python environment..."
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
    ACTIVATE_CMD="source .venv/bin/activate"
elif [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
    ACTIVATE_CMD="source venv/bin/activate"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    ACTIVATE_CMD=""
else
    PYTHON_CMD="python"
    ACTIVATE_CMD=""
fi

echo "Using Python: $PYTHON_CMD"

# Test Python and vLLM
echo -e "\n2. Testing Python environment..."
$PYTHON_CMD -c "import sys; print(f'Python: {sys.version}')"
$PYTHON_CMD -c "import vllm; print('vLLM is installed')" 2>/dev/null || echo "vLLM not found in this Python"

# Create a working vLLM startup script
echo -e "\n3. Creating vLLM startup script..."
cat > start_vllm_working.sh << EOF
#!/bin/bash
# vLLM startup with proper environment

cd /data/qwen-awq-miner

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Clear any CUDA errors
nvidia-smi > /dev/null 2>&1

# Activate environment if exists
$ACTIVATE_CMD

# Start vLLM
echo "Starting vLLM server..."
$PYTHON_CMD -m vllm serve \\
    /data/qwen-awq-miner/Qwen2.5-32B-Instruct-AWQ \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --max-model-len 12000 \\
    --gpu-memory-utilization 0.95 \\
    --dtype auto \\
    --served-model-name "Qwen2.5-32B-Instruct-AWQ" \\
    --enable-prefix-caching \\
    --backend xformers \\
    2>&1 | tee vllm_server.log
EOF

chmod +x start_vllm_working.sh

# Also create a simpler version for the original miner
echo -e "\n4. Updating miner startup..."
cat > start_miner_fixed.sh << EOF
#!/bin/bash
# Fixed miner startup

cd /data/qwen-awq-miner

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Clear CUDA cache
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

# Start the original miner
echo "Starting miner..."
$PYTHON_CMD miner.py 2>&1 | tee miner.log
EOF

chmod +x start_miner_fixed.sh

echo -e "\n=== Setup Complete ==="
echo "Found Python at: $PYTHON_CMD"
echo ""
echo "To start vLLM: ./start_vllm_working.sh"
echo "To start original miner: ./start_miner_fixed.sh"
echo ""
echo "Both scripts now set CUDA_VISIBLE_DEVICES=0 to avoid the CUDA error."