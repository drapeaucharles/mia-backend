#!/bin/bash
# Fix CUDA environment error for vLLM

cd /data/qwen-awq-miner

echo "=== Diagnosing CUDA Environment ==="

# Check current CUDA setup
echo -e "\n1. CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "2. Checking nvidia-smi..."
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv

# Kill any existing processes
echo -e "\n3. Stopping all GPU processes..."
pkill -f vllm || true
pkill -f miner.py || true
sleep 3

# Clear CUDA cache
echo -e "\n4. Clearing CUDA cache..."
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true
rm -rf /tmp/cuda* 2>/dev/null || true

# Create a proper startup script
echo -e "\n5. Creating fixed startup script..."
cat > start_vllm_fixed.sh << 'EOF'
#!/bin/bash
# Fixed vLLM startup with proper CUDA settings

# Set CUDA device explicitly
export CUDA_VISIBLE_DEVICES=0

# Clear any previous CUDA errors
nvidia-smi > /dev/null 2>&1

# Activate virtual environment
source /data/qwen-awq-miner/.venv/bin/activate

# Start vLLM with explicit settings
echo "Starting vLLM server..."
python -m vllm serve \
    /data/qwen-awq-miner/Qwen2.5-32B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 12000 \
    --gpu-memory-utilization 0.95 \
    --dtype auto \
    --served-model-name "Qwen2.5-32B-Instruct-AWQ" \
    --enable-prefix-caching \
    --tool-call-parser qwen2 \
    2>&1 | tee vllm_server.log
EOF

chmod +x start_vllm_fixed.sh

# Test CUDA access
echo -e "\n6. Testing CUDA access..."
cat > test_cuda.py << 'EOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available!")
EOF

source /data/qwen-awq-miner/.venv/bin/activate
python test_cuda.py
rm test_cuda.py

echo -e "\n=== Fix Applied ==="
echo "1. Start vLLM with: ./start_vllm_fixed.sh"
echo "2. Or use the original miner: ./start_miner.sh"
echo ""
echo "The CUDA error was likely due to environment variables."
echo "The new script explicitly sets CUDA_VISIBLE_DEVICES=0"