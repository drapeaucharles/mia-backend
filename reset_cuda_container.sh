#!/bin/bash
# Reset CUDA in container environment

echo "=== Resetting CUDA in Container ==="

# 1. Kill ALL processes using GPU
echo "1. Killing all GPU processes..."
pkill -9 -f python
pkill -9 -f vllm
pkill -9 -f miner
sleep 5

# 2. Try nvidia-smi device reset
echo -e "\n2. Attempting GPU reset..."
nvidia-smi --gpu-reset -i 0 2>/dev/null || echo "GPU reset not available in container"

# 3. Clear all caches and temp files
echo -e "\n3. Clearing all caches..."
rm -rf ~/.cache/torch*
rm -rf ~/.cache/huggingface*
rm -rf /tmp/cuda*
rm -rf /tmp/torch*
rm -rf /var/tmp/*python*

# 4. Unset all CUDA variables
echo -e "\n4. Clearing environment..."
unset CUDA_VISIBLE_DEVICES
unset CUDA_DEVICE_ORDER
unset CUDA_LAUNCH_BLOCKING

# 5. Test basic CUDA access
echo -e "\n5. Testing CUDA access..."
export CUDA_VISIBLE_DEVICES=0
python3 -c "import os; os.environ['CUDA_VISIBLE_DEVICES']='0'"

# Check nvidia-smi
nvidia-smi

echo -e "\n=== Reset Complete ==="
echo ""
echo "Since you're in a container, you may need to:"
echo "1. Contact your VPS provider to restart the container"
echo "2. Or exit the container and restart it from the host"
echo ""
echo "The CUDA 'unknown error' in containers often requires container restart."