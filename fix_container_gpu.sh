#!/bin/bash
# Fix GPU access in limited container environment

cd /data/qwen-awq-miner

echo "=== Fixing Container GPU Access ==="

# 1. Check what nvidia devices we have
echo "1. Available NVIDIA devices:"
ls -la /dev/nvidia* 2>/dev/null

# 2. Try to create missing device files if we have permissions
echo -e "\n2. Checking for nvidia0 device..."
if [ ! -e /dev/nvidia0 ]; then
    echo "nvidia0 device missing - container has limited GPU access"
fi

# 3. Set container-specific environment
echo -e "\n3. Setting container GPU environment..."
cat > setup_container_gpu.sh << 'EOF'
#!/bin/bash
# Container GPU setup

# Set nvidia container runtime variables
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=0

# Try to force CUDA initialization
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=1

# LD library path for container
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
EOF

source setup_container_gpu.sh

# 4. Test with these settings
echo -e "\n4. Testing with container settings..."
source venv/bin/activate
python -c "
import os
print('Environment set:')
for k, v in os.environ.items():
    if 'CUDA' in k or 'NVIDIA' in k:
        print(f'  {k}={v}')

import torch
print(f'\\nCUDA available: {torch.cuda.is_available()}')
"

# 5. Alternative: CPU-only mode
echo -e "\n5. Creating CPU-only fallback..."
cat > start_miner_cpu.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source venv/bin/activate

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""

echo "Starting miner in CPU mode (slow but should work)..."
python miner.py --cpu-only 2>&1 | tee miner_cpu.log
EOF
chmod +x start_miner_cpu.sh

echo -e "\n=== Container GPU Status ==="
echo ""
echo "Your container has limited GPU access (only nvidia-caps)."
echo "This suggests:"
echo "1. Container wasn't started with --gpus flag"
echo "2. Or GPU passthrough is limited"
echo ""
echo "Options:"
echo "1. Contact provider to enable full GPU access"
echo "2. Run in CPU mode (very slow): ./start_miner_cpu.sh"
echo "3. Try exiting SSH completely and reconnecting"