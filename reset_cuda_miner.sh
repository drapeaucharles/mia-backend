#!/bin/bash
# Reset CUDA and get miner working

cd /data/qwen-awq-miner

echo "=== Resetting CUDA and Miner ==="

# 1. Kill ALL Python processes
echo "1. Killing all Python processes..."
pkill -9 python
pkill -9 python3
sleep 3

# 2. Clear CUDA cache
echo "2. Clearing CUDA cache..."
rm -rf ~/.cache/torch*
rm -rf ~/.cache/huggingface/
rm -rf /tmp/cuda*

# 3. Reset GPU if possible
echo "3. Checking GPU..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo "GPU is responsive"
else
    echo "GPU may need system reboot"
fi

# 4. Test CUDA with simple script
echo -e "\n4. Testing basic CUDA..."
cat > test_basic.py << 'EOF'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Simple test
    x = torch.randn(10).cuda()
    print("✓ Basic CUDA works")
else:
    print("✗ CUDA not available")
EOF

./venv/bin/python test_basic.py
rm test_basic.py

# 5. Start miner with minimal settings
echo -e "\n5. Starting miner with basic settings..."
cat > start_basic_miner.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

# Clear environment
unset CUDA_VISIBLE_DEVICES

# Set minimal CUDA settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Start with explicit settings
./venv/bin/python miner.py
EOF

chmod +x start_basic_miner.sh

echo -e "\n=== Reset Complete ==="
echo ""
echo "Try starting the miner with:"
echo "  ./start_basic_miner.sh"
echo ""
echo "If CUDA still fails, you may need to:"
echo "  1. Check dmesg for GPU errors: dmesg | grep -i nvidia"
echo "  2. Restart GPU driver: sudo rmmod nvidia && sudo modprobe nvidia"
echo "  3. Or simply reboot: sudo reboot"