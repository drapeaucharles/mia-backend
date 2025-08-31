#!/bin/bash
# Fix CUDA driver issues

echo "=== Checking CUDA Driver Status ==="

# 1. Check kernel messages for GPU errors
echo "1. Recent GPU errors in kernel log:"
dmesg | tail -50 | grep -iE "(nvidia|gpu|cuda)" || echo "No recent GPU errors in dmesg"

# 2. Check if NVIDIA driver is loaded
echo -e "\n2. NVIDIA kernel modules:"
lsmod | grep nvidia

# 3. Check nvidia-smi
echo -e "\n3. nvidia-smi status:"
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ nvidia-smi works"
    nvidia-smi
else
    echo "✗ nvidia-smi failed - driver issue"
fi

# 4. Try to reload the driver
echo -e "\n4. Attempting to reload NVIDIA driver..."
echo "This may disconnect your SSH session briefly..."

# First, kill all GPU processes
sudo pkill -9 python
sudo pkill -9 python3

# Try to unload and reload
sudo rmmod nvidia_uvm 2>/dev/null
sudo rmmod nvidia_drm 2>/dev/null  
sudo rmmod nvidia_modeset 2>/dev/null
sudo rmmod nvidia 2>/dev/null

# Reload
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_drm

# 5. Test again
echo -e "\n5. Testing after reload:"
nvidia-smi

echo -e "\n=== Fix Complete ==="
echo ""
echo "If nvidia-smi still fails, you need to reboot:"
echo "  sudo reboot"
echo ""
echo "After reboot:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_miner.sh"