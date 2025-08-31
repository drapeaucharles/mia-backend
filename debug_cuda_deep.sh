#!/bin/bash
# Deep CUDA debugging

cd /data/qwen-awq-miner

echo "=== Deep CUDA Debug ==="

# 1. Check container GPU access
echo "1. Container GPU access:"
ls -la /dev/nvidia* 2>/dev/null || echo "No nvidia devices found"

# 2. Check processes
echo -e "\n2. Check for stuck processes:"
ps aux | grep -E "(python|cuda|gpu)" | grep -v grep

# 3. Test with minimal Python
echo -e "\n3. Test with fresh Python:"
/usr/bin/python3 -c "
import os
print(f'Python: {os.sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}')
except:
    print('No PyTorch in system Python')
"

# 4. Check dmesg for errors
echo -e "\n4. Recent kernel messages:"
dmesg | tail -20 | grep -iE "(nvidia|gpu|cuda|error)" || echo "No GPU errors in dmesg"

# 5. Try loading with specific device
echo -e "\n5. Testing specific CUDA device:"
source venv/bin/activate
python -c "
import os
# Clear all CUDA env vars
for key in list(os.environ.keys()):
    if 'CUDA' in key:
        del os.environ[key]

# Set explicitly
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
print(f'After clearing env: {torch.cuda.is_available()}')

# Try direct CUDA init
import ctypes
try:
    cuda = ctypes.CDLL('libcudart.so')
    print('✓ Can load CUDA library')
except:
    print('✗ Cannot load CUDA library')
"

echo -e "\n=== Debug Complete ==="
echo ""
echo "Possible issues:"
echo "1. Container doesn't have GPU access - check with provider"
echo "2. CUDA driver/runtime mismatch"
echo "3. Need to exit SSH completely and reconnect"
echo ""
echo "Try: Exit SSH, wait 30 seconds, reconnect"