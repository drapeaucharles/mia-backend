#!/bin/bash
# Diagnose CUDA initialization issue

cd /data/qwen-awq-miner

echo "=== CUDA Diagnostic ==="

# 1. Check GPU status
echo -e "\n1. GPU Status:"
nvidia-smi

# 2. Check if any process is using GPU
echo -e "\n2. GPU Processes:"
nvidia-smi | grep -A 20 "Processes:" || echo "No GPU processes found"

# 3. Test CUDA in the venv
echo -e "\n3. Testing CUDA in venv:"
./venv/bin/python -c "
import os
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"not set\")}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    
    # Try to allocate memory
    try:
        x = torch.zeros(1).cuda()
        print('✓ Can allocate CUDA memory')
    except Exception as e:
        print(f'✗ Cannot allocate CUDA memory: {e}')
"

# 4. Check vLLM imports
echo -e "\n4. Testing vLLM imports:"
./venv/bin/python -c "
try:
    import vllm
    print('✓ vLLM imported successfully')
    from vllm import LLM
    print('✓ Can import LLM class')
except Exception as e:
    print(f'✗ vLLM import error: {e}')
"

# 5. Try minimal CUDA initialization
echo -e "\n5. Testing minimal CUDA init:"
cat > test_cuda_init.py << 'EOF'
import os
import torch

# Set CUDA device before any CUDA operations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Force CUDA initialization
if torch.cuda.is_available():
    torch.cuda.init()
    print("✓ CUDA initialized successfully")
    
    # Try creating a tensor
    device = torch.device('cuda:0')
    x = torch.tensor([1.0]).to(device)
    print(f"✓ Created tensor on {device}")
else:
    print("✗ CUDA not available")
EOF

./venv/bin/python test_cuda_init.py
rm test_cuda_init.py

# 6. Check for zombie processes
echo -e "\n6. Checking for zombie CUDA processes:"
ps aux | grep -E "(defunct|python.*miner)" | grep -v grep || echo "No zombie processes found"

echo -e "\n=== Diagnostic Complete ==="
echo ""
echo "Common fixes:"
echo "1. Kill all Python processes: pkill -9 python"
echo "2. Reset GPU: nvidia-smi --gpu-reset"
echo "3. Clear CUDA cache: rm -rf ~/.cache/torch*"
echo "4. Reboot if needed: sudo reboot"