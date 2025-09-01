#!/bin/bash
# Fix CUDA version mismatch

cd /data/qwen-awq-miner

echo "=== Fixing CUDA Version Mismatch ==="

# 1. Check current CUDA version
echo "1. System CUDA version:"
nvcc --version 2>/dev/null | grep "release" || echo "CUDA toolkit not in PATH"
ls -la /usr/local/cuda | grep -E "cuda-[0-9]"

# 2. Reinstall PyTorch with correct CUDA version
echo -e "\n2. Reinstalling PyTorch for CUDA 12.0..."
source venv/bin/activate

# Uninstall current PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch for CUDA 12.1 (closest to 12.0)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

# 3. Test after reinstall
echo -e "\n3. Testing PyTorch with CUDA 12.1:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    x = torch.zeros(1).cuda()
    print('✓ CUDA working!')
"

# 4. Reinstall vLLM compatible with this PyTorch
echo -e "\n4. Reinstalling vLLM..."
pip uninstall -y vllm
pip install vllm==0.5.0

echo -e "\n=== Fix Complete ==="
echo "Now try starting the miner:"
echo "  cd /data/qwen-awq-miner"
echo "  source venv/bin/activate"
echo "  python miner.py"