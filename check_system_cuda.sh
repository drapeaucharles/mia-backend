#!/bin/bash
# Check CUDA at system level

echo "=== System CUDA Check ==="

# 1. Check if nvidia-smi works at all
echo "1. Testing nvidia-smi directly:"
/usr/bin/nvidia-smi
NVIDIA_STATUS=$?

if [ $NVIDIA_STATUS -ne 0 ]; then
    echo -e "\n✗ NVIDIA driver is broken at system level"
    echo "You MUST reboot to fix this:"
    echo "  sudo reboot"
    exit 1
fi

# 2. Test with system Python (no venv)
echo -e "\n2. Testing CUDA with system Python:"
/usr/bin/python3 -c "
import subprocess
# First check nvidia-smi
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if result.returncode == 0:
    print('✓ nvidia-smi works from Python')
else:
    print('✗ nvidia-smi fails from Python')
    print(result.stderr)
"

# 3. Check for driver/library mismatch
echo -e "\n3. CUDA libraries:"
ls -la /usr/local/cuda*/lib64/libcudart.so* 2>/dev/null || echo "No CUDA toolkit found"

# 4. Check LD_LIBRARY_PATH
echo -e "\n4. Library paths:"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 5. Simple C CUDA test
echo -e "\n5. Testing CUDA runtime directly:"
cat > /tmp/test_cuda.c << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("CUDA Devices: %d\n", deviceCount);
    return 0;
}
EOF

# Try to compile if nvcc exists
if command -v nvcc &> /dev/null; then
    nvcc /tmp/test_cuda.c -o /tmp/test_cuda 2>/dev/null
    if [ -f /tmp/test_cuda ]; then
        /tmp/test_cuda
    fi
fi
rm -f /tmp/test_cuda.c /tmp/test_cuda

echo -e "\n=== Diagnosis Complete ==="
echo ""
if [ $NVIDIA_STATUS -ne 0 ]; then
    echo "NVIDIA driver is completely broken."
    echo "The ONLY solution is to reboot:"
    echo "  sudo reboot"
else
    echo "Driver seems OK, might be a library issue."
fi