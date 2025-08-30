#!/bin/bash
# Qwen2.5-7B-Instruct-AWQ Strict Installer - AWQ Only, No Fallbacks
# Production grade with vLLM and strict requirements

set -euo pipefail
trap 'echo "[!] Failed at line $LINENO" >&2' ERR

# Logging
exec > >(tee -a setup.log) 2>&1

echo "=== Qwen2.5-7B-Instruct-AWQ Strict Installer ==="
echo "Date: $(date)"
echo "Environment Info:"
uname -a
python3 --version

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi
else
    echo "No GPU detected"
fi

# Install system prerequisites
echo "=== Installing system prerequisites ==="
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip build-essential git curl jq

echo "✓ System dependencies installed"

# Environment setup
[ -d "/data" ] && BASE_DIR="/data" || BASE_DIR="$HOME"
MINER_DIR="$BASE_DIR/qwen-awq-miner"

echo "=== Setting up in: $MINER_DIR ==="

# Stop existing miners
pkill -f "miner.py" || true
pkill -f "vllm" || true
sleep 2

# Create fresh directory
rm -rf "$MINER_DIR"
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Create fresh venv
echo "=== Creating fresh virtual environment ==="
python3 -m venv vllm_env
source vllm_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Clean PyPI configuration
echo "=== Configuring pip for official PyPI only ==="
pip config unset global.index-url 2>/dev/null || true
pip config unset global.extra-index-url 2>/dev/null || true

# Set environment vars for PyPI only
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Upgrade pip again with clean config
pip install --no-cache-dir --upgrade pip

# Detect CUDA & install PyTorch
echo "=== Detecting CUDA and installing PyTorch ==="
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    echo "Detected CUDA version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12.4"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [[ "$CUDA_VERSION" == "12"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif [[ "$CUDA_VERSION" == "11"* ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    else
        echo "Unknown CUDA version, defaulting to cu118"
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    
    echo "Installing PyTorch from: $TORCH_INDEX"
    pip install --index-url "$TORCH_INDEX" torch --upgrade
else
    echo "No CUDA detected, skipping PyTorch GPU install"
fi

# Install vLLM from official PyPI
echo "=== Installing vLLM from official PyPI ==="
pip install --index-url https://pypi.org/simple vllm

# Install additional required packages
echo "=== Installing additional dependencies ==="
pip install --index-url https://pypi.org/simple transformers accelerate sentencepiece

# Create dependency report
echo "=== Creating dependency report ==="
pip install pipdeptree
pipdeptree -p vllm -fl || true
pip freeze | sort > requirements.lock

# Allowlist sanity check
echo "=== Running allowlist sanity check ==="
ALLOWLIST=(
    "vllm" "pip" "setuptools" "wheel" "pipdeptree" "torch" "torchvision" "torchaudio"
    "uvicorn" "fastapi" "pydantic" "httpx" "starlette" "numpy" "tqdm"
    "safetensors" "transformers" "accelerate" "tokenizers" "huggingface-hub"
    "filelock" "fsspec" "packaging" "pyyaml" "regex" "requests" "typing-extensions"
    "certifi" "charset-normalizer" "idna" "urllib3" "markupsafe" "jinja2"
    "click" "h11" "anyio" "sniffio" "nvidia-ml-py" "psutil" "sentencepiece"
    "protobuf" "triton" "xformers" "einops" "ninja" "outlines" "jsonschema"
    "lm-format-enforcer" "cloudpickle" "msgpack" "py-cpuinfo" "prometheus-client"
    "vllm-nccl-cu12" "interegular" "lark" "nest-asyncio" "numba" "llvmlite"
    "ray" "aiosignal" "frozenlist" "multidict" "yarl" "aiohttp"
    "scipy" "threadpoolctl" "joblib" "scikit-learn" "attrs" "rpds-py"
    "referencing" "jsonschema-specifications" "importlib-metadata" "zipp"
    "pydantic-core" "annotated-types" "httpcore" "exceptiongroup" "outcome"
    "trio" "httpx-sse" "distro" "openai" "tiktoken" "jiter" "grpcio"
)

# Check installed packages
INSTALLED_PACKAGES=$(pip list --format=freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]')

echo "=== Checking for packages not in allowlist ==="
REVIEW_REQUIRED=false
for pkg in $INSTALLED_PACKAGES; do
    pkg_lower=$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    found=false
    for allowed in "${ALLOWLIST[@]}"; do
        allowed_lower=$(echo "$allowed" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        if [[ "$pkg_lower" == "$allowed_lower" ]]; then
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "[REVIEW REQUIRED] Package not in allowlist: $pkg"
        REVIEW_REQUIRED=true
    fi
done

if [ "$REVIEW_REQUIRED" = false ]; then
    echo "✓ All packages are in the allowlist"
fi

# Create simple AWQ test script
cat > test_awq.py << 'EOF'
#!/usr/bin/env python3
"""Test AWQ model loading"""
import sys
try:
    from vllm import LLM
    print("Testing AWQ model load...")
    model = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        trust_remote_code=True
    )
    print("✓ AWQ model loaded successfully!")
except Exception as e:
    print(f"✗ Failed to load AWQ model: {e}")
    sys.exit(1)
EOF

chmod +x test_awq.py

# Create serve script with AWQ command
cat > serve_awq.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source vllm_env/bin/activate

echo "Starting vLLM with Qwen AWQ..."
echo "This will serve the model with tool calling support"
echo ""

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --max-model-len 65536 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
EOF
chmod +x serve_awq.sh

echo ""
echo "=== Installation Complete ==="
echo "Model: Qwen/Qwen2.5-7B-Instruct-AWQ (AWQ ONLY - No fallbacks)"
echo "Location: $MINER_DIR"
echo ""
echo "To serve the model with tool calling:"
echo ""
echo "cd $MINER_DIR"
echo "./serve_awq.sh"
echo ""
echo "Or run directly:"
echo ""
echo "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \\"
echo "    --max-model-len 65536 \\"
echo "    --enable-auto-tool-choice \\"
echo "    --tool-call-parser hermes"
echo ""