#!/bin/bash
# Qwen2.5-7B-Instruct-AWQ Strict Installer - Wheels Only to Prevent Build Errors
# Production grade with vLLM and strict requirements

set -euo pipefail
trap 'echo "[!] Failed at line $LINENO" >&2' ERR

# Logging
exec > >(tee -a setup.log) 2>&1

echo "=== Qwen2.5-7B-Instruct-AWQ Strict Installer (Wheels Only) ==="
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

# Upgrade pip again with clean config - WHEELS ONLY
pip install --no-cache-dir --upgrade --only-binary :all: pip

# Detect CUDA & install PyTorch (NO --only-binary for PyTorch index)
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
    # NO --only-binary for PyTorch official index
    pip install --index-url "$TORCH_INDEX" torch --upgrade
else
    echo "No CUDA detected, skipping PyTorch GPU install"
fi

# Preinstall core dependencies as wheels only
echo "=== Preinstalling core dependencies (wheels only) ==="
pip install --index-url https://pypi.org/simple --only-binary :all: \
    numpy \
    scipy \
    psutil \
    pydantic \
    fastapi \
    uvicorn \
    httpx \
    aiohttp \
    transformers \
    tokenizers \
    safetensors \
    accelerate \
    sentencepiece \
    protobuf \
    grpcio \
    typing-extensions \
    packaging \
    tqdm \
    filelock \
    fsspec \
    huggingface-hub

# Install vLLM from official PyPI - WHEELS ONLY
echo "=== Installing vLLM from official PyPI (wheels only) ==="
pip install --index-url https://pypi.org/simple --only-binary :all: vllm

# Install additional required packages - WHEELS ONLY
echo "=== Installing additional dependencies (wheels only) ==="
pip install --index-url https://pypi.org/simple --only-binary :all: transformers accelerate sentencepiece

# Create dependency report
echo "=== Creating dependency report ==="
pip install --only-binary :all: pipdeptree
pipdeptree -p vllm -fl || true
pip freeze | sort > requirements.lock

# Allowlist sanity check
echo "=== Running allowlist sanity check ==="
ALLOWLIST_PATTERN='^(accelerate|anyio|fastapi|filelock|h11|httpcore|httpx|huggingface-hub|jinja2|markdown-it-py|mdurl|numpy|packaging|pydantic|pydantic-core|pip|pipdeptree|prometheus-fastapi-instrumentator|python-dotenv|requests|rich|safetensors|scipy|sentencepiece|setuptools|sniffio|starlette|sympy|tiktoken|tokenizers|torch(|vision|audio)?|tqdm|transformers|triton|typing-extensions|uvicorn|uvloop|vllm(-flash-attn)?|watchfiles|websockets|nvidia-(cublas|cudnn|cufft|curand|cusolver|cusparse|nccl|nvjitlink|nvtx)-cu(11|12)|pkg_resources|pkgutil_resolve_name)$'

# Additional allowed packages for comprehensive coverage
ADDITIONAL_ALLOWED='^(wheel|certifi|charset-normalizer|idna|urllib3|markupsafe|click|nvidia-ml-py|psutil|protobuf|grpcio|fsspec|regex|pyyaml|aiohttp|aiosignal|attrs|frozenlist|multidict|yarl|importlib-metadata|zipp|annotated-types|exceptiongroup|mpmath|networkx|jmespath|msgpack|py-cpuinfo|pyarrow|xxhash|ray|cloudpickle|distro|openai|gputil|einops|xformers|ninja|outlines|jsonschema|lm-format-enforcer|interegular|lark|nest-asyncio|numba|llvmlite|prometheus-client|rpds-py|referencing|jsonschema-specifications|jiter|threadpoolctl|joblib|scikit-learn|orjson|python-multipart|sse-starlette|humanfriendly|coloredlogs|inflect|lm-eval|sqlitedict|word2number|more-itertools|dill|multiprocess|datasets|pandas|pytz|tzdata|python-dateutil|six|tqdm-multiprocess|zstandard|responses|tomli|pytest|pluggy|iniconfig|exceptiongroup|hf-transfer)$'

# Check installed packages
INSTALLED_PACKAGES=$(pip list --format=freeze | cut -d'=' -f1 | tr '[:upper:]' '[:lower:]')

echo "=== Checking for packages not in allowlist ==="
UNEXPECTED_COUNT=0
declare -A UNEXPECTED_PACKAGES

for pkg in $INSTALLED_PACKAGES; do
    pkg_lower=$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    
    # Check against both patterns
    if [[ ! "$pkg_lower" =~ $ALLOWLIST_PATTERN ]] && [[ ! "$pkg_lower" =~ $ADDITIONAL_ALLOWED ]]; then
        if [[ -z "${UNEXPECTED_PACKAGES[$pkg_lower]}" ]]; then
            echo "[!] Unexpected package: $pkg_lower"
            UNEXPECTED_PACKAGES[$pkg_lower]=1
            ((UNEXPECTED_COUNT++))
        fi
    fi
done

echo ""
if [ $UNEXPECTED_COUNT -eq 0 ]; then
    echo "✓ All packages match allowlist"
else
    echo "[!] $UNEXPECTED_COUNT unexpected packages found above"
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