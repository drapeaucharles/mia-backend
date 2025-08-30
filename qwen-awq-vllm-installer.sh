#!/bin/bash
# Qwen2.5-7B-Instruct-AWQ with vLLM - Production Installer
# Strict adherence to official PyPI and security best practices

set -euo pipefail
trap 'echo "[!] Failed at line $LINENO" >&2' ERR

# Logging
exec > >(tee -a setup.log) 2>&1

echo "=== Qwen2.5-7B-Instruct-AWQ vLLM Installer ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"

# Show GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU Information ==="
    nvidia-smi
else
    echo "No GPU detected (CPU mode)"
fi

# Environment setup
[ -d "/data" ] && BASE_DIR="/data" || BASE_DIR="$HOME"
MINER_DIR="$BASE_DIR/qwen-awq-miner"

echo "=== Setting up in: $MINER_DIR ==="

# Stop existing miners
pkill -f "miner.py" || true
sleep 2

# Create fresh directory
rm -rf "$MINER_DIR"
mkdir -p "$MINER_DIR"
cd "$MINER_DIR"

# Fresh venv
echo "=== Creating fresh virtual environment ==="
python3 -m venv vllm_env
source vllm_env/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Force official PyPI only & no cache
echo "=== Configuring pip for official PyPI only ==="
pip config unset global.index-url 2>/dev/null || true
pip config unset global.extra-index-url 2>/dev/null || true
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
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
        echo "Unknown CUDA version, skipping PyTorch preinstall"
        TORCH_INDEX=""
    fi
    
    if [ -n "$TORCH_INDEX" ]; then
        echo "Installing PyTorch from: $TORCH_INDEX"
        pip install --index-url "$TORCH_INDEX" torch --upgrade
    fi
else
    echo "No CUDA detected, skipping PyTorch preinstall"
fi

# Install vLLM from official PyPI
echo "=== Installing vLLM from official PyPI ==="
pip install --index-url https://pypi.org/simple vllm

# Install additional required packages
echo "=== Installing additional dependencies ==="
pip install --index-url https://pypi.org/simple transformers accelerate

# Sanity & lock
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
    "protobuf" "absl-py" "grpcio" "tensorboard" "werkzeug" "tensorboard-data-server"
    "sympy" "networkx" "mpmath" "triton" "nvidia-cuda-runtime-cu12" "nvidia-cuda-nvrtc-cu12"
    "nvidia-cuda-cupti-cu12" "nvidia-cudnn-cu12" "nvidia-cublas-cu12" "nvidia-cufft-cu12"
    "nvidia-curand-cu12" "nvidia-cusolver-cu12" "nvidia-cusparse-cu12" "nvidia-nccl-cu12"
    "nvidia-nvtx-cu12" "xformers" "einops" "flash-attn" "ninja" "outlines" "jsonschema"
    "lm-format-enforcer" "cloudpickle" "msgpack" "py-cpuinfo" "pyarrow" "xxhash"
    "pyasn1" "pyasn1-modules" "rsa" "cachetools" "google-auth" "oauthlib"
    "requests-oauthlib" "google-auth-oauthlib" "scipy" "threadpoolctl" "joblib"
    "scikit-learn" "prometheus-client" "prometheus-fastapi-instrumentator" "wrapt"
    "deprecated" "importlib-metadata" "zipp" "attrs" "jsonschema-specifications"
    "rpds-py" "referencing" "annotated-types" "pydantic-core" "email-validator"
    "dnspython" "httpcore" "httptools" "python-dotenv" "watchfiles" "websockets"
    "uvloop" "aiosignal" "frozenlist" "multidict" "yarl" "aiohttp" "async-timeout"
    "asyncio" "nest-asyncio" "numba" "llvmlite" "openai" "distro" "tiktoken"
    "ray" "aiorwlock" "opencensus" "opencensus-context" "google-api-core"
    "googleapis-common-protos" "smart-open" "pytz" "pandas" "tzdata" "six"
    "python-dateutil" "py" "colorama" "iniconfig" "pluggy" "exceptiongroup"
    "tomli" "pytest" "virtualenv" "platformdirs" "distlib" "pyproject-api"
    "hatchling" "pathspec" "editables" "hatch-vcs" "hatch-fancy-pypi-readme"
    "babel" "pynvml" "vllm-nccl-cu12" "torch-tensorrt" "jiter" "orjson"
    "python-multipart" "sse-starlette" "gradio" "aiofiles" "altair" "ffmpy"
    "gradio-client" "markdown-it-py" "mdit-py-plugins" "mdurl" "pygments"
    "rich" "typer" "shellingham" "tomlkit" "pydub" "ruff" "pyparsing"
    "cycler" "fonttools" "kiwisolver" "matplotlib" "pillow" "contourpy"
    "importlib-resources" "seaborn" "cmake" "lit" "pyaml" "inflect"
    "unidecode" "omegaconf" "hydra-core" "antlr4-python3-runtime" "humanfriendly"
    "coloredlogs" "flatbuffers" "onnx" "onnxruntime" "librosa" "audioread"
    "lazy-loader" "msgpack-numpy" "numcodecs" "zarr" "asciitree" "entrypoints"
    "fasteners" "monotonic" "importlib-resources" "lark" "interegular"
    "greenlet" "sqlalchemy" "alembic" "mako" "blinker" "itsdangerous"
    "flask" "waitress" "flask-cors" "flask-sqlalchemy" "flask-migrate"
    "email-validator" "flask-login" "flask-mail" "flask-wtf" "wtforms"
    "passlib" "argon2-cffi" "bcrypt" "argon2-cffi-bindings" "cffi" "pycparser"
)

# Get installed packages
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

# Create Qwen AWQ miner
echo "=== Creating Qwen AWQ miner ==="
cat > miner.py << 'EOF'
#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct-AWQ Miner with Tool Calling
Production grade with vLLM
"""
import os
import sys
import json
import time
import logging
import requests
import threading
import re
import socket
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qwen-awq-miner')

# Global model
model = None
model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"

def load_model():
    """Load Qwen AWQ model"""
    global model
    
    logger.info(f"Loading {model_name}...")
    try:
        model = LLM(
            model=model_name,
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=0.95,
            max_model_len=8192,
            trust_remote_code=True,
            enforce_eager=True
        )
        logger.info(f"✓ Successfully loaded {model_name}")
        logger.info("Expected performance: 60-80 tokens/sec with AWQ")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    """Format prompt with tool definitions for Qwen"""
    if not tools:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Build tool description
    tools_text = json.dumps(tools, indent=2)
    
    # Add context if provided
    ctx = ""
    if context and context.get("business_name"):
        ctx = f"\nContext: You are helping customers at {context['business_name']}.\n"
    
    # Qwen format with tools
    system_prompt = f"""You are a helpful assistant.{ctx}

You have access to the following tools:
{tools_text}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>

Only use tools when you need specific information not provided in the context."""
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response"""
    patterns = [
        r'<tool_call>\s*({[^}]+})\s*</tool_call>',
        r'<function_call>\s*({[^}]+})\s*</function_call>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool call: {match.group(1)}")
    
    return None

def generate(prompt: str, tools: List[Dict] = None, **kwargs) -> Dict:
    """Generate response with optional tool calling"""
    
    # Format prompt
    formatted_prompt = format_prompt_with_tools(prompt, tools, kwargs.get('context'))
    
    # Sampling parameters for optimal performance
    sampling_params = SamplingParams(
        temperature=kwargs.get('temperature', 0.7),
        max_tokens=kwargs.get('max_tokens', 512),
        top_p=0.95,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # Generate
    start_time = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    generation_time = time.time() - start_time
    
    # Extract response
    response_text = outputs[0].outputs[0].text.strip()
    token_count = len(outputs[0].outputs[0].token_ids)
    tokens_per_second = token_count / generation_time if generation_time > 0 else 0
    
    logger.info(f"Generated {token_count} tokens in {generation_time:.2f}s = {tokens_per_second:.1f} tok/s")
    
    # Check for tool calls
    tool_call = extract_tool_call(response_text) if tools else None
    
    return {
        'response': response_text,
        'tool_call': tool_call,
        'tokens': token_count,
        'time': generation_time,
        'speed': tokens_per_second
    }

# MIA Backend Integration
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    """Background worker for MIA jobs"""
    miner_id = f"qwen-awq-{socket.gethostname()}"
    logger.info(f"Starting worker with ID: {miner_id}")
    
    # Register with backend
    try:
        requests.post(
            f"{backend_url}/register_miner",
            json={
                "miner_id": miner_id,
                "model": model_name,
                "features": ["tool_calling", "awq", "high_performance"],
                "tokens_per_second": 70
            },
            timeout=10
        )
        logger.info("Registered with MIA backend")
    except Exception as e:
        logger.warning(f"Failed to register: {e}")
    
    # Main work loop
    while True:
        try:
            # Get work
            response = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                work = response.json()
                
                if work and work.get('request_id'):
                    logger.info(f"Processing job: {work['request_id']}")
                    
                    # Generate response
                    result = generate(
                        prompt=work.get('prompt', ''),
                        tools=work.get('tools'),
                        context=work.get('context'),
                        max_tokens=work.get('max_tokens', 512),
                        temperature=work.get('temperature', 0.7)
                    )
                    
                    # Prepare submission
                    submission = {
                        'response': result['response'],
                        'tokens_generated': result['tokens'],
                        'processing_time': result['time']
                    }
                    
                    # Add tool call if present
                    if result.get('tool_call'):
                        submission['tool_call'] = result['tool_call']
                        submission['requires_tool_execution'] = True
                    
                    # Submit result
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': miner_id,
                        'request_id': work['request_id'],
                        'result': submission
                    })
                    
                    logger.info(f"✓ Completed job: {work['request_id']} @ {result['speed']:.1f} tok/s")
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
        
        time.sleep(1)

def test_model():
    """Test model performance and tool calling"""
    logger.info("=== Running model tests ===")
    
    # Test 1: Simple generation
    result = generate("Hello, how are you?", max_tokens=50)
    logger.info(f"Test 1 - Speed: {result['speed']:.1f} tok/s")
    
    # Test 2: Tool calling
    tools = [{
        "name": "get_dish_details",
        "description": "Get detailed information about a menu item",
        "parameters": {
            "dish_name": {"type": "string", "description": "Name of the dish"}
        }
    }]
    
    result = generate(
        "What are the ingredients in the Lobster Ravioli?",
        tools=tools,
        context={"business_name": "Test Restaurant"}
    )
    
    if result.get('tool_call'):
        logger.info(f"Test 2 - Tool call detected: {result['tool_call']}")
    else:
        logger.info(f"Test 2 - Response: {result['response'][:100]}...")
    
    logger.info(f"Test 2 - Speed: {result['speed']:.1f} tok/s")

# API server
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def api_generate():
    """API endpoint for generation"""
    data = request.json
    result = generate(
        prompt=data.get('prompt', ''),
        tools=data.get('tools'),
        context=data.get('context'),
        max_tokens=data.get('max_tokens', 512),
        temperature=data.get('temperature', 0.7)
    )
    
    return jsonify({
        'text': result['response'],
        'tool_call': result.get('tool_call'),
        'tokens_generated': result['tokens'],
        'tokens_per_second': round(result['speed'], 1),
        'model': model_name
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if model else 'loading',
        'model': model_name,
        'quantization': 'AWQ',
        'expected_speed': '60-80 tokens/sec'
    })

if __name__ == "__main__":
    # Load model
    if not load_model():
        logger.error("Failed to load model, exiting")
        sys.exit(1)
    
    # Run tests
    test_model()
    
    # Start worker thread
    threading.Thread(target=worker, daemon=True).start()
    
    # Start API server
    logger.info("Starting API server on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

chmod +x miner.py

# Create start/stop scripts
cat > start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source vllm_env/bin/activate
nohup python miner.py > miner.log 2>&1 &
echo $! > miner.pid
echo "Qwen AWQ miner started. PID: $(cat miner.pid)"
echo "Logs: tail -f miner.log"
EOF
chmod +x start.sh

cat > stop.sh << 'EOF'
#!/bin/bash
if [ -f miner.pid ]; then
    kill $(cat miner.pid) 2>/dev/null && rm miner.pid
    echo "Miner stopped"
else
    pkill -f "miner.py" || echo "No miner process found"
fi
EOF
chmod +x stop.sh

echo ""
echo "=== Installation Complete ==="
echo "Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "Location: $MINER_DIR"
echo "Performance: 60-80 tokens/sec expected"
echo ""
echo "Commands:"
echo "  ./start.sh - Start miner"
echo "  ./stop.sh  - Stop miner"
echo "  tail -f miner.log - View logs"
echo ""

# Offer to start
read -p "Start miner now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./start.sh
fi