#!/usr/bin/env bash
# vLLM Production Installer - Fast, Stable, Tool-calling enabled
set -Eeuo pipefail

echo "🚀 vLLM Production Installer"
echo "==========================="
echo "• Python 3.11 with .venv at /data/qwen-awq-miner/.venv"
echo "• Wheels-only from official indexes"
echo "• All caches on /data disk"
echo "• Auto tool-calling with Hermes parser"
echo ""

# === 1. Create directories on /data ===
echo "📁 Creating directories..."
mkdir -p /data/qwen-awq-miner/logs
mkdir -p /data/cache/hf /data/cache/torch /data/.cache /data/tmp
cd /data/qwen-awq-miner

# === 2. Install Python 3.11 if needed ===
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "📦 Installing Python 3.11..."
    apt-get update -qq
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.11 python3.11-venv python3.11-distutils
fi

# === 3. Create .venv (NOT venv) ===
echo "🐍 Creating .venv with Python 3.11..."
if [[ -d ".venv" ]]; then
    echo "  Found existing .venv, using it"
else
    python3.11 -m venv .venv
fi
source .venv/bin/activate

# Verify we're in the right venv
echo "  Python: $(which python)"
echo "  Version: $(python --version)"

# === 4. Upgrade pip and core packages ===
echo "📦 Upgrading pip, wheel, setuptools..."
python -m pip install -U pip wheel setuptools packaging

# === 5. Configure pip for wheels-only ===
echo "🔒 Configuring pip (wheels-only, official PyPI)..."
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST || true
export PIP_NO_CACHE_DIR=1
export PIP_ONLY_BINARY=":all:"

# === 6. Install PyTorch with CUDA ===
echo "🔥 Installing PyTorch 2.7.1..."
if ! pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1; then
    echo "  CUDA 12.8 not available, trying CUDA 12.1..."
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
fi

# === 7. Install vLLM and backends ===
echo "⚡ Installing vLLM 0.10.1.1 and xFormers 0.0.31..."
pip install vllm==0.10.1.1
pip install xformers==0.0.31 || echo "  Warning: xFormers wheel not available"

# Try flash-attn (wheel only, no source builds)
echo "📦 Checking for flash-attn wheel..."
PIP_ONLY_BINARY=":all:" pip install flash-attn 2>/dev/null || echo "  No flash-attn wheel available (OK)"

# === 8. Move caches to /data ===
echo "🗂️ Moving caches to /data..."
if [[ -d "/root/.cache" ]] && [[ ! -L "/root/.cache" ]]; then
    echo "  Migrating existing cache..."
    rsync -aH --remove-source-files /root/.cache/ /data/.cache/ 2>/dev/null || true
    rm -rf /root/.cache
fi
[[ ! -e "/root/.cache" ]] && ln -s /data/.cache /root/.cache

# === 9. Create start_vllm.sh ===
echo "✍️ Creating start_vllm.sh..."
cat > start_vllm.sh << 'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

# Change to script directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Activate the correct venv
source .venv/bin/activate

# Set all caches to /data
export HF_HOME=/data/cache/hf
export HUGGINGFACE_HUB_CACHE=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export TORCH_HOME=/data/cache/torch
export XDG_CACHE_HOME=/data/.cache
export TMPDIR=/data/tmp

# Use xFormers by default (fastest on RTX 3090)
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

# Server configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-12288}"  # 12k context
UTIL="${GPU_UTIL:-0.90}"   # 90% GPU memory

# Paths
LOGS="$DIR/logs"
PID="$DIR/vllm.pid"
mkdir -p "$LOGS"

# Stop only our previous instance (by PID)
if [[ -f "$PID" ]]; then
    OLD_PID=$(cat "$PID")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping previous vLLM (PID $OLD_PID)..."
        kill -TERM "$OLD_PID" || true
        sleep 2
    fi
fi

# Start vLLM with production settings
echo "Starting vLLM server..."
echo "  Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "  Quantization: AWQ"
echo "  Context: ${MAXLEN} tokens"
echo "  Tool parser: Hermes"
echo "  API: http://${HOST}:${PORT}/v1"

nohup vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --host "$HOST" \
    --port "$PORT" \
    --quantization awq \
    --dtype half \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "$LOGS/vllm.out" 2>&1 &

# Save PID
echo $! > "$PID"
echo "✅ Started vLLM (PID $(cat "$PID"))"
echo "📋 Logs: tail -f $LOGS/vllm.out"
SCRIPT
chmod +x start_vllm.sh

# === 10. Create test script ===
echo "✍️ Creating test_vllm.sh..."
cat > test_vllm.sh << 'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
AUTH="Authorization: Bearer ${OPENAI_API_KEY:-sk-LOCAL}"

echo "🧪 Testing vLLM API"
echo "=================="

# Test 1: Check models endpoint
echo -e "\n📋 GET /models:"
curl -sS "$BASE/models" -H "$AUTH" | python -m json.tool | head -20

# Test 2: Simple completion (no tools)
echo -e "\n💬 Chat completion (no tools):"
curl -sS "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "$AUTH" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Say hello in 5 words"}],
        "max_tokens": 50
    }' | python -m json.tool | grep -E "(content|role)" || echo "Failed"

# Test 3: Tool calling with auto mode
echo -e "\n🔧 Tool calling (auto mode):"
curl -sS "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "$AUTH" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "What fish dishes do you have?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_menu_items",
                "description": "Search menu items by ingredient, category, or name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string"},
                        "search_type": {"type": "string", "enum": ["ingredient", "category", "name"]}
                    },
                    "required": ["search_term", "search_type"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0
    }' | python -m json.tool | grep -A10 "tool_calls" || echo "No tool calls"

echo -e "\n✅ Tests complete"
SCRIPT
chmod +x test_vllm.sh

# === 11. Create management script ===
echo "✍️ Creating vllm_manage.sh..."
cat > vllm_manage.sh << 'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

case "${1:-help}" in
    start)
        ./start_vllm.sh
        ;;
    stop)
        if [[ -f vllm.pid ]]; then
            PID=$(cat vllm.pid)
            if kill -0 "$PID" 2>/dev/null; then
                echo "Stopping vLLM (PID $PID)..."
                kill -TERM "$PID"
                rm -f vllm.pid
                echo "✅ Stopped"
            else
                echo "Process not running (stale PID file)"
                rm -f vllm.pid
            fi
        else
            echo "No PID file found"
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    status)
        if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
            echo "✅ vLLM running (PID $(cat vllm.pid))"
            echo "📊 Memory usage:"
            ps -p "$(cat vllm.pid)" -o pid,vsz,rss,comm
        else
            echo "❌ vLLM not running"
        fi
        ;;
    logs)
        tail -f logs/vllm.out
        ;;
    test)
        ./test_vllm.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "  start    - Start vLLM server"
        echo "  stop     - Stop vLLM server (by PID)"
        echo "  restart  - Restart vLLM server"
        echo "  status   - Check if running"
        echo "  logs     - Tail logs"
        echo "  test     - Run API tests"
        ;;
esac
SCRIPT
chmod +x vllm_manage.sh

# === 12. Verify installation ===
echo -e "\n✅ Installation complete!"
echo "========================"

# Quick verification
python -c "
import sys, torch
print(f'Python: {sys.version.split()[0]} at {sys.executable}')
print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
try:
    import vllm
    print(f'vLLM: {vllm.__version__}')
except ImportError:
    print('vLLM: NOT INSTALLED - check pip logs')
try:
    import xformers
    print(f'xFormers: installed')
except ImportError:
    print('xFormers: not available')
"

echo -e "\n📁 Installation location: /data/qwen-awq-miner"
echo "🐍 Virtual environment: /data/qwen-awq-miner/.venv"
echo ""
echo "To start vLLM:"
echo "  cd /data/qwen-awq-miner"
echo "  ./vllm_manage.sh start    # or ./start_vllm.sh directly"
echo ""
echo "Management commands:"
echo "  ./vllm_manage.sh status   # Check if running"
echo "  ./vllm_manage.sh logs     # View logs"
echo "  ./vllm_manage.sh test     # Test API"
echo "  ./vllm_manage.sh stop     # Stop server"
echo ""
echo "The server will download the model on first start (~4GB)."