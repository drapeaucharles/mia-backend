#!/usr/bin/env bash
# Production vLLM installer - robust, uses /data, wheels-only

bash -lc '
set -Eeuo pipefail

### 0) Stop any previous server/miner (safe to ignore errors)
pkill -f "vllm serve" 2>/dev/null || true
pkill -f miner.py 2>/dev/null || true

### 1) Folders on the big disk
mkdir -p /data/qwen-awq-miner /data/cache/hf /data/cache/torch /data/.cache /data/tmp
cd /data/qwen-awq-miner

### 2) Python 3.11 + fresh venv (Ubuntu 20.04+)
apt-get update
apt-get install -y software-properties-common curl ca-certificates gnupg
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-distutils
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools packaging

### 3) Harden pip: official PyPI + wheels-only (avoids bad/poisoned sdists)
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_NO_CACHE_DIR=1
export PIP_ONLY_BINARY=":all:"

### 4) Torch 2.7.1 CUDA wheels (first try cu128, fallback to cu121)
if ! python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 ; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
fi

### 5) vLLM + fast attention backend
python -m pip install vllm==0.10.1.1
python -m pip install xformers==0.0.31 || true              # prefer xFormers wheel
PIP_ONLY_BINARY=":all:" python -m pip install flash-attn || true  # optional, wheel-only

### 6) Make caches live on /data (avoid filling /)
# move any existing root caches over, then link
rsync -aH --remove-source-files /root/.cache/huggingface/ /data/cache/hf/ 2>/dev/null || true
rsync -aH --remove-source-files /root/.cache/torch/       /data/cache/torch/ 2>/dev/null || true
rsync -aH --remove-source-files /root/.cache/             /data/.cache/      2>/dev/null || true
rm -rf /root/.cache || true
ln -s /data/.cache /root/.cache

### 7) Write a robust start script (OpenAI-compatible API + tools)
cat > start_vllm.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
source .venv/bin/activate

# All caches on /data
export HF_HOME=/data/cache/hf
export HUGGINGFACE_HUB_CACHE=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export TORCH_HOME=/data/cache/torch
export XDG_CACHE_HOME=/data/.cache
export TMPDIR=/data/tmp

# Fast attention + sane context
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-12288}"
UTIL="${GPU_UTIL:-0.90}"
LOGS="$DIR/logs"; PID="$DIR/vllm.pid"
mkdir -p "$LOGS"

# stop old instance (only the one we started)
if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  kill -TERM "$(cat "$PID")" || true
  sleep 1
fi

nohup vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host "$HOST" --port "$PORT" \
  --quantization awq \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$LOGS/vllm.out" 2>&1 &

echo $! > "$PID"
echo "vLLM started: PID $(cat "$PID") | Logs: $LOGS/vllm.out | Base URL: http://127.0.0.1:$PORT/v1"
SH
chmod +x start_vllm.sh

### 8) Simple verify + test scripts
cat > verify_setup.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
source .venv/bin/activate
python - <<PY
import torch, importlib, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda_build:", getattr(torch.version,"cuda",None), "cuda_avail:", torch.cuda.is_available())
try:
    import vllm; print("vllm:", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
print("xformers present:", importlib.util.find_spec("xformers") is not None)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
SH
chmod +x verify_setup.sh

cat > test_vllm.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
BASE="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
AUTH="Authorization: Bearer ${OPENAI_API_KEY:-sk-LOCAL}"
echo "== /models =="
curl -sS "$BASE/models" -H "$AUTH" | sed -e "s/{/\\n{/g" | head -n 10
echo
echo "== Chat =="
curl -sS "$BASE/chat/completions" -H "Content-Type: application/json" -H "$AUTH" \
  -d "{\"model\":\"Qwen/Qwen2.5-7B-Instruct-AWQ\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in one sentence.\"}],\"tool_choice\":\"none\",\"max_tokens\":64}" \
  | sed -e "s/{/\\n{/g" | head -n 40
echo
SH
chmod +x test_vllm.sh

### 9) Management commands
cat > vllm_manage.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

case "${1:-}" in
  start)
    ./start_vllm.sh
    ;;
  stop)
    if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
      kill -TERM "$(cat vllm.pid)"
      echo "Stopped vLLM (PID $(cat vllm.pid))"
      rm -f vllm.pid
    else
      echo "vLLM not running"
    fi
    ;;
  status)
    if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
      echo "vLLM running: PID $(cat vllm.pid)"
      echo "Logs: tail -f logs/vllm.out"
    else
      echo "vLLM not running"
    fi
    ;;
  logs)
    tail -f logs/vllm.out
    ;;
  test)
    ./test_vllm.sh
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|test}"
    exit 1
    ;;
esac
SH
chmod +x vllm_manage.sh

### 10) Show summary and start the server
echo "== Sanity =="
./verify_setup.sh
echo
echo "== Launching vLLM =="
./start_vllm.sh
sleep 5
echo
echo "== Quick test =="
./test_vllm.sh || echo "(Server may still be starting, try: ./test_vllm.sh)"
echo
echo "== Management =="
echo "  ./vllm_manage.sh start   # Start server"
echo "  ./vllm_manage.sh stop    # Stop server"
echo "  ./vllm_manage.sh status  # Check status"
echo "  ./vllm_manage.sh logs    # View logs"
echo "  ./vllm_manage.sh test    # Test API"
'