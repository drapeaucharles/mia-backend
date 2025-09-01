#!/bin/bash
# Final production vLLM installer with all best practices

bash -lc '
set -Eeuo pipefail

# === Paths ===
DIR=/data/qwen-awq-miner
mkdir -p "$DIR/logs" /data/cache/hf /data/cache/torch /data/.cache /data/tmp
cd "$DIR"

# === Stop any vLLM we started earlier (safe) ===
if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
  kill -TERM "$(cat vllm.pid)" || true
  sleep 1
fi
pkill -f "vllm serve" 2>/dev/null || true

# === Python 3.11 & venv ===
if ! command -v python3.11 >/dev/null 2>&1; then
  apt-get update && apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update && apt-get install -y python3.11 python3.11-venv python3.11-distutils
fi
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools packaging

# === Harden pip to avoid sketchy build backends (NO source builds) ===
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST || true
export PIP_NO_CACHE_DIR=1
export PIP_ONLY_BINARY=":all:"

# === Torch stack (try CUDA 12.8 first, fallback to 12.1) ===
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
|| pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# === vLLM + speed backends ===
pip install vllm==0.10.1.1
pip install xformers==0.0.31 || true
PIP_ONLY_BINARY=":all:" pip install flash-attn || true   # only if a wheel exists; skip otherwise

# === Put all caches on /data (free root FS) ===
rm -rf /root/.cache 2>/dev/null || true
ln -s /data/.cache /root/.cache

# === Create start script (fast defaults + auto tool calling) ===
cat > start_vllm.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
source .venv/bin/activate
export HF_HOME=/data/cache/hf
export HUGGINGFACE_HUB_CACHE=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export TORCH_HOME=/data/cache/torch
export XDG_CACHE_HOME=/data/.cache
export TMPDIR=/data/tmp
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-12288}          # 12k context for speed on 24GB
UTIL=${GPU_UTIL:-0.90}
LOGS="$DIR/logs"
PID="$DIR/vllm.pid"
mkdir -p "$LOGS"
# stop only our own previous process
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
echo "Started vLLM (PID $(cat "$PID")) | URL: http://$HOST:$PORT/v1 | Logs: $LOGS/vllm.out"
SH
chmod +x start_vllm.sh

# === Simple API smoke test (models + auto tool-call example) ===
cat > test_vllm.sh << "SH"
#!/usr/bin/env bash
set -Eeuo pipefail
BASE=${BASE:-http://127.0.0.1:8000/v1}
AUTH="Authorization: Bearer sk-LOCAL"
echo "== /models =="
curl -sS "$BASE/models" -H "$AUTH" | sed -n "1,120p"
echo -e "\n== Tool (auto) example: =="
curl -sS "$BASE/chat/completions" \
  -H "Content-Type: application/json" -H "$AUTH" \
  -d "{\"model\":\"Qwen/Qwen2.5-7B-Instruct-AWQ\",
       \"messages\":[{\"role\":\"user\",\"content\":\"I want fish\"}],
       \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"search_menu_items\",
         \"description\":\"Search menu\",\"parameters\":{\"type\":\"object\",
         \"properties\":{\"search_term\":{\"type\":\"string\"},\"search_type\":{\"type\":\"string\",\"enum\":[\"ingredient\",\"category\",\"name\"]}},
         \"required\":[\"search_term\",\"search_type\"]}}}],
       \"tool_choice\":\"auto\",\"temperature\":0}" \
  | sed -n "1,200p"
SH
chmod +x test_vllm.sh

# === Launch now ===
./start_vllm.sh
sleep 3
echo
echo "Sanity check:"
./test_vllm.sh || true
'