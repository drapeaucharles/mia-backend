#!/usr/bin/env bash
# vLLM Production Installer with Job Polling Miner
bash -lc '
set -Eeuo pipefail

### 0) Stop any previous server/miner
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

### 3) Harden pip: official PyPI + wheels-only
export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL PIP_TRUSTED_HOST
export PIP_NO_CACHE_DIR=1
export PIP_ONLY_BINARY=":all:"

### 4) Torch 2.7.1 CUDA wheels
if ! python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 ; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
fi

### 5) vLLM + fast attention backend
python -m pip install vllm==0.10.1.1
python -m pip install xformers==0.0.31 || true
PIP_ONLY_BINARY=":all:" python -m pip install flash-attn || true

### 6) Install requests for miner
python -m pip install requests

### 7) Make caches live on /data
rsync -aH --remove-source-files /root/.cache/huggingface/ /data/cache/hf/ 2>/dev/null || true
rsync -aH --remove-source-files /root/.cache/torch/ /data/cache/torch/ 2>/dev/null || true
rsync -aH --remove-source-files /root/.cache/ /data/.cache/ 2>/dev/null || true
rm -rf /root/.cache || true
ln -s /data/.cache /root/.cache

### 8) Write vLLM start script
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

# stop old instance
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
echo "vLLM started: PID $(cat "$PID") | Logs: $LOGS/vllm.out"
SH
chmod +x start_vllm.sh

### 9) Write polling miner
cat > miner.py << "PY"
#!/usr/bin/env python3
"""MIA Job Polling Miner"""
import os, sys, json, time, logging, requests
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mia-miner")

MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000/v1"
MINER_ID = int(os.getenv("MINER_ID", "1"))

class MIAMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = MINER_ID
        
    def get_work(self) -> Optional[Dict]:
        try:
            r = self.session.get(f"{MIA_BACKEND_URL}/get_work", params={"miner_id": self.miner_id}, timeout=30)
            if r.status_code == 200:
                work = r.json()
                if work and work.get("request_id"):
                    return work
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            logger.error(f"Error getting work: {e}")
        return None
    
    def process_with_vllm(self, job: Dict) -> Dict:
        try:
            messages = []
            context = job.get("context", {})
            if isinstance(context, dict):
                if context.get("system_prompt"):
                    messages.append({"role": "system", "content": context["system_prompt"]})
                elif context.get("business_name"):
                    messages.append({"role": "system", "content": f"You are a helpful assistant at {context["business_name"]}."})
            
            messages.append({"role": "user", "content": job.get("prompt", "")})
            
            vllm_request = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": messages,
                "max_tokens": job.get("max_tokens", 150),
                "temperature": job.get("temperature", 0.7)
            }
            
            if job.get("tools"):
                vllm_request["tools"] = job["tools"]
                vllm_request["tool_choice"] = job.get("tool_choice", "auto")
            
            start_time = time.time()
            r = self.session.post(f"{VLLM_URL}/chat/completions", json=vllm_request, timeout=60)
            processing_time = time.time() - start_time
            
            if r.status_code != 200:
                return {"response": f"Error: vLLM returned {r.status_code}", "tokens_generated": 0, "processing_time": processing_time}
            
            vllm_result = r.json()
            message = vllm_result["choices"][0]["message"]
            
            result = {
                "response": message.get("content", ""),
                "tokens_generated": vllm_result.get("usage", {}).get("completion_tokens", 0),
                "processing_time": processing_time,
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ"
            }
            
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                result["tool_call"] = {
                    "name": tool_call["function"]["name"],
                    "parameters": json.loads(tool_call["function"]["arguments"])
                }
                result["requires_tool_execution"] = True
                logger.info(f"Tool call: {result["tool_call"]["name"]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"response": f"Error: {str(e)}", "tokens_generated": 0, "processing_time": 0}
    
    def submit_result(self, request_id: str, result: Dict) -> bool:
        try:
            r = self.session.post(
                f"{MIA_BACKEND_URL}/submit_result",
                json={"miner_id": self.miner_id, "request_id": request_id, "result": result},
                timeout=10
            )
            if r.status_code == 200:
                logger.info(f"✓ Submitted {request_id}")
                return True
            else:
                logger.error(f"Submit failed: {r.status_code}")
        except Exception as e:
            logger.error(f"Submit error: {e}")
        return False
    
    def run(self):
        logger.info(f"MIA Miner | Backend: {MIA_BACKEND_URL} | Miner ID: {self.miner_id}")
        
        # Test vLLM
        try:
            test = self.session.get(f"{VLLM_URL}/models", timeout=5)
            if test.status_code == 200:
                logger.info("✓ vLLM is running")
            else:
                logger.error("vLLM not ready")
                sys.exit(1)
        except Exception as e:
            logger.error(f"vLLM not running: {e}")
            sys.exit(1)
        
        jobs_completed = 0
        total_tokens = 0
        errors = 0
        
        while True:
            try:
                job = self.get_work()
                if job:
                    request_id = job["request_id"]
                    logger.info(f"Job: {request_id}")
                    
                    result = self.process_with_vllm(job)
                    
                    if self.submit_result(request_id, result):
                        jobs_completed += 1
                        total_tokens += result.get("tokens_generated", 0)
                        errors = 0
                        logger.info(f"Stats: {jobs_completed} jobs, {total_tokens} tokens")
                    else:
                        errors += 1
                else:
                    time.sleep(2)
                
                if errors > 5:
                    logger.warning("Too many errors, pausing...")
                    time.sleep(30)
                    errors = 0
                    
            except KeyboardInterrupt:
                logger.info(f"\nDone: {jobs_completed} jobs, {total_tokens} tokens")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                errors += 1
                time.sleep(5)

if __name__ == "__main__":
    MIAMiner().run()
PY
chmod +x miner.py

### 10) Write start script
cat > start_miner.sh << "SH"
#!/bin/bash
cd "$(dirname "$0")"
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM..."
    ./start_vllm.sh
    sleep 10
fi
source .venv/bin/activate
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export MIA_BACKEND_URL=${MIA_BACKEND_URL:-https://mia-backend-production.up.railway.app}
export MINER_ID=${MINER_ID:-1}
echo "Starting polling miner (ID: $MINER_ID)..."
python miner.py 2>&1 | tee -a miner.log
SH
chmod +x start_miner.sh

### 11) Management script
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
      rm -f vllm.pid
      echo "Stopped vLLM"
    fi
    pkill -f miner.py || true
    ;;
  status)
    if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
      echo "✓ vLLM running: PID $(cat vllm.pid)"
    else
      echo "✗ vLLM not running"
    fi
    if pgrep -f miner.py > /dev/null; then
      echo "✓ Miner running"
    else
      echo "✗ Miner not running"
    fi
    ;;
  logs)
    tail -f logs/vllm.out
    ;;
  miner)
    ./start_miner.sh
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|miner}"
    exit 1
    ;;
esac
SH
chmod +x vllm_manage.sh

### 12) Start everything
echo "== Starting vLLM =="
./start_vllm.sh
sleep 10

echo "== Installation complete! =="
echo "To start mining: export MINER_ID=your_id && ./start_miner.sh"
echo "Management: ./vllm_manage.sh {start|stop|status|logs|miner}"
'