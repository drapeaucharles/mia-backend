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

# === 10. Create polling miner ===
echo "✍️ Creating miner.py..."
cat > miner.py << 'SCRIPT'
#!/usr/bin/env python3
"""MIA Job Polling Miner - Auto-registers with backend"""
import os, sys, json, time, logging, requests, socket
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mia-miner")

MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000/v1"

class MIAMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = None
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        
    def register(self) -> bool:
        """Register with backend and get miner ID"""
        try:
            response = self.session.post(
                f"{MIA_BACKEND_URL}/register_miner",
                json={"name": self.miner_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.miner_id = int(data["miner_id"])
                logger.info(f"✓ Registered as miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Registration error: {e}")
        return False
        
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
                    messages.append({"role": "system", "content": f"You are a helpful assistant at {context['business_name']}."})
            
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
                logger.info(f"Tool call: {result['tool_call']['name']}")
            
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
        logger.info(f"MIA Miner starting...")
        logger.info(f"Backend: {MIA_BACKEND_URL}")
        
        # Wait for vLLM to be ready
        logger.info("Waiting for vLLM...")
        for i in range(60):
            try:
                test = self.session.get(f"{VLLM_URL}/models", timeout=5)
                if test.status_code == 200:
                    logger.info("✓ vLLM is ready")
                    break
            except:
                pass
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()
        else:
            logger.error("vLLM timeout - make sure it's running")
            sys.exit(1)
        
        # Register with backend
        if not self.register():
            logger.error("Failed to register with backend")
            sys.exit(1)
        
        jobs_completed = 0
        total_tokens = 0
        errors = 0
        
        logger.info(f"Starting job polling loop as miner {self.miner_id}...")
        
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
                    logger.warning("Too many errors, re-registering...")
                    self.register()
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
SCRIPT
chmod +x miner.py

# === 11. Create start_miner.sh ===
echo "✍️ Creating start_miner.sh..."
cat > start_miner.sh << 'SCRIPT'
#!/bin/bash
cd "$(dirname "$0")"

# First ensure vLLM is running
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM..."
    ./start_vllm.sh
    echo "Waiting for vLLM to download model and start (this may take a few minutes)..."
    sleep 15
fi

# Activate virtual environment
source .venv/bin/activate

# Install requests if needed
pip install requests 2>/dev/null || true

# Set environment
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export MIA_BACKEND_URL=${MIA_BACKEND_URL:-https://mia-backend-production.up.railway.app}

echo "Starting polling miner (auto-registering with backend)..."
python miner.py 2>&1 | tee -a miner.log
SCRIPT
chmod +x start_miner.sh

# === 12. Create test script ===
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

# === 11. Update management script to include miner ===
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
    miner)
        ./start_miner.sh
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test|miner}"
        echo ""
        echo "  start    - Start vLLM server"
        echo "  stop     - Stop vLLM server (by PID)"
        echo "  restart  - Restart vLLM server"
        echo "  status   - Check if running"
        echo "  logs     - Tail logs"
        echo "  test     - Run API tests"
        echo "  miner    - Start polling miner"
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

# === 11. Install bore client ===
echo "🌐 Installing bore client..."
if [ ! -f "/usr/local/bin/bore" ]; then
    cd /tmp
    wget -q https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    tar -xzf bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    mv bore /usr/local/bin/
    rm bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    cd /data/qwen-awq-miner
fi

# === 12. Install additional packages for heartbeat miner ===
echo "📦 Installing packages for heartbeat miner..."
pip install flask waitress aiohttp

# === 13. Create heartbeat miner with bore auto-restart ===
cat > /data/qwen-awq-miner/mia_miner_heartbeat.py << 'EOF'
#!/usr/bin/env python3
"""MIA Heartbeat Miner with bore.pub support and auto-restart"""
import requests
import time
import logging
import sys
import json
import os
import socket
import threading
import asyncio
from flask import Flask, request, jsonify
from datetime import datetime
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('miner_heartbeat.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app for receiving pushed work
app = Flask(__name__)

class HeartbeatMiner:
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "https://mia-backend-production.up.railway.app")
        self.miner_id = None
        self.miner_key = None
        self.is_processing = False
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        self.heartbeat_interval = 1.0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.public_url = None
        self.bore_process = None
        self.bore_port = None
        self.bore_check_interval = 30  # Check bore health every 30 seconds
        self.last_bore_check = time.time()
        self.bore_failures = 0
        self.max_bore_failures = 3
        
    def start_bore(self):
        """Start bore tunnel and get public URL"""
        try:
            # Kill any existing bore process
            if self.bore_process:
                logger.info("Killing existing bore process...")
                self.bore_process.terminate()
                time.sleep(2)
                if self.bore_process.poll() is None:
                    self.bore_process.kill()
                self.bore_process = None
            
            logger.info("🌐 Starting bore tunnel...")
            
            # Start bore client
            cmd = ['bore', 'local', '5000', '--to', 'bore.pub']
            
            self.bore_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read output to get the port
            for i in range(30):
                if self.bore_process.stdout:
                    line = self.bore_process.stdout.readline()
                    if line:
                        logger.info(f"Bore output: {line.strip()}")
                        
                        # Look for port assignment
                        port_match = re.search(r'bore\.pub:(\d+)', line)
                        if port_match:
                            self.bore_port = port_match.group(1)
                            self.public_url = f"http://bore.pub:{self.bore_port}"
                            logger.info(f"✅ Public URL: {self.public_url}")
                            
                            # Reset failure counter
                            self.bore_failures = 0
                            self.last_bore_check = time.time()
                            
                            # Continue reading in background
                            threading.Thread(target=self._read_bore_output, daemon=True).start()
                            
                            # Re-register with new URL if we already have a miner_id
                            if self.miner_id:
                                logger.info("Re-registering with new bore URL...")
                                self.register()
                            
                            return True
                
                time.sleep(1)
                
                if self.bore_process.poll() is not None:
                    output = self.bore_process.stdout.read() if self.bore_process.stdout else ""
                    logger.error(f"Bore process died: {output}")
                    return False
                    
            logger.error("Failed to get bore port after 30 seconds")
            return False
                
        except Exception as e:
            logger.error(f"Failed to start bore: {e}")
            return False
    
    def _read_bore_output(self):
        """Keep reading bore output and monitor for disconnections"""
        while self.bore_process and self.bore_process.poll() is None:
            try:
                line = self.bore_process.stdout.readline()
                if line:
                    logger.debug(f"Bore: {line.strip()}")
                    # Look for disconnection messages
                    if "disconnected" in line.lower() or "error" in line.lower():
                        logger.warning(f"Bore tunnel issue detected: {line.strip()}")
                        self.bore_failures += 1
            except:
                break
        
        logger.warning("Bore output reader stopped - tunnel may be down")
        self.bore_failures += 1
    
    def check_bore_health(self):
        """Check if bore tunnel is healthy and restart if needed"""
        try:
            if not self.public_url:
                return False
            
            # Check if bore process is alive
            if self.bore_process and self.bore_process.poll() is not None:
                logger.warning("Bore process died")
                return False
            
            # Test if URL is accessible (internal test)
            try:
                response = requests.get(f"http://localhost:5000/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            # If we can't reach ourselves, bore might be down
            logger.warning("Bore tunnel health check failed")
            return False
            
        except Exception as e:
            logger.error(f"Bore health check error: {e}")
            return False
    
    def restart_bore_if_needed(self):
        """Check and restart bore if needed"""
        current_time = time.time()
        
        # Only check every bore_check_interval seconds
        if current_time - self.last_bore_check < self.bore_check_interval:
            return
        
        self.last_bore_check = current_time
        
        # Check bore health
        if not self.check_bore_health():
            logger.warning(f"Bore tunnel unhealthy (failures: {self.bore_failures})")
            
            # Restart if we've hit the failure threshold
            if self.bore_failures >= self.max_bore_failures:
                logger.info("🔄 Restarting bore tunnel...")
                self.bore_failures = 0  # Reset counter
                
                # Restart bore
                if self.start_bore():
                    logger.info("✅ Bore tunnel restarted successfully")
                else:
                    logger.error("❌ Failed to restart bore tunnel")
                    # Wait a bit before trying again
                    time.sleep(10)
            
    def register(self):
        """Register with backend using bore URL"""
        if not self.public_url:
            logger.error("No public URL available")
            return False
            
        try:
            hostname = socket.gethostname()
            
            data = {
                "ip_address": f"bore.pub:{self.bore_port}",
                "gpu_info": self.get_gpu_info(),
                "hostname": hostname,
                "backend_type": "vllm-heartbeat-bore",
                "capabilities": ["chat", "completion", "tools"],
                "public_url": self.public_url
            }
            
            response = requests.post(
                f"{self.backend_url}/register",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.miner_id = result["miner_id"]
                self.miner_key = result["auth_key"]
                logger.info(f"✅ Registered as miner {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "name": torch.cuda.get_device_name(0),
                    "memory_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2
                }
        except:
            pass
        return {"name": "vLLM GPU", "memory_mb": 8192}
    
    async def send_heartbeat(self):
        """Send heartbeat to backend"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Check if bore needs restart (non-blocking)
                    self.restart_bore_if_needed()
                    
                    if not self.is_processing and self.public_url:
                        data = {
                            "miner_id": self.miner_id,
                            "status": "available",
                            "timestamp": datetime.utcnow().isoformat(),
                            "port": int(self.bore_port) if self.bore_port else 80,
                            "public_url": self.public_url
                        }
                        
                        async with session.post(
                            f"{self.backend_url}/heartbeat",
                            json=data,
                            headers={"Authorization": f"Bearer {self.miner_key}"},
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                logger.debug("❤️ Heartbeat sent")
                                # Reset bore failures on successful heartbeat
                                if self.bore_failures > 0 and self.bore_failures < self.max_bore_failures:
                                    self.bore_failures = max(0, self.bore_failures - 1)
                            else:
                                logger.warning(f"Heartbeat failed: {response.status}")
                    
                    await asyncio.sleep(self.heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    await asyncio.sleep(5)
    
    def process_job(self, job_data):
        """Process a job pushed from backend"""
        self.is_processing = True
        request_id = job_data.get('request_id', 'unknown')
        
        try:
            logger.info(f"🔧 Processing job {request_id}")
            start_time = time.time()
            
            # Extract prompt and parameters
            prompt = job_data.get('prompt', '')
            messages = job_data.get('messages', [{'role': 'user', 'content': prompt}])
            tools = job_data.get('tools', [])
            
            # Prepare vLLM request - IMPORTANT: Use correct model name
            vllm_request = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": messages,
                "temperature": job_data.get('temperature', 0.7),
                "max_tokens": job_data.get('max_tokens', 2000),
                "stream": False
            }
            
            if tools:
                vllm_request["tools"] = tools
                vllm_request["tool_choice"] = job_data.get('tool_choice', 'auto')
            
            # Call vLLM
            response = requests.post(
                self.vllm_url,
                json=vllm_request,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                choice = result['choices'][0]
                
                # Extract response
                response_data = {
                    "success": True,
                    "response": choice['message']['content'],
                    "tool_calls": choice['message'].get('tool_calls', []),
                    "tokens_generated": result['usage']['completion_tokens'],
                    "processing_time": time.time() - start_time
                }
                
                logger.info(f"✅ Job {request_id} completed in {response_data['processing_time']:.2f}s")
                return response_data
            else:
                logger.error(f"vLLM error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"vLLM error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.is_processing = False
    
    def run_flask_server(self):
        """Run Flask server to receive pushed work"""
        @app.route('/process', methods=['POST'])
        def receive_work():
            try:
                logger.info(f"Received work request")
                job_data = request.json
                future = self.executor.submit(self.process_job, job_data)
                result = future.result(timeout=60)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error receiving work: {e}")
                return jsonify({"error": str(e)}), 500
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "ready",
                "miner_id": self.miner_id,
                "is_processing": self.is_processing,
                "public_url": self.public_url,
                "bore_status": "healthy" if self.bore_failures < self.max_bore_failures else "unhealthy",
                "bore_failures": self.bore_failures
            })
        
        @app.route('/', methods=['GET'])
        def index():
            return jsonify({
                "service": "MIA GPU Miner",
                "status": "running",
                "public_url": self.public_url,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        from waitress import serve
        logger.info("🌐 Starting Flask server on port 5000")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    
    def run(self):
        """Main run loop"""
        # Start bore first
        attempts = 0
        while not self.start_bore():
            attempts += 1
            if attempts > 5:
                logger.error("Failed to start bore tunnel after 5 attempts")
                sys.exit(1)
            logger.info(f"Retrying bore start in 10s... (attempt {attempts}/5)")
            time.sleep(10)
            
        # Register with backend
        attempts = 0
        while not self.register():
            attempts += 1
            if attempts > 5:
                logger.error("Failed to register after 5 attempts")
                sys.exit(1)
            logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
            time.sleep(30)
        
        # Start Flask server in separate thread
        flask_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        flask_thread.start()
        
        # Start heartbeat loop
        logger.info("💓 Starting heartbeat loop with bore monitoring")
        try:
            asyncio.run(self.send_heartbeat())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.bore_process:
                self.bore_process.terminate()

if __name__ == "__main__":
    # Wait for vLLM to be ready
    logger.info("⏳ Waiting for vLLM to start...")
    for i in range(60):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                logger.info("✅ vLLM is ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        logger.error("❌ vLLM failed to start after 120 seconds")
        sys.exit(1)
    
    # Start miner
    miner = HeartbeatMiner()
    
    try:
        miner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
EOF
chmod +x /data/qwen-awq-miner/mia_miner_heartbeat.py

# === 14. Create unified start script ===
cat > /data/qwen-awq-miner/start_mia_gpu.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

echo "🚀 Starting MIA GPU Miner..."

# Stop any existing processes
echo "Stopping old processes..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill bore 2>/dev/null || true
for pid_file in *.pid; do
    if [ -f "$pid_file" ]; then
        kill $(cat $pid_file) 2>/dev/null || true
        rm -f $pid_file
    fi
done
sleep 3

# Start vLLM
echo "Starting vLLM..."
./vllm_manage.sh start

# Wait for vLLM
echo "Waiting for vLLM to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "✅ vLLM is ready!"
        break
    fi
    sleep 2
done

# Start heartbeat miner
echo "Starting heartbeat miner with bore.pub..."
nohup python3 mia_miner_heartbeat.py > heartbeat.out 2>&1 &
echo $! > heartbeat.pid

echo ""
echo "✅ MIA GPU Miner started!"
echo ""
echo "📊 Status commands:"
echo "  - vLLM logs: ./vllm_manage.sh logs"
echo "  - Miner logs: tail -f heartbeat.out"
echo "  - Stop all: pkill -f vllm && kill \$(cat heartbeat.pid)"
echo ""
echo "Your GPU will:"
echo "1. Register with MIA backend"
echo "2. Get a public bore.pub URL"
echo "3. Receive AI jobs via push architecture"
echo "4. Process with vLLM (Qwen 7B AWQ)"
echo "5. Auto-restart bore tunnel if disconnected"
EOF
chmod +x /data/qwen-awq-miner/start_mia_gpu.sh

echo -e "\n📁 Installation location: /data/qwen-awq-miner"
echo "🐍 Virtual environment: /data/qwen-awq-miner/.venv"
echo ""
echo "✅ Installation Complete!"
echo "======================================="
echo ""
echo "📌 Quick Start:"
echo "  cd /data/qwen-awq-miner"
echo "  ./start_mia_gpu.sh"
echo ""
echo "📌 Management commands:"
echo "  ./vllm_manage.sh status   # Check if running"
echo "  ./vllm_manage.sh logs     # View vLLM logs"
echo "  ./vllm_manage.sh stop     # Stop server"
echo ""
echo "📌 One-line installer for next time:"
echo "  curl -sL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-mia-gpu-miner-universal.sh | sudo bash"
echo ""
echo "The server will download the model on first start (~4GB)."

# Ask if user wants to start now
read -p "Start MIA GPU miner now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /data/qwen-awq-miner
    ./start_mia_gpu.sh
fi