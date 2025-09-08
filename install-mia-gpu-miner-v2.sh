#!/bin/bash
# MIA GPU Miner - Complete Installation Script
# Installs vLLM + bore.pub heartbeat miner in one command

set -e

echo "======================================"
echo "🚀 MIA GPU Miner Complete Installer"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Configuration
INSTALL_DIR="/data/qwen-awq-miner"
MODEL_DIR="/data/models"
VENV_DIR="$INSTALL_DIR/.venv"
BACKEND_URL="${BACKEND_URL:-https://mia-backend-production.up.railway.app}"

echo "📋 Configuration:"
echo "  Install directory: $INSTALL_DIR"
echo "  Model directory: $MODEL_DIR"
echo "  Backend URL: $BACKEND_URL"
echo ""

# Create directories
echo "📁 Creating directories..."
mkdir -p $INSTALL_DIR
mkdir -p $MODEL_DIR
cd $INSTALL_DIR

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git wget curl > /dev/null 2>&1

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install Python packages
echo "📚 Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet vllm flask waitress aiohttp requests

# Download model if not exists
if [ ! -d "$MODEL_DIR/Qwen2.5-7B-Instruct-AWQ" ]; then
    echo "🤖 Downloading Qwen 7B AWQ model..."
    cd $MODEL_DIR
    git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ
    cd $INSTALL_DIR
else
    echo "✓ Model already exists"
fi

# Install bore client
echo "🌐 Installing bore client..."
if [ ! -f "/usr/local/bin/bore" ]; then
    cd /tmp
    wget -q https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    tar -xzf bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    mv bore /usr/local/bin/
    rm bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    cd $INSTALL_DIR
fi

# Create vLLM start script
cat > $INSTALL_DIR/start_vllm.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner
source .venv/bin/activate

# Stop any existing vLLM
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

echo "Starting vLLM server..."
# Force legacy engine to avoid v1 engine issues
export VLLM_USE_V1=0
nohup python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 > vllm.log 2>&1 &

echo $! > vllm.pid
echo "vLLM started with PID $(cat vllm.pid)"
echo "Logs: tail -f vllm.log"
EOF
chmod +x $INSTALL_DIR/start_vllm.sh

# Create heartbeat miner with bore and auto-restart
cat > $INSTALL_DIR/mia_miner_heartbeat.py << 'EOF'
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
chmod +x $INSTALL_DIR/mia_miner_heartbeat.py

# Create unified start script
cat > $INSTALL_DIR/start_mia_gpu.sh << 'EOF'
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
./start_vllm.sh

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
echo "  - vLLM logs: tail -f vllm.log"
echo "  - Miner logs: tail -f heartbeat.out"
echo "  - Stop all: pkill -f vllm && kill \$(cat heartbeat.pid)"
echo ""
echo "Your GPU will:"
echo "1. Register with MIA backend"
echo "2. Get a public bore.pub URL"
echo "3. Receive AI jobs via push architecture"
echo "4. Process with vLLM (Qwen 7B AWQ)"
EOF
chmod +x $INSTALL_DIR/start_mia_gpu.sh

# Create systemd service (optional)
cat > /etc/systemd/system/mia-gpu-miner.service << EOF
[Unit]
Description=MIA GPU Miner
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/start_mia_gpu.sh
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "✅ Installation Complete!"
echo "======================================"
echo ""
echo "📌 Quick Start:"
echo "  cd $INSTALL_DIR"
echo "  ./start_mia_gpu.sh"
echo ""
echo "📌 Auto-start on boot (optional):"
echo "  systemctl enable mia-gpu-miner"
echo "  systemctl start mia-gpu-miner"
echo ""
echo "📌 One-line installer for next time:"
echo "  curl -sL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-mia-gpu-miner.sh | sudo bash"
echo ""

# Ask if user wants to start now
read -p "Start MIA GPU miner now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd $INSTALL_DIR
    ./start_mia_gpu.sh
fi