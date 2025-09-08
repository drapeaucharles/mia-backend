#!/bin/bash
# Install heartbeat miner using bore.pub (truly no auth needed!)

echo "🌐 Setting up bore.pub tunnel for heartbeat miner"
echo "================================================"

cd /data/qwen-awq-miner

# Install bore client
echo "📦 Installing bore client..."
if [ ! -f "/usr/local/bin/bore" ]; then
    wget https://github.com/ekzhang/bore/releases/download/v0.5.0/bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    tar -xzf bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
    sudo mv bore /usr/local/bin/
    rm bore-v0.5.0-x86_64-unknown-linux-musl.tar.gz
fi

# Create heartbeat miner with bore support
cat > mia_miner_heartbeat_bore.py << 'EOF'
#!/usr/bin/env python3
"""MIA Heartbeat Miner with bore.pub support"""
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
        logging.FileHandler('miner_heartbeat_bore.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app for receiving pushed work
app = Flask(__name__)

class HeartbeatMinerBore:
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
        
    def start_bore(self):
        """Start bore tunnel and get public URL"""
        try:
            logger.info("🌐 Starting bore tunnel...")
            
            # Start bore client
            # bore local 5000 --to bore.pub
            cmd = ['bore', 'local', '5000', '--to', 'bore.pub']
            
            self.bore_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read output to get the port
            for i in range(30):  # Wait up to 30 seconds
                if self.bore_process.stdout:
                    line = self.bore_process.stdout.readline()
                    if line:
                        logger.info(f"Bore output: {line.strip()}")
                        
                        # Look for port assignment
                        # bore outputs like: "listening at bore.pub:12345"
                        port_match = re.search(r'bore\.pub:(\d+)', line)
                        if port_match:
                            self.bore_port = port_match.group(1)
                            self.public_url = f"http://bore.pub:{self.bore_port}"
                            logger.info(f"✅ Public URL: {self.public_url}")
                            
                            # Continue reading in background
                            threading.Thread(target=self._read_bore_output, daemon=True).start()
                            return True
                
                time.sleep(1)
                
                # Check if process died
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
        """Keep reading bore output"""
        while self.bore_process and self.bore_process.poll() is None:
            try:
                line = self.bore_process.stdout.readline()
                if line:
                    logger.debug(f"Bore: {line.strip()}")
            except:
                break
            
    def register(self):
        """Register with backend using bore URL"""
        if not self.public_url:
            logger.error("No public URL available")
            return False
            
        try:
            hostname = socket.gethostname()
            
            # Send bore.pub URL
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
            
            # Prepare vLLM request
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
            """Endpoint to receive pushed work from backend"""
            try:
                logger.info(f"Received work request")
                
                # Process job asynchronously
                job_data = request.json
                future = self.executor.submit(self.process_job, job_data)
                
                # Get result (with timeout)
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
                "public_url": self.public_url
            })
        
        @app.route('/', methods=['GET'])
        def index():
            return jsonify({
                "service": "MIA GPU Miner",
                "status": "running",
                "public_url": self.public_url,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Run Flask in production mode
        from waitress import serve
        logger.info("🌐 Starting Flask server on port 5000")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    
    def run(self):
        """Main run loop"""
        # Start bore first
        if not self.start_bore():
            logger.error("Failed to start bore tunnel, exiting")
            sys.exit(1)
            
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
        logger.info("💓 Starting heartbeat loop")
        try:
            asyncio.run(self.send_heartbeat())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.bore_process:
                self.bore_process.terminate()

if __name__ == "__main__":
    # Wait for vLLM to be ready
    logger.info("⏳ Waiting for vLLM to start...")
    for i in range(30):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                logger.info("✅ vLLM is ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        logger.error("❌ vLLM failed to start after 60 seconds")
        sys.exit(1)
    
    # Start miner
    miner = HeartbeatMinerBore()
    
    # Run miner
    try:
        miner.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
EOF

# Make executable
chmod +x mia_miner_heartbeat_bore.py

# Create start script
cat > start_heartbeat_bore.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

# Stop old miners
for pid_file in miner.pid heartbeat*.pid; do
    if [ -f "$pid_file" ]; then
        echo "Stopping old miner..."
        kill $(cat $pid_file) 2>/dev/null || true
        rm -f $pid_file
    fi
done

# Kill any existing tunnels
pkill bore 2>/dev/null || true
pkill -f "ssh.*localhost.run" 2>/dev/null || true
pkill -f "ssh.*pinggy" 2>/dev/null || true
pkill ngrok 2>/dev/null || true
sleep 2

# Ensure we're in venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start new heartbeat miner with bore
echo "Starting heartbeat miner with bore.pub..."
nohup python3 mia_miner_heartbeat_bore.py > heartbeat_bore.out 2>&1 &
echo $! > heartbeat_bore.pid
echo "Heartbeat miner (bore) started with PID $(cat heartbeat_bore.pid)"
echo ""
echo "Logs: tail -f heartbeat_bore.out"
EOF

chmod +x start_heartbeat_bore.sh

echo ""
echo "✅ bore.pub heartbeat miner installed!"
echo ""
echo "To start:"
echo "  ./start_heartbeat_bore.sh"
echo ""
echo "This will:"
echo "1. Start bore tunnel (NO authentication needed!)"
echo "2. Register with backend using public URL"
echo "3. Receive pushed work via tunnel"
echo ""
echo "📌 Advantages of bore.pub:"
echo "  - TRULY no authentication required"
echo "  - Works immediately - just run!"
echo "  - Fast and reliable"
echo "  - Free public relay"