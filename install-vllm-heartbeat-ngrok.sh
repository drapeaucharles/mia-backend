#!/bin/bash
# Install ngrok and update heartbeat miner to use it

echo "🌐 Setting up ngrok for heartbeat miner"
echo "======================================"

cd /data/qwen-awq-miner

# Install ngrok if not already installed
if ! command -v ngrok >/dev/null 2>&1; then
    echo "📦 Installing ngrok..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update -qq && sudo apt install -y ngrok
fi

# Create ngrok config (free tier)
cat > ngrok.yml << 'EOF'
version: 2
tunnels:
  miner:
    proto: http
    addr: 5000
EOF

# Create modified heartbeat miner that gets ngrok URL
cat > mia_miner_heartbeat_ngrok.py << 'EOF'
#!/usr/bin/env python3
"""MIA Heartbeat Miner with ngrok support"""
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('miner_heartbeat_ngrok.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app for receiving pushed work
app = Flask(__name__)

class HeartbeatMinerNgrok:
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "https://mia-backend-production.up.railway.app")
        self.miner_id = None
        self.miner_key = None
        self.is_processing = False
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        self.heartbeat_interval = 1.0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.ngrok_url = None
        
    def start_ngrok(self):
        """Start ngrok and get public URL"""
        try:
            # Start ngrok in background
            logger.info("🌐 Starting ngrok...")
            subprocess.Popen(['ngrok', 'http', '5000', '--log=stdout'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for ngrok to start
            time.sleep(3)
            
            # Get public URL from ngrok API
            try:
                response = requests.get('http://localhost:4040/api/tunnels')
                data = response.json()
                for tunnel in data['tunnels']:
                    if tunnel['proto'] == 'https':
                        self.ngrok_url = tunnel['public_url']
                        logger.info(f"✅ ngrok URL: {self.ngrok_url}")
                        return True
            except:
                logger.error("Failed to get ngrok URL")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            return False
            
    def register(self):
        """Register with backend using ngrok URL"""
        if not self.ngrok_url:
            logger.error("No ngrok URL available")
            return False
            
        try:
            hostname = socket.gethostname()
            
            # Parse ngrok URL to get just the domain
            ngrok_domain = self.ngrok_url.replace('https://', '').split('.ngrok')[0]
            
            data = {
                "ip_address": ngrok_domain,  # Send ngrok domain as "IP"
                "gpu_info": self.get_gpu_info(),
                "hostname": hostname,
                "backend_type": "vllm-heartbeat-ngrok",
                "capabilities": ["chat", "completion", "tools"],
                "public_url": self.ngrok_url  # Also send full URL
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
                logger.info(f"✅ Registered as miner {self.miner_id} with ngrok URL")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code}")
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
        """Send heartbeat to backend with ngrok URL"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    if not self.is_processing and self.ngrok_url:
                        data = {
                            "miner_id": self.miner_id,
                            "status": "available",
                            "timestamp": datetime.utcnow().isoformat(),
                            "port": 443,  # ngrok uses HTTPS
                            "public_url": self.ngrok_url
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
        request_id = job_data.get('request_id')
        
        try:
            logger.info(f"🔧 Processing job {request_id}")
            start_time = time.time()
            
            # Extract prompt and parameters
            prompt = job_data.get('prompt', '')
            messages = job_data.get('messages', [{'role': 'user', 'content': prompt}])
            tools = job_data.get('tools', [])
            
            # Prepare vLLM request
            vllm_request = {
                "model": "/data/models/Qwen2.5-14B-Instruct-AWQ",
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
                "ngrok_url": self.ngrok_url
            })
        
        # Run Flask in production mode
        from waitress import serve
        logger.info("🌐 Starting Flask server on port 5000 (exposed via ngrok)")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    
    def run(self):
        """Main run loop"""
        # Start ngrok first
        if not self.start_ngrok():
            logger.error("Failed to start ngrok, exiting")
            sys.exit(1)
            
        # Start Flask server in separate thread
        flask_thread = threading.Thread(target=self.run_flask_server, daemon=True)
        flask_thread.start()
        
        # Start heartbeat loop
        logger.info("💓 Starting heartbeat loop with ngrok URL")
        asyncio.run(self.send_heartbeat())

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
    miner = HeartbeatMinerNgrok()
    
    # Register with backend
    attempts = 0
    while not miner.register():
        attempts += 1
        if attempts > 5:
            logger.error("Failed to register after 5 attempts")
            sys.exit(1)
        logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
        time.sleep(30)
    
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
chmod +x mia_miner_heartbeat_ngrok.py

# Create start script
cat > start_heartbeat_ngrok.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

# Stop old miners
for pid_file in miner.pid heartbeat_miner.pid; do
    if [ -f "$pid_file" ]; then
        echo "Stopping old miner..."
        kill $(cat $pid_file) 2>/dev/null || true
        rm -f $pid_file
    fi
done

# Kill any existing ngrok
pkill ngrok 2>/dev/null || true
sleep 2

# Start new heartbeat miner with ngrok
echo "Starting heartbeat miner with ngrok..."
nohup python3 mia_miner_heartbeat_ngrok.py > heartbeat_ngrok.out 2>&1 &
echo $! > heartbeat_ngrok.pid
echo "Heartbeat miner (ngrok) started with PID $(cat heartbeat_ngrok.pid)"
echo ""
echo "Logs: tail -f heartbeat_ngrok.out"
echo "ngrok web UI: http://localhost:4040"
EOF

chmod +x start_heartbeat_ngrok.sh

echo ""
echo "✅ ngrok heartbeat miner installed!"
echo ""
echo "To start:"
echo "  ./start_heartbeat_ngrok.sh"
echo ""
echo "This will:"
echo "1. Start ngrok to expose port 5000"
echo "2. Register with backend using ngrok URL"
echo "3. Receive pushed work via ngrok tunnel"
echo ""
echo "⚠️ Note: Free ngrok has limits (40 requests/minute)"