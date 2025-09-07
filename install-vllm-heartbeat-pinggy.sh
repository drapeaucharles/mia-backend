#!/bin/bash
# Install heartbeat miner using Pinggy (no token needed!)

echo "🌐 Setting up Pinggy for heartbeat miner"
echo "======================================"

cd /data/qwen-awq-miner

# Create heartbeat miner with Pinggy support
cat > mia_miner_heartbeat_pinggy.py << 'EOF'
#!/usr/bin/env python3
"""MIA Heartbeat Miner with Pinggy support"""
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
        logging.FileHandler('miner_heartbeat_pinggy.log')
    ]
)
logger = logging.getLogger(__name__)

# Flask app for receiving pushed work
app = Flask(__name__)

class HeartbeatMinerPinggy:
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "https://mia-backend-production.up.railway.app")
        self.miner_id = None
        self.miner_key = None
        self.is_processing = False
        self.vllm_url = "http://localhost:8000/v1/chat/completions"
        self.heartbeat_interval = 1.0
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pinggy_url = None
        self.pinggy_process = None
        
    def start_pinggy(self):
        """Start Pinggy and get public URL"""
        try:
            logger.info("🌐 Starting Pinggy tunnel...")
            
            # Start pinggy in background, capturing output
            self.pinggy_process = subprocess.Popen(
                ['ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'ServerAliveInterval=30', '-R', '0:localhost:5000', 'a.pinggy.io'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read output to get the URL
            for i in range(30):  # Wait up to 30 seconds
                if self.pinggy_process.stdout:
                    line = self.pinggy_process.stdout.readline()
                    if line:
                        logger.debug(f"Pinggy output: {line.strip()}")
                        
                        # Look for the URL pattern
                        url_match = re.search(r'https://[a-zA-Z0-9-]+\.a\.pinggy\.io', line)
                        if url_match:
                            self.pinggy_url = url_match.group(0)
                            logger.info(f"✅ Pinggy URL: {self.pinggy_url}")
                            return True
                
                time.sleep(1)
                
                # Check if process died
                if self.pinggy_process.poll() is not None:
                    stdout, stderr = self.pinggy_process.communicate()
                    logger.error(f"Pinggy died: stdout={stdout}, stderr={stderr}")
                    return False
                    
            logger.error("Failed to get Pinggy URL after 30 seconds")
            return False
                
        except Exception as e:
            logger.error(f"Failed to start Pinggy: {e}")
            return False
            
    def register(self):
        """Register with backend using Pinggy URL"""
        if not self.pinggy_url:
            logger.error("No Pinggy URL available")
            return False
            
        try:
            hostname = socket.gethostname()
            
            # Parse Pinggy URL to get subdomain
            pinggy_subdomain = self.pinggy_url.replace('https://', '').split('.')[0]
            
            data = {
                "ip_address": pinggy_subdomain,  # Send subdomain as "IP"
                "gpu_info": self.get_gpu_info(),
                "hostname": hostname,
                "backend_type": "vllm-heartbeat-pinggy",
                "capabilities": ["chat", "completion", "tools"],
                "public_url": self.pinggy_url  # Also send full URL
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
                logger.info(f"✅ Registered as miner {self.miner_id} with Pinggy URL")
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
        """Send heartbeat to backend with Pinggy URL"""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    if not self.is_processing and self.pinggy_url:
                        data = {
                            "miner_id": self.miner_id,
                            "status": "available",
                            "timestamp": datetime.utcnow().isoformat(),
                            "port": 443,  # Pinggy uses HTTPS
                            "public_url": self.pinggy_url
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
                "pinggy_url": self.pinggy_url
            })
        
        # Run Flask in production mode
        from waitress import serve
        logger.info("🌐 Starting Flask server on port 5000 (exposed via Pinggy)")
        serve(app, host='0.0.0.0', port=5000, threads=4)
    
    def run(self):
        """Main run loop"""
        # Start Pinggy first
        if not self.start_pinggy():
            logger.error("Failed to start Pinggy, exiting")
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
        logger.info("💓 Starting heartbeat loop with Pinggy URL")
        try:
            asyncio.run(self.send_heartbeat())
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            if self.pinggy_process:
                self.pinggy_process.terminate()

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
    miner = HeartbeatMinerPinggy()
    
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
chmod +x mia_miner_heartbeat_pinggy.py

# Create start script
cat > start_heartbeat_pinggy.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

# Stop old miners
for pid_file in miner.pid heartbeat_miner.pid heartbeat_ngrok.pid; do
    if [ -f "$pid_file" ]; then
        echo "Stopping old miner..."
        kill $(cat $pid_file) 2>/dev/null || true
        rm -f $pid_file
    fi
done

# Kill any existing tunnels
pkill -f "ssh.*pinggy" 2>/dev/null || true
pkill ngrok 2>/dev/null || true
sleep 2

# Start new heartbeat miner with Pinggy
echo "Starting heartbeat miner with Pinggy..."
nohup python3 mia_miner_heartbeat_pinggy.py > heartbeat_pinggy.out 2>&1 &
echo $! > heartbeat_pinggy.pid
echo "Heartbeat miner (Pinggy) started with PID $(cat heartbeat_pinggy.pid)"
echo ""
echo "Logs: tail -f heartbeat_pinggy.out"
EOF

chmod +x start_heartbeat_pinggy.sh

echo ""
echo "✅ Pinggy heartbeat miner installed!"
echo ""
echo "To start:"
echo "  ./start_heartbeat_pinggy.sh"
echo ""
echo "This will:"
echo "1. Start Pinggy tunnel (NO token needed!)"
echo "2. Register with backend using Pinggy URL"
echo "3. Receive pushed work via Pinggy tunnel"
echo ""
echo "📌 Advantages of Pinggy:"
echo "  - No authentication token required"
echo "  - Works immediately"
echo "  - Stable connections"
echo "  - Free tier with good limits"