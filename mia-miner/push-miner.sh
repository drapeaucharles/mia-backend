#!/bin/bash

# Push-based miner - receives jobs directly from backend

cat > /opt/mia-gpu-miner/push_miner.py << 'EOF'
#!/usr/bin/env python3
"""
Push-based MIA GPU Miner
Receives jobs directly from backend for instant processing
"""

import os, time, requests, logging, socket, threading
from flask import Flask, request, jsonify
import uvicorn
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MIA_URL = "https://mia-backend-production.up.railway.app"
VLLM_URL = "http://localhost:8000"
MINER_PORT = 8001

# Flask app for receiving jobs
app = Flask(__name__)

class PushMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        self.processing = False
        
    def register(self):
        try:
            # Get public IP
            public_ip = requests.get('https://api.ipify.org', timeout=10).text
            hostname = socket.gethostname()
            
            data = {
                "name": f"PUSH-{hostname}-{public_ip.split('.')[-1]}",
                "wallet_address": os.getenv("WALLET_ADDRESS", "0x1234567890123456789012345678901234567890"),
                "endpoint_url": f"http://{public_ip}:{MINER_PORT}",
                "model": "Mistral-7B-OpenOrca-GPTQ",
                "max_tokens": 4096
            }
            
            logger.info(f"Registering push miner at {data['endpoint_url']}...")
            r = self.session.post(f"{MIA_URL}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                self.miner_id = r.json().get("id", r.json().get("miner_id"))
                logger.info(f"✅ Registered! ID: {self.miner_id}")
                
                # Start heartbeat thread
                threading.Thread(target=self.heartbeat_loop, daemon=True).start()
                return True
            else:
                logger.error(f"Registration failed: {r.status_code} - {r.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def heartbeat_loop(self):
        """Send heartbeat to backend every 30 seconds"""
        while True:
            try:
                self.session.post(
                    f"{MIA_URL}/miner/{self.miner_id}/status",
                    json={"status": "idle" if not self.processing else "active"},
                    timeout=5
                )
            except:
                pass
            time.sleep(30)
    
    def process_job(self, job_data):
        """Process a job received from backend"""
        self.processing = True
        start_time = time.time()
        
        try:
            job_id = job_data.get("job_id", job_data.get("request_id"))
            prompt = job_data.get("prompt", "")
            
            logger.info(f"⚡ Processing job {job_id} - INSTANT DISPATCH!")
            
            # Call vLLM
            gen_response = self.session.post(
                f"{VLLM_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": job_data.get("max_tokens", 500),
                    "temperature": job_data.get("temperature", 0.7)
                },
                timeout=120
            )
            
            if gen_response.status_code == 200:
                result = gen_response.json()
                processing_time = time.time() - start_time
                
                # Submit result back to backend
                submit_response = self.session.post(
                    f"{MIA_URL}/submit_result",
                    json={
                        "miner_id": self.miner_id,
                        "request_id": job_id,
                        "result": {
                            "response": result.get("text", ""),
                            "tokens_generated": result.get("tokens_generated", 0),
                            "processing_time": processing_time,
                            "model": result.get("model", "")
                        }
                    },
                    timeout=30
                )
                
                if submit_response.status_code == 200:
                    logger.info(f"✅ Completed {job_id} in {processing_time:.1f}s")
                    return True
                else:
                    logger.error(f"Failed to submit result: {submit_response.status_code}")
                    return False
            else:
                logger.error(f"vLLM generation failed: {gen_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return False
        finally:
            self.processing = False

# Global miner instance
miner = PushMiner()

@app.route('/receive_job', methods=['POST'])
def receive_job():
    """Endpoint to receive jobs from backend"""
    try:
        job_data = request.json
        logger.info(f"Received job: {job_data.get('job_id', 'unknown')}")
        
        # Process in background thread to respond quickly
        threading.Thread(
            target=miner.process_job,
            args=(job_data,),
            daemon=True
        ).start()
        
        return jsonify({"status": "accepted"}), 200
        
    except Exception as e:
        logger.error(f"Error receiving job: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "miner_id": miner.miner_id,
        "processing": miner.processing
    })

def wait_for_vllm():
    """Wait for vLLM server to be ready"""
    logger.info("Waiting for vLLM server...")
    for i in range(60):
        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=2)
            if r.status_code == 200:
                logger.info("✓ vLLM server ready!")
                return True
        except:
            pass
        time.sleep(2)
    logger.error("vLLM server not responding!")
    return False

def main():
    # Wait for vLLM
    if not wait_for_vllm():
        return
    
    # Register miner
    while not miner.register():
        logger.info("Retrying registration in 30s...")
        time.sleep(30)
    
    # Start Flask server to receive jobs
    logger.info(f"🚀 Push-based miner listening on port {MINER_PORT}")
    logger.info("Jobs will be received instantly from backend!")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=MINER_PORT, debug=False)

if __name__ == "__main__":
    main()
EOF

# Create combined vLLM + Push Miner script
cat > /opt/mia-gpu-miner/start_push_miner.sh << 'EOF'
#!/bin/bash

cd /opt/mia-gpu-miner
source venv/bin/activate

# Install Flask if not present
pip install flask 2>/dev/null

# Kill existing processes
pkill -f vllm_server.py || true
pkill -f push_miner.py || true
pkill -f miner_final.py || true
sleep 2

# Start vLLM in background
echo "Starting vLLM server..."
nohup python vllm_server.py > vllm.log 2>&1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM to load model..."
sleep 30

# Check if vLLM is running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM failed to start! Check vllm.log"
    exit 1
fi

# Start push miner
echo "Starting push-based miner..."
python push_miner.py
EOF

chmod +x /opt/mia-gpu-miner/push_miner.py
chmod +x /opt/mia-gpu-miner/start_push_miner.sh

echo "✅ Push-based miner created!"
echo ""
echo "This miner receives jobs instantly from the backend!"
echo "No more polling delays!"
echo ""
echo "To run:"
echo "cd /opt/mia-gpu-miner && ./start_push_miner.sh"