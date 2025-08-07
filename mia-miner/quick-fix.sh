#!/bin/bash

# Quick fix for all miner issues - one command solution

# Kill existing processes
pkill -f vllm_server.py
pkill -f gpu_miner.py

# Create working miner
cat > /opt/mia-gpu-miner/miner_final.py << 'EOF'
#!/usr/bin/env python3
import os, time, requests, logging, json, socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MIA_URL = "https://mia-backend-production.up.railway.app"
VLLM_URL = "http://localhost:8000"

class MIAMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
    
    def wait_vllm(self):
        logger.info("Waiting for vLLM server...")
        for i in range(60):
            try:
                r = requests.get(f"{VLLM_URL}/health", timeout=2)
                if r.status_code == 200:
                    logger.info(f"✓ vLLM ready!")
                    return True
            except: pass
            time.sleep(2)
        return False
    
    def register(self):
        try:
            ip = requests.get('https://api.ipify.org', timeout=10).text
            hostname = socket.gethostname()
            
            data = {
                "name": f"MIA-{hostname}-{ip.split('.')[-1]}",
                "wallet_address": os.getenv("WALLET_ADDRESS", "0x1234567890123456789012345678901234567890"),
                "endpoint_url": f"http://{ip}:8000",
                "model": "Mistral-7B-OpenOrca-GPTQ",
                "max_tokens": 4096
            }
            
            logger.info(f"Registering as {data['name']}...")
            r = self.session.post(f"{MIA_URL}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                self.miner_id = r.json().get("id", r.json().get("miner_id"))
                logger.info(f"✅ Registered! ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code} - {r.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def process_work(self, work):
        try:
            logger.info(f"Processing request {work['request_id']}...")
            
            gen = self.session.post(
                f"{VLLM_URL}/generate",
                json={
                    "prompt": work.get("prompt", ""),
                    "max_tokens": work.get("max_tokens", 500),
                    "temperature": work.get("temperature", 0.7)
                },
                timeout=120
            )
            
            if gen.status_code == 200:
                result = gen.json()
                
                submit = self.session.post(
                    f"{MIA_URL}/submit_result",
                    json={
                        "miner_id": self.miner_id,
                        "request_id": work["request_id"],
                        "result": result.get("text", ""),
                        "tokens_generated": len(result.get("text", "").split()),
                        "processing_time": 1.0
                    },
                    timeout=30
                )
                
                if submit.status_code == 200:
                    logger.info(f"✅ Completed {work['request_id']}")
                else:
                    logger.error(f"Submit failed: {submit.status_code}")
            else:
                logger.error(f"Generation failed: {gen.status_code}")
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
    
    def run(self):
        if not self.wait_vllm():
            logger.error("vLLM not responding")
            return
        
        while not self.register():
            logger.info("Retrying in 30s...")
            time.sleep(30)
        
        logger.info("🚀 Mining started! Waiting for work...")
        
        while True:
            try:
                # Get work
                r = self.session.get(f"{MIA_URL}/get_work", params={"miner_id": self.miner_id}, timeout=10)
                
                if r.status_code == 200:
                    work = r.json()
                    if work and work.get("request_id"):
                        self.process_work(work)
                    else:
                        logger.debug("No work available")
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(10)
            
            time.sleep(5)

if __name__ == "__main__":
    logger.info("MIA GPU Miner v2.0")
    MIAMiner().run()
EOF

# Start everything
cd /opt/mia-gpu-miner && source venv/bin/activate && nohup python vllm_server.py > vllm.log 2>&1 & sleep 10 && python miner_final.py