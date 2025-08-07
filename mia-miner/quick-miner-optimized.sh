#!/bin/bash

# Optimized miner with instant response time

cat > /opt/mia-gpu-miner/miner_optimized.py << 'EOF'
#!/usr/bin/env python3
import os, time, requests, logging, socket

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MIA_URL = "https://mia-backend-production.up.railway.app"
VLLM_URL = "http://localhost:8000"

class OptimizedMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        self.session.headers.update({'Connection': 'keep-alive'})
    
    def wait_vllm(self):
        logger.info("Checking vLLM server...")
        for i in range(60):
            try:
                r = requests.get(f"{VLLM_URL}/health", timeout=2)
                if r.status_code == 200:
                    logger.info("✓ vLLM ready!")
                    return True
            except: pass
            time.sleep(2)
        return False
    
    def register(self):
        try:
            ip = requests.get('https://api.ipify.org', timeout=10).text
            hostname = socket.gethostname()
            
            data = {
                "name": f"MIA-OPT-{hostname}-{ip.split('.')[-1]}",
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
        start_time = time.time()
        try:
            logger.info(f"⚡ Processing {work['request_id']} instantly!")
            
            # Call vLLM
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
                processing_time = time.time() - start_time
                
                # Submit result
                submit = self.session.post(
                    f"{MIA_URL}/submit_result",
                    json={
                        "miner_id": self.miner_id,
                        "request_id": work["request_id"],
                        "result": {
                            "response": result.get("text", ""),
                            "tokens_generated": result.get("tokens_generated", 0),
                            "processing_time": processing_time
                        }
                    },
                    timeout=30
                )
                
                if submit.status_code == 200:
                    logger.info(f"✅ Completed in {processing_time:.1f}s")
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
        
        logger.info("🚀 Optimized miner started! Using long polling for instant response...")
        consecutive_errors = 0
        
        while True:
            try:
                # Long polling - waits up to 30 seconds for work
                r = self.session.get(
                    f"{MIA_URL}/get_work",
                    params={"miner_id": self.miner_id, "wait": "true"},
                    timeout=35  # Slightly longer than server timeout
                )
                
                if r.status_code == 200:
                    work = r.json()
                    if work and work.get("request_id"):
                        self.process_work(work)
                    # If no work, loop immediately (server already waited)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
                    time.sleep(min(consecutive_errors * 2, 30))
                
            except requests.exceptions.Timeout:
                # Timeout is normal for long polling, just retry
                consecutive_errors = 0
                continue
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error: {e}")
                time.sleep(min(consecutive_errors * 2, 30))

if __name__ == "__main__":
    logger.info("MIA Optimized GPU Miner v3.0")
    OptimizedMiner().run()
EOF

chmod +x /opt/mia-gpu-miner/miner_optimized.py

echo "✓ Optimized miner created!"
echo ""
echo "To run:"
echo "pkill -f miner_final.py"
echo "cd /opt/mia-gpu-miner && source venv/bin/activate && python miner_optimized.py"
echo ""
echo "This version uses long polling for near-instant response times!"