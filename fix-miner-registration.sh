#!/bin/bash

# Quick fix for miner registration issue

cat > /opt/mia-gpu-miner/gpu_miner_fixed.py << 'EOF'
#!/usr/bin/env python3
import os
import time
import requests
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIA_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend.up.railway.app")
VLLM_URL = "http://localhost:8000"

class MIAMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
    
    def wait_vllm(self):
        logger.info("Checking vLLM server...")
        for i in range(60):
            try:
                r = requests.get(f"{VLLM_URL}/health")
                if r.status_code == 200:
                    logger.info(f"✓ vLLM ready with model: {r.json()['model']}")
                    return True
            except:
                pass
            time.sleep(2)
        logger.error("vLLM server not responding")
        return False
    
    def register(self):
        try:
            # Get public IP
            logger.info("Getting public IP...")
            ip = requests.get('https://api.ipify.org', timeout=10).text
            logger.info(f"Public IP: {ip}")
            
            # Prepare registration data
            wallet = os.getenv("WALLET_ADDRESS", "0x" + "0"*40)
            data = {
                "wallet_address": wallet,
                "endpoint_url": f"http://{ip}:8000",
                "model": "Mistral-7B-OpenOrca-GPTQ",
                "max_tokens": 4096
            }
            
            logger.info(f"Registering miner...")
            logger.info(f"  Wallet: {wallet}")
            logger.info(f"  Endpoint: http://{ip}:8000")
            logger.info(f"  Backend: {MIA_URL}")
            
            # Send registration
            r = self.session.post(
                f"{MIA_URL}/miner/register", 
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            logger.info(f"Response status: {r.status_code}")
            
            if r.status_code == 422:
                logger.error(f"Validation error: {r.text}")
                return False
            elif r.status_code != 200:
                logger.error(f"Registration failed: {r.status_code} - {r.text}")
                return False
            
            # Parse response
            try:
                response_data = r.json()
            except:
                logger.error(f"Invalid JSON response: {r.text}")
                return False
            
            # Extract miner ID
            if isinstance(response_data, dict):
                if "miner_id" in response_data:
                    self.miner_id = response_data["miner_id"]
                elif "id" in response_data:
                    self.miner_id = response_data["id"]
                else:
                    logger.error(f"No miner ID in response: {response_data}")
                    return False
            else:
                logger.error(f"Unexpected response format: {response_data}")
                return False
            
            logger.info(f"✅ Registered successfully! Miner ID: {self.miner_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during registration: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}")
            return False
    
    def run(self):
        logger.info("MIA GPU Miner starting...")
        
        if not self.wait_vllm():
            return
        
        # Try to register
        attempts = 0
        while not self.register():
            attempts += 1
            if attempts > 5:
                logger.error("Failed to register after 5 attempts. Exiting.")
                return
            logger.info(f"Retrying registration in 30 seconds... (attempt {attempts}/5)")
            time.sleep(30)
        
        logger.info("Starting work polling loop...")
        errors = 0
        
        while True:
            try:
                # Poll for work
                r = self.session.get(
                    f"{MIA_URL}/miner/{self.miner_id}/work",
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    if work.get("request_id"):
                        logger.info(f"📥 Received work: {work['request_id']}")
                        
                        # Process request
                        gen_response = self.session.post(
                            f"{VLLM_URL}/generate",
                            json={
                                "prompt": work.get("prompt", ""),
                                "max_tokens": work.get("max_tokens", 500),
                                "temperature": work.get("temperature", 0.7)
                            },
                            timeout=120
                        )
                        
                        if gen_response.status_code == 200:
                            result = gen_response.json()
                            
                            # Submit result
                            submit_response = self.session.post(
                                f"{MIA_URL}/miner/{self.miner_id}/submit",
                                json={
                                    "request_id": work["request_id"],
                                    "result": {
                                        "success": True,
                                        "response": result.get("text", ""),
                                        "tokens_generated": result.get("tokens_generated", 0),
                                        "model": result.get("model", "Mistral-7B-OpenOrca-GPTQ")
                                    }
                                },
                                timeout=30
                            )
                            
                            if submit_response.status_code == 200:
                                logger.info(f"✅ Completed request {work['request_id']}")
                            else:
                                logger.error(f"Failed to submit result: {submit_response.status_code}")
                        else:
                            logger.error(f"Generation failed: {gen_response.status_code}")
                
                errors = 0  # Reset error counter on success
                
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
                break
            except Exception as e:
                errors += 1
                logger.error(f"Error in work loop: {e}")
                if errors > 10:
                    logger.error("Too many consecutive errors. Exiting.")
                    break
                time.sleep(min(errors * 5, 60))
            
            time.sleep(5)

if __name__ == "__main__":
    miner = MIAMiner()
    miner.run()
EOF

chmod +x /opt/mia-gpu-miner/gpu_miner_fixed.py

echo "Fixed miner created. To run it:"
echo "cd /opt/mia-gpu-miner"
echo "source venv/bin/activate"
echo "python gpu_miner_fixed.py"