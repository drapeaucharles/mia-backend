#!/bin/bash

# Create a fully working miner with all fixes

cat > /opt/mia-gpu-miner/working_miner.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Fully Working Version
All issues fixed, ready to mine!
"""

import os
import time
import requests
import logging
import json
import socket
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MIA_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000"

class WorkingMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        self.stats = {
            "jobs_completed": 0,
            "total_tokens": 0,
            "start_time": datetime.now()
        }
    
    def wait_vllm(self):
        """Wait for vLLM server to be ready"""
        logger.info("Checking vLLM server...")
        for i in range(60):
            try:
                r = requests.get(f"{VLLM_URL}/health", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    logger.info(f"✓ vLLM ready with model: {data.get('model', 'unknown')}")
                    return True
            except:
                if i % 10 == 0:
                    logger.info(f"Waiting for vLLM... ({i}s)")
            time.sleep(2)
        logger.error("vLLM server not responding!")
        return False
    
    def register(self):
        """Register with MIA backend"""
        try:
            # Get public IP
            public_ip = requests.get('https://api.ipify.org', timeout=10).text
            hostname = socket.gethostname()
            
            data = {
                "name": f"GPU-{hostname}-{public_ip.split('.')[-1]}",
                "wallet_address": os.getenv("WALLET_ADDRESS", "0x1234567890123456789012345678901234567890"),
                "endpoint_url": f"http://{public_ip}:8000",
                "model": "Mistral-7B-OpenOrca-GPTQ",
                "max_tokens": 4096
            }
            
            logger.info(f"Registering miner...")
            logger.info(f"  Name: {data['name']}")
            logger.info(f"  Endpoint: {data['endpoint_url']}")
            
            r = self.session.post(
                f"{MIA_URL}/register_miner",
                json=data,
                timeout=30
            )
            
            if r.status_code == 200:
                resp_data = r.json()
                self.miner_id = resp_data.get("id", resp_data.get("miner_id"))
                logger.info(f"✅ Registered successfully! Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def get_work(self):
        """Get work from backend"""
        try:
            r = self.session.get(
                f"{MIA_URL}/get_work",
                params={"miner_id": self.miner_id},
                timeout=10
            )
            
            if r.status_code == 200:
                data = r.json()
                if data and data.get("request_id"):
                    return data
            return None
            
        except Exception as e:
            logger.error(f"Error getting work: {e}")
            return None
    
    def process_job(self, job):
        """Process a single job"""
        start_time = time.time()
        request_id = job.get("request_id")
        
        try:
            logger.info(f"⚙️  Processing job {request_id}")
            logger.info(f"   Prompt: {job.get('prompt', '')[:100]}...")
            
            # Call vLLM
            vllm_response = self.session.post(
                f"{VLLM_URL}/generate",
                json={
                    "prompt": job.get("prompt", ""),
                    "max_tokens": job.get("max_tokens", 500),
                    "temperature": job.get("temperature", 0.7)
                },
                timeout=120
            )
            
            if vllm_response.status_code != 200:
                logger.error(f"vLLM error: {vllm_response.status_code}")
                return False
            
            vllm_data = vllm_response.json()
            processing_time = time.time() - start_time
            
            # Prepare result in correct format
            result_data = {
                "response": vllm_data.get("text", ""),
                "tokens_generated": vllm_data.get("tokens_generated", 0),
                "processing_time": processing_time,
                "model": vllm_data.get("model", "Mistral-7B-OpenOrca-GPTQ")
            }
            
            # Submit result - FIXED FORMAT
            submit_data = {
                "miner_id": self.miner_id,
                "request_id": request_id,
                "result": result_data  # This is now a dict as expected
            }
            
            logger.debug(f"Submitting: {json.dumps(submit_data, indent=2)}")
            
            submit_response = self.session.post(
                f"{MIA_URL}/submit_result",
                json=submit_data,
                timeout=30
            )
            
            if submit_response.status_code == 200:
                # Update stats
                self.stats["jobs_completed"] += 1
                self.stats["total_tokens"] += result_data["tokens_generated"]
                
                logger.info(f"✅ Completed job {request_id}")
                logger.info(f"   Generated {result_data['tokens_generated']} tokens in {processing_time:.1f}s")
                logger.info(f"   Total jobs: {self.stats['jobs_completed']}")
                return True
            else:
                logger.error(f"❌ Submit failed: {submit_response.status_code}")
                logger.error(f"   Response: {submit_response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Main mining loop"""
        logger.info("=" * 60)
        logger.info("MIA GPU Miner v4.0 - All Issues Fixed!")
        logger.info("=" * 60)
        
        # Wait for vLLM
        if not self.wait_vllm():
            return
        
        # Register
        attempts = 0
        while not self.register():
            attempts += 1
            if attempts > 5:
                logger.error("Failed to register after 5 attempts")
                return
            logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
            time.sleep(30)
        
        logger.info("🚀 Mining started! Waiting for work...")
        logger.info("-" * 60)
        
        consecutive_errors = 0
        no_work_count = 0
        
        while True:
            try:
                # Get work
                job = self.get_work()
                
                if job:
                    no_work_count = 0
                    # Process the job
                    if self.process_job(job):
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                else:
                    no_work_count += 1
                    if no_work_count % 12 == 0:  # Log every minute
                        logger.info(f"No work available... (checked {no_work_count} times)")
                
                # Handle too many errors
                if consecutive_errors > 5:
                    logger.error("Too many consecutive errors, restarting...")
                    time.sleep(60)
                    consecutive_errors = 0
                
                # Sleep between polls
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("\nShutting down gracefully...")
                runtime = datetime.now() - self.stats["start_time"]
                logger.info(f"\nSession stats:")
                logger.info(f"  Runtime: {runtime}")
                logger.info(f"  Jobs completed: {self.stats['jobs_completed']}")
                logger.info(f"  Tokens generated: {self.stats['total_tokens']}")
                if self.stats['jobs_completed'] > 0:
                    logger.info(f"  Avg tokens/job: {self.stats['total_tokens'] / self.stats['jobs_completed']:.1f}")
                break
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Unexpected error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    miner = WorkingMiner()
    miner.run()
EOF

chmod +x /opt/mia-gpu-miner/working_miner.py

echo "✅ Working miner created!"
echo ""
echo "This version has ALL fixes:"
echo "- ✓ Correct submit_result format"
echo "- ✓ Better error handling"
echo "- ✓ Statistics tracking"
echo "- ✓ Graceful shutdown"
echo ""
echo "To run:"
echo "pkill -f 'miner_final.py|gpu_miner_fixed.py'"
echo "cd /opt/mia-gpu-miner && source venv/bin/activate && python working_miner.py"