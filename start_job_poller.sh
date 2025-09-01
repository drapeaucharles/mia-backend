#!/bin/bash
# Start the MIA job polling wrapper for vLLM

cd /data/qwen-awq-miner

# First ensure vLLM is running
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM server first..."
    ./start_vllm.sh
    sleep 10
fi

# Now start the job poller in the same venv
source .venv/bin/activate

# Install requests if not already installed
pip install requests 2>/dev/null || true

# Create the job polling script
cat > job_poller.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Job Poller - Polls jobs from MIA backend and uses local vLLM
"""
import os
import sys
import json
import time
import logging
import requests
import socket
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-job-poller')

# Configuration
MIA_BACKEND_URL = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
VLLM_URL = 'http://localhost:8000/v1'  # Local vLLM
MINER_ID = f"vllm-qwen-{socket.gethostname()}"

class JobPoller:
    def __init__(self):
        self.session = requests.Session()
        
    def get_work(self) -> Optional[Dict]:
        """Poll for work from MIA backend"""
        try:
            response = self.session.get(
                f"{MIA_BACKEND_URL}/get_work",
                params={"miner_id": MINER_ID},
                timeout=30
            )
            
            if response.status_code == 200:
                work = response.json()
                if work and work.get('request_id'):
                    return work
                    
        except requests.exceptions.Timeout:
            pass  # Normal timeout, no work
        except Exception as e:
            logger.error(f"Error getting work: {e}")
            
        return None
    
    def process_job(self, job: Dict) -> Dict:
        """Process job using local vLLM"""
        try:
            # Build request for vLLM
            messages = []
            
            # Add system message if provided
            context = job.get('context', {})
            if context.get('system_prompt'):
                messages.append({"role": "system", "content": context['system_prompt']})
            elif context.get('business_name'):
                messages.append({"role": "system", "content": f"You are a helpful assistant at {context['business_name']}."})
            
            # Add user message
            messages.append({"role": "user", "content": job.get('prompt', '')})
            
            # Prepare vLLM request
            vllm_request = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": messages,
                "max_tokens": job.get('max_tokens', 150),
                "temperature": job.get('temperature', 0.7)
            }
            
            # Add tools if provided
            if job.get('tools'):
                vllm_request['tools'] = job['tools']
                vllm_request['tool_choice'] = job.get('tool_choice', 'auto')
            
            # Call vLLM
            start_time = time.time()
            response = self.session.post(
                f"{VLLM_URL}/chat/completions",
                json=vllm_request,
                timeout=60
            )
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                return {
                    'error': f"vLLM error: {response.status_code}",
                    'processing_time': processing_time
                }
            
            # Parse response
            vllm_result = response.json()
            choice = vllm_result['choices'][0]
            message = choice['message']
            
            # Build result
            result = {
                'response': message.get('content', ''),
                'tokens_generated': vllm_result.get('usage', {}).get('completion_tokens', 0),
                'processing_time': processing_time,
                'model': 'Qwen/Qwen2.5-7B-Instruct-AWQ'
            }
            
            # Include tool calls if present
            if 'tool_calls' in message and message['tool_calls']:
                tool_call = message['tool_calls'][0]
                result['tool_call'] = {
                    'name': tool_call['function']['name'],
                    'parameters': json.loads(tool_call['function']['arguments'])
                }
                result['requires_tool_execution'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def submit_result(self, request_id: str, result: Dict) -> bool:
        """Submit result to MIA backend"""
        try:
            response = self.session.post(
                f"{MIA_BACKEND_URL}/submit_result",
                json={
                    'miner_id': MINER_ID,
                    'request_id': request_id,
                    'result': result
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Submitted result for {request_id}")
                return True
            else:
                logger.error(f"Submit failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error submitting result: {e}")
            
        return False
    
    def register(self):
        """Register with MIA backend"""
        try:
            data = {
                "miner_id": MINER_ID,
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "endpoint": VLLM_URL,
                "features": ["tool_calling", "fast_inference"],
                "max_tokens": 12288
            }
            
            response = self.session.post(
                f"{MIA_BACKEND_URL}/register_miner",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Registered as {MINER_ID}")
                return True
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
        
        return False
    
    def run(self):
        """Main polling loop"""
        logger.info(f"Starting MIA Job Poller")
        logger.info(f"Backend: {MIA_BACKEND_URL}")
        logger.info(f"vLLM: {VLLM_URL}")
        logger.info(f"Miner ID: {MINER_ID}")
        
        # Test vLLM
        try:
            test = self.session.get(f"{VLLM_URL}/models", timeout=5)
            if test.status_code == 200:
                logger.info("✓ vLLM server is running")
            else:
                logger.error("vLLM server not responding")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Cannot connect to vLLM: {e}")
            sys.exit(1)
        
        # Register
        self.register()
        
        # Main loop
        consecutive_errors = 0
        while True:
            try:
                # Get work
                job = self.get_work()
                
                if job:
                    logger.info(f"Processing job: {job['request_id']}")
                    
                    # Process
                    result = self.process_job(job)
                    
                    # Submit
                    if self.submit_result(job['request_id'], result):
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                
                # Brief pause
                time.sleep(1)
                
                # Re-register if errors
                if consecutive_errors > 5:
                    self.register()
                    consecutive_errors = 0
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                consecutive_errors += 1
                time.sleep(5)

if __name__ == "__main__":
    poller = JobPoller()
    poller.run()
EOF

# Start the job poller
echo "Starting job poller..."
nohup python job_poller.py > logs/job_poller.log 2>&1 &
echo $! > job_poller.pid

echo "✅ Job poller started (PID $(cat job_poller.pid))"
echo "📋 Logs: tail -f logs/job_poller.log"
echo ""
echo "The poller will:"
echo "- Poll jobs from $MIA_BACKEND_URL/get_work"
echo "- Send to local vLLM at localhost:8000"
echo "- Submit results to $MIA_BACKEND_URL/submit_result"