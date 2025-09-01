#!/bin/bash
# Direct installation of polling miner to existing setup

cd /data/qwen-awq-miner || exit 1

# Download the polling miner directly
echo "Downloading job polling miner..."
cat > miner.py << 'MINER_EOF'
#!/usr/bin/env python3
"""
MIA Job Polling Miner - Uses existing vLLM installation
Polls jobs from backend, no API server needed
"""
import os
import sys
import json
import time
import logging
import requests
import socket
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-miner')

# Configuration
MIA_BACKEND_URL = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
VLLM_URL = 'http://localhost:8000/v1'  # Local vLLM server
MINER_ID = int(os.getenv('MINER_ID', '1'))  # Get from environment

class MIAMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = MINER_ID
        
    def get_work(self) -> Optional[Dict]:
        """Poll for work from MIA backend"""
        try:
            response = self.session.get(
                f"{MIA_BACKEND_URL}/get_work",
                params={"miner_id": self.miner_id},
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
    
    def process_with_vllm(self, job: Dict) -> Dict:
        """Process job using local vLLM server"""
        try:
            # Build messages
            messages = []
            
            # Add system message if provided
            context = job.get('context', {})
            if isinstance(context, dict):
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
            logger.info(f"Calling vLLM with{'out' if not job.get('tools') else ''} tools")
            start_time = time.time()
            
            response = self.session.post(
                f"{VLLM_URL}/chat/completions",
                json=vllm_request,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"vLLM error: {response.status_code} - {response.text}")
                return {
                    'response': f"Error: vLLM returned {response.status_code}",
                    'tokens_generated': 0,
                    'processing_time': processing_time
                }
            
            # Parse vLLM response
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
            
            # Handle tool calls if present
            if 'tool_calls' in message and message['tool_calls']:
                tool_call = message['tool_calls'][0]
                result['tool_call'] = {
                    'name': tool_call['function']['name'],
                    'parameters': json.loads(tool_call['function']['arguments'])
                }
                result['requires_tool_execution'] = True
                logger.info(f"Tool call detected: {result['tool_call']['name']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return {
                'response': f"Error: {str(e)}",
                'tokens_generated': 0,
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def submit_result(self, request_id: str, result: Dict) -> bool:
        """Submit result to MIA backend"""
        try:
            data = {
                'miner_id': self.miner_id,
                'request_id': request_id,
                'result': result
            }
            
            response = self.session.post(
                f"{MIA_BACKEND_URL}/submit_result",
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✓ Submitted result for {request_id}")
                return True
            else:
                logger.error(f"Submit failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error submitting result: {e}")
            
        return False
    
    def run(self):
        """Main mining loop"""
        logger.info(f"Starting MIA Miner")
        logger.info(f"Backend: {MIA_BACKEND_URL}")
        logger.info(f"vLLM: {VLLM_URL}")
        logger.info(f"Miner ID: {self.miner_id}")
        
        # Test vLLM connection
        logger.info("Testing vLLM connection...")
        try:
            test = self.session.get(f"{VLLM_URL}/models", timeout=5)
            if test.status_code == 200:
                models = test.json()
                logger.info(f"✓ vLLM is running with model: {models['data'][0]['id']}")
            else:
                logger.error("✗ vLLM server not responding correctly")
                logger.error("Please start vLLM: cd /data/qwen-awq-miner && ./start_vllm.sh")
                sys.exit(1)
        except Exception as e:
            logger.error(f"✗ Cannot connect to vLLM server: {e}")
            logger.error("Please start vLLM: cd /data/qwen-awq-miner && ./start_vllm.sh")
            sys.exit(1)
        
        # Main loop
        jobs_completed = 0
        total_tokens = 0
        consecutive_errors = 0
        
        logger.info("Starting job polling loop...")
        
        while True:
            try:
                # Get work
                job = self.get_work()
                
                if job:
                    request_id = job['request_id']
                    logger.info(f"Got job: {request_id}")
                    
                    # Show job details
                    if job.get('tools'):
                        logger.info(f"  Tools: {len(job['tools'])} available")
                    if job.get('context'):
                        logger.info(f"  Context: {job['context']}")
                    
                    # Process with vLLM
                    result = self.process_with_vllm(job)
                    
                    # Submit result
                    if self.submit_result(request_id, result):
                        jobs_completed += 1
                        total_tokens += result.get('tokens_generated', 0)
                        consecutive_errors = 0
                        
                        logger.info(f"Completed {jobs_completed} jobs, {total_tokens} tokens total")
                    else:
                        consecutive_errors += 1
                else:
                    # No work available, brief pause
                    time.sleep(2)
                
                # Handle persistent errors
                if consecutive_errors > 5:
                    logger.warning("Too many errors, pausing for 30 seconds...")
                    time.sleep(30)
                    consecutive_errors = 0
                    
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                logger.info(f"Completed {jobs_completed} jobs, generated {total_tokens} tokens")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                consecutive_errors += 1
                time.sleep(5)

if __name__ == "__main__":
    miner = MIAMiner()
    miner.run()
MINER_EOF

chmod +x miner.py

# Create start script
cat > start_miner.sh << 'START_EOF'
#!/bin/bash
cd "$(dirname "$0")"

# First ensure vLLM is running
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM server first..."
    ./start_vllm.sh
    echo "Waiting for vLLM to start..."
    sleep 10
fi

# Activate virtual environment
source .venv/bin/activate

# Set environment
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export MIA_BACKEND_URL=${MIA_BACKEND_URL:-https://mia-backend-production.up.railway.app}
export MINER_ID=${MINER_ID:-1}

# Start the polling miner
echo "Starting MIA job polling miner..."
echo "Backend: $MIA_BACKEND_URL"
echo "Miner ID: $MINER_ID"
echo ""

python miner.py 2>&1 | tee -a miner.log
START_EOF
chmod +x start_miner.sh

echo "✅ Polling miner installed!"
echo ""
echo "To start:"
echo "1. Set your miner ID: export MINER_ID=your_id"
echo "2. Run: cd /data/qwen-awq-miner && ./start_miner.sh"