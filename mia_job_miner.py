#!/usr/bin/env python3
"""
MIA Job Polling Miner - Uses vLLM for inference
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
logger = logging.getLogger('mia-job-miner')

# Configuration
MIA_BACKEND_URL = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
VLLM_URL = os.getenv('VLLM_URL', 'http://localhost:8000/v1')
MINER_ID = f"vllm-qwen-{socket.gethostname()}"

class MIAJobMiner:
    def __init__(self):
        self.session = requests.Session()
        self.registered = False
        
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
                logger.info(f"✓ Registered with MIA backend as {MINER_ID}")
                self.registered = True
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
        
        return False
    
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
            # Normal timeout, no work available
            pass
        except Exception as e:
            logger.error(f"Error getting work: {e}")
            
        return None
    
    def process_job(self, job: Dict) -> Dict:
        """Process job using vLLM"""
        try:
            # Build OpenAI-compatible request
            messages = [{"role": "user", "content": job.get('prompt', '')}]
            
            # Add system prompt from context if provided
            context = job.get('context', {})
            if context.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": context['system_prompt']})
            elif context.get('business_name'):
                system_msg = f"You are a helpful assistant at {context['business_name']}."
                messages.insert(0, {"role": "system", "content": system_msg})
            
            # Prepare request
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
                logger.error(f"vLLM error: {response.status_code} - {response.text}")
                return {
                    'error': f"vLLM error: {response.status_code}",
                    'processing_time': processing_time
                }
            
            # Parse response
            vllm_result = response.json()
            choice = vllm_result['choices'][0]
            message = choice['message']
            
            # Build result for MIA backend
            result = {
                'response': message.get('content', ''),
                'tokens_generated': vllm_result.get('usage', {}).get('completion_tokens', 0),
                'processing_time': processing_time,
                'model': vllm_result.get('model', 'unknown')
            }
            
            # Include tool calls if present
            if 'tool_calls' in message:
                result['tool_call'] = {
                    'name': message['tool_calls'][0]['function']['name'],
                    'parameters': json.loads(message['tool_calls'][0]['function']['arguments'])
                }
                result['requires_tool_execution'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time
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
                logger.error(f"Submit failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error submitting result: {e}")
            
        return False
    
    def run(self):
        """Main mining loop"""
        logger.info(f"Starting MIA Job Miner")
        logger.info(f"MIA Backend: {MIA_BACKEND_URL}")
        logger.info(f"vLLM Server: {VLLM_URL}")
        logger.info(f"Miner ID: {MINER_ID}")
        
        # Test vLLM connection
        try:
            test_response = self.session.get(f"{VLLM_URL}/models", timeout=5)
            if test_response.status_code == 200:
                logger.info("✓ vLLM server is accessible")
            else:
                logger.error("✗ vLLM server not responding correctly")
                sys.exit(1)
        except Exception as e:
            logger.error(f"✗ Cannot connect to vLLM server: {e}")
            sys.exit(1)
        
        # Register with backend
        if not self.register():
            logger.warning("Failed to register, will retry during polling")
        
        # Main loop
        consecutive_errors = 0
        while True:
            try:
                # Get work
                job = self.get_work()
                
                if job:
                    logger.info(f"Got job: {job['request_id']}")
                    
                    # Process job
                    result = self.process_job(job)
                    
                    # Submit result
                    if self.submit_result(job['request_id'], result):
                        logger.info(f"Completed job: {job['request_id']}")
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                
                # Brief pause between polls
                time.sleep(1)
                
                # Re-register if needed
                if not self.registered or consecutive_errors > 5:
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
    miner = MIAJobMiner()
    miner.run()