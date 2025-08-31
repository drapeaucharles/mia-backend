#!/usr/bin/env python3
"""
MIA GPU Miner with OpenAI-Compatible Tool Handling
Polls MIA backend for jobs but processes OpenAI-format tools
"""
import json
import logging
import os
import time
import socket
import requests
from typing import Dict, List, Optional, Any
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('miner.log')
    ]
)
logger = logging.getLogger(__name__)

# MIA Backend configuration
MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
MINER_NAME = os.getenv("MINER_NAME", f"gpu-miner-{socket.gethostname()}")

# vLLM server URL (local)
VLLM_URL = "http://localhost:8000/v1"

class OpenAICompatibleMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
        self.vllm_process = None
        
    def start_vllm_server(self):
        """Start the local vLLM OpenAI server if not running"""
        try:
            # Check if already running
            response = requests.get(f"{VLLM_URL}/models", timeout=2)
            if response.status_code == 200:
                logger.info("vLLM server already running")
                return True
        except:
            pass
            
        logger.info("Starting vLLM OpenAI server...")
        # Start using the script we created
        if os.path.exists("start_vllm_openai.sh"):
            subprocess.run(["./start_vllm_openai.sh"], check=False)
            time.sleep(10)  # Wait for startup
            
        # Verify it's running
        try:
            response = requests.get(f"{VLLM_URL}/models", timeout=5)
            if response.status_code == 200:
                logger.info("vLLM server started successfully")
                return True
        except:
            logger.error("Failed to start vLLM server")
            return False
            
    def register_miner(self):
        """Register with MIA backend"""
        try:
            # Get GPU info
            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                text=True
            ).strip().split(', ')
            gpu_name = gpu_info[0]
            gpu_memory = int(gpu_info[1])
            
            data = {
                "name": MINER_NAME,
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "gpu_name": gpu_name,
                "gpu_memory_mb": gpu_memory,
                "status": "idle",
                "capabilities": ["openai_tools", "qwen2.5-7b-awq"]  # Advertise OpenAI support
            }
            
            response = self.session.post(f"{MIA_BACKEND_URL}/register_miner", json=data)
            if response.status_code == 200:
                result = response.json()
                self.miner_id = result.get("miner_id")
                logger.info(f"Registered with MIA backend. Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
            
    def get_next_job(self) -> Optional[Dict]:
        """Poll for next job from MIA backend"""
        if not self.miner_id:
            return None
            
        try:
            response = self.session.get(
                f"{MIA_BACKEND_URL}/job/next",
                params={"miner_id": self.miner_id}
            )
            
            if response.status_code == 200:
                job = response.json()
                if job and job.get('request_id'):
                    return job
            elif response.status_code != 204:  # 204 = no jobs
                logger.error(f"Error getting job: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error polling for jobs: {e}")
            
        return None
        
    def process_openai_job(self, job: Dict) -> Dict:
        """Process job using OpenAI format"""
        try:
            # Extract OpenAI-format request from job
            if job.get('openai_format'):
                # Job already in OpenAI format
                request_data = job['openai_format']
            else:
                # Convert legacy format to OpenAI format
                request_data = self.convert_to_openai_format(job)
                
            # Call vLLM OpenAI API
            response = requests.post(
                f"{VLLM_URL}/chat/completions",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response and tool calls
                choice = result.get('choices', [{}])[0]
                message = choice.get('message', {})
                
                return {
                    'success': True,
                    'response': message.get('content', ''),
                    'tool_calls': message.get('tool_calls', []),
                    'usage': result.get('usage', {}),
                    'model': result.get('model', 'qwen2.5-7b-awq')
                }
            else:
                logger.error(f"vLLM API error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def convert_to_openai_format(self, job: Dict) -> Dict:
        """Convert legacy job format to OpenAI format"""
        messages = []
        
        # Add system message if context provided
        context = job.get('context', {})
        if context.get('system_prompt'):
            messages.append({
                "role": "system",
                "content": context['system_prompt']
            })
        elif context.get('business_name'):
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant for {context['business_name']}."
            })
            
        # Add user message
        messages.append({
            "role": "user",
            "content": job.get('prompt', '')
        })
        
        # Build request
        request = {
            "model": "qwen2.5-7b-instruct-awq",
            "messages": messages,
            "temperature": job.get('temperature', 0.7),
            "max_tokens": job.get('max_tokens', 500)
        }
        
        # Add tools if provided
        if job.get('tools'):
            # Convert tool format if needed
            openai_tools = []
            for tool in job['tools']:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get('name'),
                        "description": tool.get('description', ''),
                        "parameters": tool.get('parameters', {})
                    }
                })
            request['tools'] = openai_tools
            
            # Force tool if specified
            if job.get('force_tool'):
                request['tool_choice'] = {
                    "type": "function",
                    "function": {"name": job['force_tool']}
                }
                
        return request
        
    def submit_result(self, job_id: str, session_id: str, result: Dict):
        """Submit job result to MIA backend"""
        try:
            # Calculate tokens
            tokens = 0
            if result.get('usage'):
                tokens = result['usage'].get('completion_tokens', 0)
            elif result.get('response'):
                # Estimate tokens
                tokens = len(result['response'].split()) * 1.3
                
            data = {
                "job_id": job_id,
                "session_id": session_id,
                "output": result.get('response', ''),
                "miner_id": str(self.miner_id),
                "success": result.get('success', False),
                "tokens_generated": int(tokens),
                "tool_calls": result.get('tool_calls', []),  # Include tool calls
                "model": result.get('model', 'qwen2.5-7b-awq')
            }
            
            response = self.session.post(f"{MIA_BACKEND_URL}/job/result", json=data)
            if response.status_code == 200:
                logger.info(f"Submitted result for job {job_id}")
            else:
                logger.error(f"Failed to submit result: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error submitting result: {e}")
            
    def update_status(self, status: str):
        """Update miner status"""
        if not self.miner_id:
            return
            
        try:
            response = self.session.post(
                f"{MIA_BACKEND_URL}/miner/{self.miner_id}/status",
                json={"status": status}
            )
            if response.status_code != 200:
                logger.error(f"Failed to update status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            
    def run(self):
        """Main miner loop"""
        logger.info("Starting OpenAI-compatible MIA miner...")
        
        # Start vLLM server
        if not self.start_vllm_server():
            logger.error("Failed to start vLLM server. Exiting.")
            return
            
        # Register with MIA backend
        registered = False
        while not registered:
            registered = self.register_miner()
            if not registered:
                logger.info("Retrying registration in 10 seconds...")
                time.sleep(10)
                
        # Main processing loop
        logger.info("Starting job processing loop...")
        while True:
            try:
                # Get next job
                job = self.get_next_job()
                
                if job:
                    job_id = job['request_id']
                    logger.info(f"Processing job {job_id}")
                    
                    # Update status to busy
                    self.update_status("busy")
                    
                    # Process job using OpenAI format
                    result = self.process_openai_job(job)
                    
                    # Submit result
                    self.submit_result(
                        job_id,
                        job.get('session_id', ''),
                        result
                    )
                    
                    # Update status back to idle
                    self.update_status("idle")
                    
                else:
                    # No jobs, wait a bit
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)
                
        # Cleanup
        self.update_status("offline")
        logger.info("Miner shutdown complete")

if __name__ == "__main__":
    miner = OpenAICompatibleMiner()
    miner.run()