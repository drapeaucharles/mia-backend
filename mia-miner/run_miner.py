#!/usr/bin/env python3
"""
MIA GPU Miner Client
Polls the MIA backend for jobs and processes them using GPU inference
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Optional, Any
import random
import signal
from fallback_manager import FallbackManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('mia-miner')

class MIAMiner:
    def __init__(self):
        self.api_url = os.environ.get('MIA_API_URL', 'https://mia-backend-production.up.railway.app').rstrip('/')
        self.miner_name = os.environ.get('MINER_NAME', 'gpu-miner-001')
        self.poll_interval = int(os.environ.get('POLL_INTERVAL', '5'))
        self.miner_id = None
        self.auth_key = None
        
        # Network settings
        self.request_timeout = 10
        self.network_retry_delay = 10
        
        # Initialize fallback manager
        self.fallback_manager = FallbackManager(self.miner_name, self.api_url)
        self.fallback_idle_threshold = 10  # Start fallback after 10 seconds of no jobs
        self.last_job_time = time.time()
        self.fallback_active = False
        
        logger.info("=" * 50)
        logger.info("[MIA] Miner initialized")
        logger.info(f"[MIA] API URL: {self.api_url}")
        logger.info(f"[MIA] Miner Name: {self.miner_name}")
        logger.info(f"[MIA] Poll Interval: {self.poll_interval}s")
        logger.info("=" * 50)
        
        # Register miner on startup (with retry)
        self.register_miner_with_retry()
    
    def register_miner_with_retry(self):
        """Register miner with retry logic"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            if self.register_miner():
                return
            
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"[MIA] Registration retry {retry_count}/{max_retries} in {self.network_retry_delay} seconds...")
                time.sleep(self.network_retry_delay)
        
        logger.warning("[MIA] Could not register miner after retries. Continuing anyway...")
    
    def safe_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and clean error handling"""
        kwargs.setdefault('timeout', self.request_timeout)
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = requests.request(method, url, **kwargs)
                return response
            except requests.exceptions.ConnectionError:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"[MIA] Connection failed. Retry {retry_count}/{max_retries} in {self.network_retry_delay} seconds...")
                    time.sleep(self.network_retry_delay)
                else:
                    logger.warning("[MIA] Backend temporarily unavailable")
                    return None
            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"[MIA] Request timed out. Retry {retry_count}/{max_retries}...")
                    time.sleep(2)
                else:
                    logger.warning("[MIA] Request timed out")
                    return None
            except Exception as e:
                logger.warning(f"[MIA] Request error: {type(e).__name__}")
                return None
        
        return None
    
    def register_miner(self) -> bool:
        """Register this miner with the backend"""
        try:
            response = requests.post(
                f"{self.api_url}/register_miner",
                json={"name": self.miner_name},
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.miner_id = data.get('miner_id')
                self.auth_key = data.get('auth_key')
                logger.info(f"[MIA] Miner registered successfully. ID: {self.miner_id}")
                return True
            else:
                logger.warning(f"[MIA] Registration failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.warning("[MIA] Backend temporarily unavailable. Will retry...")
            return False
        except requests.exceptions.Timeout:
            logger.warning("[MIA] Request timed out. Will retry...")
            return False
        except Exception as e:
            logger.warning(f"[MIA] Registration error: {type(e).__name__}")
            return False
    
    def simulate_inference(self, prompt: str, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Simulate model inference (to be replaced with actual Mixtral later)
        """
        # Simulate processing time (1-3 seconds)
        processing_time = random.uniform(1.0, 3.0)
        time.sleep(processing_time)
        
        # Generate dummy response
        responses = [
            "This is a simulated response from the MIA miner. In production, this would be generated by Mixtral.",
            "I'm processing your request on a GPU. This is a test response.",
            "MIA miner here! Your job has been processed successfully.",
            f"Response to '{prompt[:50]}...' - Processing complete!",
            "GPU inference simulation complete. Real Mixtral integration coming soon!"
        ]
        
        response_text = random.choice(responses)
        
        # Simulate token count (roughly 1.3 tokens per word)
        word_count = len(response_text.split())
        output_tokens = int(word_count * 1.3)
        
        return {
            "output": response_text,
            "output_tokens": output_tokens,
            "processing_time": processing_time,
            "gpu_utilization": random.randint(70, 95)  # Simulated GPU usage
        }
    
    def process_mia_job(self, job: Dict[str, Any]) -> bool:
        """Process a MIA job and submit the result"""
        job_id = job.get('job_id')
        prompt = job.get('prompt', '')
        context = job.get('context', '')
        session_id = job.get('session_id')
        
        logger.info(f"[MIA] Processing job {job_id}")
        
        try:
            # Simulate inference
            full_prompt = f"{context}\n{prompt}" if context else prompt
            result = self.simulate_inference(full_prompt)
            
            # Submit result with retry
            response = self.safe_request(
                'POST',
                f"{self.api_url}/job/result",
                json={
                    "job_id": job_id,
                    "session_id": session_id,
                    "output": result["output"],
                    "miner_id": self.miner_id
                }
            )
            
            if response and response.status_code == 200:
                logger.info(f"[MIA] Job {job_id} completed. Tokens: {result['output_tokens']}")
                return True
            else:
                logger.warning(f"[MIA] Could not submit job result")
                return False
                
        except Exception as e:
            logger.warning(f"[MIA] Error processing job: {type(e).__name__}")
            return False
    
    def process_idle_job(self, job: Dict[str, Any]) -> bool:
        """Process an idle job and submit the result"""
        job_id = job.get('job_id')
        prompt = job.get('prompt', '')
        max_tokens = job.get('max_tokens', 500)
        
        logger.info(f"[MIA] Processing idle job {job_id}")
        
        try:
            # Simulate inference
            result = self.simulate_inference(prompt, max_tokens)
            
            # Calculate simulated earnings (matching backend pricing)
            tokens_in_thousands = result["output_tokens"] / 1000
            revenue_usd = round(tokens_in_thousands * 0.001, 6)  # $0.001 per 1K tokens
            
            # Submit result with retry
            response = self.safe_request(
                'POST',
                f"{self.api_url}/idle-job/result",
                json={
                    "job_id": job_id,
                    "output": result["output"],
                    "output_tokens": result["output_tokens"],
                    "usd_earned": revenue_usd,
                    "runpod_job_id": f"sim-{job_id}-{int(time.time())}"
                }
            )
            
            if response and response.status_code == 200:
                logger.info(f"[MIA] Idle job {job_id} completed. Tokens: {result['output_tokens']}, Revenue: ${revenue_usd}")
                return True
            else:
                logger.warning(f"[MIA] Could not submit idle job result")
                return False
                
        except Exception as e:
            logger.warning(f"[MIA] Error processing idle job: {type(e).__name__}")
            return False
    
    def poll_for_jobs(self):
        """Main polling loop"""
        logger.info("[MIA] Starting job polling...")
        consecutive_errors = 0
        
        while True:
            try:
                # First, try to get a MIA job
                response = self.safe_request(
                    'GET',
                    f"{self.api_url}/job/next",
                    params={"miner_id": self.miner_id} if self.miner_id else {}
                )
                
                if response and response.status_code == 200:
                    consecutive_errors = 0
                    job = response.json()
                    
                    if job.get('job_id'):
                        self.last_job_time = time.time()
                        
                        # Stop fallback if running
                        if self.fallback_active:
                            self.fallback_manager.stop_fallback()
                            self.fallback_active = False
                        
                        self.process_mia_job(job)
                        continue  # Skip sleep to process next job quickly
                    else:
                        # No MIA jobs, try idle jobs
                        idle_response = self.safe_request(
                            'GET',
                            f"{self.api_url}/idle-job/next"
                        )
                        
                        if idle_response and idle_response.status_code == 200:
                            idle_job = idle_response.json()
                            
                            if idle_job.get('job_id'):
                                self.last_job_time = time.time()
                                
                                # Stop fallback if running
                                if self.fallback_active:
                                    self.fallback_manager.stop_fallback()
                                    self.fallback_active = False
                                
                                self.process_idle_job(idle_job)
                                continue  # Skip sleep to process next job quickly
                            else:
                                # No jobs available - check if we should start fallback
                                time_since_last_job = time.time() - self.last_job_time
                                
                                if time_since_last_job > self.fallback_idle_threshold and not self.fallback_active:
                                    self.fallback_manager.start_fallback()
                                    self.fallback_active = True
                elif response is None:
                    # Network error - continue with fallback
                    consecutive_errors += 1
                    time_since_last_job = time.time() - self.last_job_time
                    
                    if time_since_last_job > self.fallback_idle_threshold and not self.fallback_active:
                        self.fallback_manager.start_fallback()
                        self.fallback_active = True
                    
                    # Don't spam logs during network issues
                    if consecutive_errors % 10 == 1:
                        logger.info("[MIA] Waiting for backend connection...")
                
                # Sleep before next poll
                time.sleep(self.poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("[MIA] Shutdown requested")
                break
            except Exception as e:
                logger.warning(f"[MIA] Polling error: {type(e).__name__}")
                time.sleep(self.poll_interval)
        
        # Cleanup fallback on exit
        if self.fallback_active:
            self.fallback_manager.stop_fallback()
        
        self.fallback_manager.cleanup()
        logger.info("[MIA] Miner stopped")

def check_gpu():
    """Check if GPU is available (basic check)"""
    try:
        # Try to check NVIDIA GPU
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("[MIA] NVIDIA GPU detected")
            return True
        else:
            logger.info("[MIA] No GPU detected - running in CPU mode")
            return False
    except Exception:
        logger.info("[MIA] Running in CPU mode")
        return False

def main():
    """Main entry point"""
    logger.info("=== MIA GPU Miner Starting ===")
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Verify required environment variables
    if not os.environ.get('MIA_API_URL'):
        logger.info("[MIA] Using default API URL")
    
    # Create and run miner
    miner = MIAMiner()
    
    try:
        miner.poll_for_jobs()
    except Exception as e:
        logger.error(f"[MIA] Fatal error: {type(e).__name__}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("[MIA] Shutdown signal received")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    main()