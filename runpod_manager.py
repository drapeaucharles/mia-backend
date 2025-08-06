import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class RunPodManager:
    """Manages RunPod serverless Mixtral jobs for idle GPU monetization"""
    
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.base_url = "https://api.runpod.ai/v2"
        
        # Pricing estimates (adjust based on RunPod pricing)
        self.cost_per_1k_tokens = 0.0002  # $0.0002 per 1K tokens
        self.revenue_per_1k_tokens = 0.001  # $0.001 per 1K tokens (5x markup)
        
        if not self.api_key or not self.endpoint_id:
            logger.warning("RunPod credentials not configured")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def submit_job(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Submit a serverless job to RunPod"""
        if not self.api_key or not self.endpoint_id:
            logger.error("RunPod not configured")
            return None
        
        url = f"{self.base_url}/{self.endpoint_id}/run"
        
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        job_id = data.get("id")
                        logger.info(f"RunPod job submitted: {job_id}")
                        return job_id
                    else:
                        error = await response.text()
                        logger.error(f"RunPod submission failed: {error}")
                        return None
                        
        except Exception as e:
            logger.error(f"RunPod API error: {e}")
            return None
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of a RunPod job"""
        if not self.api_key or not self.endpoint_id:
            return None
            
        url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"RunPod status check error: {e}")
            return None
    
    async def wait_for_result(
        self, 
        job_id: str, 
        timeout: int = 300,
        poll_interval: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Poll for job completion and return result"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            status = await self.get_job_status(job_id)
            
            if not status:
                await asyncio.sleep(poll_interval)
                continue
            
            job_status = status.get("status")
            
            if job_status == "COMPLETED":
                output = status.get("output", {})
                return {
                    "status": "completed",
                    "output": output.get("text", ""),
                    "tokens_generated": output.get("tokens_generated", 0),
                    "job_id": job_id
                }
            elif job_status in ["FAILED", "CANCELLED"]:
                return {
                    "status": "failed",
                    "error": status.get("error", "Job failed"),
                    "job_id": job_id
                }
            
            await asyncio.sleep(poll_interval)
        
        return {
            "status": "timeout",
            "error": "Job timed out",
            "job_id": job_id
        }
    
    def calculate_revenue(self, output_tokens: int) -> Dict[str, float]:
        """Calculate revenue and profit from token generation"""
        tokens_in_thousands = output_tokens / 1000
        
        cost = tokens_in_thousands * self.cost_per_1k_tokens
        revenue = tokens_in_thousands * self.revenue_per_1k_tokens
        profit = revenue - cost
        
        return {
            "cost_usd": round(cost, 6),
            "revenue_usd": round(revenue, 6),
            "profit_usd": round(profit, 6),
            "output_tokens": output_tokens
        }
    
    async def process_idle_job(self, prompt: str) -> Dict[str, Any]:
        """Process a complete idle job through RunPod"""
        # Submit job
        job_id = await self.submit_job(prompt)
        if not job_id:
            return {
                "success": False,
                "error": "Failed to submit job to RunPod"
            }
        
        # Wait for result
        result = await self.wait_for_result(job_id)
        
        if result["status"] == "completed":
            # Calculate revenue
            tokens = result.get("tokens_generated", 0)
            revenue_info = self.calculate_revenue(tokens)
            
            return {
                "success": True,
                "job_id": job_id,
                "output": result["output"],
                "output_tokens": tokens,
                "revenue_usd": revenue_info["revenue_usd"],
                "profit_usd": revenue_info["profit_usd"],
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "job_id": job_id,
                "error": result.get("error", "Job processing failed"),
                "status": result["status"]
            }
    
    def estimate_job_revenue(self, estimated_tokens: int = 500) -> Dict[str, float]:
        """Estimate revenue for a job before processing"""
        return self.calculate_revenue(estimated_tokens)
    
    async def health_check(self) -> bool:
        """Check if RunPod API is accessible"""
        if not self.api_key:
            return False
            
        try:
            url = f"{self.base_url}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False