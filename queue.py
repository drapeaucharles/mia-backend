import redis
import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RedisQueue:
    def __init__(self):
        """Initialize Redis connection"""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.job_queue_key = "mia:job_queue"
        self.results_prefix = "mia:results:"
        
    def push_job(self, job: Dict[str, Any]) -> bool:
        """Push a job to the queue"""
        try:
            job_json = json.dumps(job)
            self.redis_client.rpush(self.job_queue_key, job_json)
            return True
        except Exception as e:
            print(f"Error pushing job to queue: {e}")
            return False
    
    def pop_job(self) -> Optional[Dict[str, Any]]:
        """Pop the next job from the queue"""
        try:
            job_json = self.redis_client.lpop(self.job_queue_key)
            if job_json:
                return json.loads(job_json)
            return None
        except Exception as e:
            print(f"Error popping job from queue: {e}")
            return None
    
    def store_result(self, job_id: str, result: Dict[str, Any], ttl: int = 3600) -> bool:
        """Store job result with TTL (default 1 hour)"""
        try:
            result_key = f"{self.results_prefix}{job_id}"
            result_json = json.dumps(result)
            self.redis_client.setex(result_key, ttl, result_json)
            return True
        except Exception as e:
            print(f"Error storing result: {e}")
            return False
    
    def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job result by job_id"""
        try:
            result_key = f"{self.results_prefix}{job_id}"
            result_json = self.redis_client.get(result_key)
            if result_json:
                return json.loads(result_json)
            return None
        except Exception as e:
            print(f"Error retrieving result: {e}")
            return None
    
    def get_queue_length(self) -> int:
        """Get the number of jobs in queue"""
        try:
            return self.redis_client.llen(self.job_queue_key)
        except Exception as e:
            print(f"Error getting queue length: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def clear_queue(self) -> bool:
        """Clear all jobs from the queue (use with caution)"""
        try:
            self.redis_client.delete(self.job_queue_key)
            return True
        except Exception as e:
            print(f"Error clearing queue: {e}")
            return False