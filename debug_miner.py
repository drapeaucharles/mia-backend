#!/usr/bin/env python3
"""Debug version of miner - shows full job details"""
import os, sys, json, time, logging, requests, socket
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mia-debug")

MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000/v1"

class DebugMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = None
        self.miner_name = f"debug-miner-{socket.gethostname()}"
        
    def register(self) -> bool:
        try:
            response = self.session.post(
                f"{MIA_BACKEND_URL}/register_miner",
                json={"name": self.miner_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.miner_id = int(data["miner_id"])
                logger.info(f"✓ Registered as miner ID: {self.miner_id}")
                return True
        except Exception as e:
            logger.error(f"Registration error: {e}")
        return False
        
    def get_work(self) -> Optional[Dict]:
        try:
            r = self.session.get(f"{MIA_BACKEND_URL}/get_work", params={"miner_id": self.miner_id}, timeout=30)
            if r.status_code == 200:
                work = r.json()
                if work and work.get("request_id"):
                    return work
        except Exception as e:
            logger.error(f"Error getting work: {e}")
        return None
    
    def process_with_vllm(self, job: Dict) -> Dict:
        try:
            # Show job details
            logger.info("📥 JOB DETAILS:")
            logger.info(f"  Prompt: {job.get('prompt', '')[:100]}...")
            logger.info(f"  Max tokens: {job.get('max_tokens', 150)}")
            logger.info(f"  Temperature: {job.get('temperature', 0.7)}")
            
            if job.get('context'):
                logger.info(f"  Context: {json.dumps(job['context'], indent=2)}")
            
            if job.get('tools'):
                logger.info(f"  Tools available: {len(job['tools'])}")
                for tool in job['tools']:
                    logger.info(f"    - {tool.get('function', {}).get('name', 'unknown')}")
            
            messages = []
            context = job.get("context", {})
            if isinstance(context, dict):
                if context.get("system_prompt"):
                    messages.append({"role": "system", "content": context["system_prompt"]})
                elif context.get("business_name"):
                    messages.append({"role": "system", "content": f"You are a helpful assistant at {context['business_name']}."})
            
            messages.append({"role": "user", "content": job.get("prompt", "")})
            
            vllm_request = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": messages,
                "max_tokens": job.get("max_tokens", 150),
                "temperature": job.get("temperature", 0.7)
            }
            
            if job.get("tools"):
                vllm_request["tools"] = job["tools"]
                vllm_request["tool_choice"] = job.get("tool_choice", "auto")
            
            start_time = time.time()
            r = self.session.post(f"{VLLM_URL}/chat/completions", json=vllm_request, timeout=60)
            processing_time = time.time() - start_time
            
            if r.status_code != 200:
                logger.error(f"vLLM error: {r.status_code}")
                return {"response": f"Error: vLLM returned {r.status_code}", "tokens_generated": 0, "processing_time": processing_time}
            
            vllm_result = r.json()
            message = vllm_result["choices"][0]["message"]
            
            # Show response
            logger.info("📤 RESPONSE:")
            logger.info(f"  Content: {message.get('content', '')[:200]}...")
            logger.info(f"  Tokens: {vllm_result.get('usage', {}).get('completion_tokens', 0)}")
            
            result = {
                "response": message.get("content", ""),
                "tokens_generated": vllm_result.get("usage", {}).get("completion_tokens", 0),
                "processing_time": processing_time,
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ"
            }
            
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                result["tool_call"] = {
                    "name": tool_call["function"]["name"],
                    "parameters": json.loads(tool_call["function"]["arguments"])
                }
                result["requires_tool_execution"] = True
                logger.info(f"🔧 TOOL CALL: {result['tool_call']['name']}")
                logger.info(f"   Parameters: {json.dumps(result['tool_call']['parameters'], indent=2)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"response": f"Error: {str(e)}", "tokens_generated": 0, "processing_time": 0}
    
    def submit_result(self, request_id: str, result: Dict) -> bool:
        try:
            r = self.session.post(
                f"{MIA_BACKEND_URL}/submit_result",
                json={"miner_id": self.miner_id, "request_id": request_id, "result": result},
                timeout=10
            )
            if r.status_code == 200:
                logger.info(f"✓ Submitted {request_id}")
                return True
            else:
                logger.error(f"Submit failed: {r.status_code}")
        except Exception as e:
            logger.error(f"Submit error: {e}")
        return False
    
    def run_debug(self):
        logger.info("Debug Miner - Shows full job/response details")
        
        # Wait for vLLM
        logger.info("Checking vLLM...")
        for i in range(30):
            try:
                test = self.session.get(f"{VLLM_URL}/models", timeout=5)
                if test.status_code == 200:
                    logger.info("✓ vLLM is ready")
                    break
            except:
                pass
            time.sleep(1)
        
        # Register
        if not self.register():
            logger.error("Failed to register")
            sys.exit(1)
        
        logger.info("\nProcessing 5 jobs to show what's happening...\n")
        
        for i in range(5):
            logger.info(f"\n{'='*60}")
            logger.info(f"JOB #{i+1}")
            logger.info('='*60)
            
            job = self.get_work()
            if job:
                request_id = job["request_id"]
                result = self.process_with_vllm(job)
                self.submit_result(request_id, result)
            else:
                logger.info("No job available")
                time.sleep(2)
            
            time.sleep(1)
        
        logger.info("\n✅ Debug complete - check the output above to see what jobs are being processed")

if __name__ == "__main__":
    DebugMiner().run_debug()