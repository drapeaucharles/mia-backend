#!/usr/bin/env python3
"""MIA Job Polling Miner - Debug Version with Full Details"""
import os, sys, json, time, logging, requests, socket
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mia-miner-debug")

MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000/v1"

class MIAMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = None
        self.miner_name = f"gpu-miner-{socket.gethostname()}-debug"
        
    def register(self) -> bool:
        """Register with backend and get miner ID"""
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
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Registration error: {e}")
        return False
        
    def get_work(self) -> Optional[Dict]:
        try:
            r = self.session.get(f"{MIA_BACKEND_URL}/get_work", params={"miner_id": self.miner_id}, timeout=30)
            if r.status_code == 200:
                work = r.json()
                if work and work.get("request_id"):
                    logger.info(f"\n{'='*80}")
                    logger.info(f"📥 NEW JOB: {work['request_id']}")
                    logger.info(f"{'='*80}")
                    return work
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            logger.error(f"Error getting work: {e}")
        return None
    
    def process_with_vllm(self, job: Dict) -> Dict:
        try:
            # Show full job details
            logger.info("\n📋 JOB DETAILS:")
            logger.info(f"  Request ID: {job.get('request_id')}")
            logger.info(f"  Max tokens: {job.get('max_tokens', 150)}")
            logger.info(f"  Temperature: {job.get('temperature', 0.7)}")
            
            # Show prompt
            prompt = job.get('prompt', '')
            logger.info(f"\n💬 PROMPT:")
            logger.info(f"  {prompt}")
            
            # Show context if any
            context = job.get("context", {})
            if context:
                logger.info(f"\n🔍 CONTEXT:")
                for key, value in context.items():
                    logger.info(f"  {key}: {value}")
            
            # Show tools if any
            if job.get('tools'):
                logger.info(f"\n🔧 TOOLS AVAILABLE: {len(job['tools'])}")
                for tool in job['tools']:
                    func = tool.get('function', {})
                    logger.info(f"  - {func.get('name')}: {func.get('description', 'No description')}")
            
            # Build messages
            messages = []
            if isinstance(context, dict):
                if context.get("system_prompt"):
                    messages.append({"role": "system", "content": context["system_prompt"]})
                elif context.get("business_name"):
                    messages.append({"role": "system", "content": f"You are a helpful assistant at {context['business_name']}."})
            
            messages.append({"role": "user", "content": prompt})
            
            # Show messages being sent
            logger.info(f"\n📨 MESSAGES TO vLLM:")
            for i, msg in enumerate(messages):
                logger.info(f"  [{msg['role']}]: {msg['content'][:100]}...")
            
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
            content = message.get("content", "")
            tokens = vllm_result.get("usage", {}).get("completion_tokens", 0)
            
            # Show response details
            logger.info(f"\n✅ RESPONSE FROM vLLM:")
            logger.info(f"  Content: {content}")
            logger.info(f"  Tokens generated: {tokens}")
            logger.info(f"  Processing time: {processing_time:.2f}s")
            
            result = {
                "response": content,
                "tokens_generated": tokens,
                "processing_time": processing_time,
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ"
            }
            
            # Check for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                result["tool_call"] = {
                    "name": tool_call["function"]["name"],
                    "parameters": json.loads(tool_call["function"]["arguments"])
                }
                result["requires_tool_execution"] = True
                logger.info(f"\n🔧 TOOL CALL DETECTED:")
                logger.info(f"  Function: {result['tool_call']['name']}")
                logger.info(f"  Parameters: {json.dumps(result['tool_call']['parameters'], indent=2)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error: {str(e)}", "tokens_generated": 0, "processing_time": 0}
    
    def submit_result(self, request_id: str, result: Dict) -> bool:
        try:
            logger.info(f"\n📤 SUBMITTING RESULT:")
            logger.info(f"  Response length: {len(result.get('response', ''))}")
            logger.info(f"  Tokens: {result.get('tokens_generated', 0)}")
            if 'tool_call' in result:
                logger.info(f"  Tool call: {result['tool_call']['name']}")
            
            r = self.session.post(
                f"{MIA_BACKEND_URL}/submit_result",
                json={"miner_id": self.miner_id, "request_id": request_id, "result": result},
                timeout=10
            )
            if r.status_code == 200:
                logger.info(f"  ✓ Successfully submitted!")
                return True
            else:
                logger.error(f"  ✗ Submit failed: {r.status_code}")
        except Exception as e:
            logger.error(f"Submit error: {e}")
        return False
    
    def run(self):
        logger.info(f"MIA Debug Miner starting...")
        logger.info(f"Backend: {MIA_BACKEND_URL}")
        
        # Wait for vLLM to be ready
        logger.info("Waiting for vLLM...")
        for i in range(60):
            try:
                test = self.session.get(f"{VLLM_URL}/models", timeout=5)
                if test.status_code == 200:
                    logger.info("✓ vLLM is ready")
                    break
            except:
                pass
            time.sleep(1)
            sys.stdout.write(".")
            sys.stdout.flush()
        else:
            logger.error("vLLM timeout - make sure it's running")
            sys.exit(1)
        
        # Register with backend
        if not self.register():
            logger.error("Failed to register with backend")
            sys.exit(1)
        
        jobs_completed = 0
        total_tokens = 0
        errors = 0
        
        logger.info(f"\n🚀 Starting job polling loop as miner {self.miner_id}...")
        logger.info("Press Ctrl+C to stop\n")
        
        while True:
            try:
                job = self.get_work()
                if job:
                    request_id = job["request_id"]
                    
                    result = self.process_with_vllm(job)
                    
                    if self.submit_result(request_id, result):
                        jobs_completed += 1
                        total_tokens += result.get("tokens_generated", 0)
                        errors = 0
                        
                        logger.info(f"\n📊 STATS: {jobs_completed} jobs completed, {total_tokens} tokens total")
                        logger.info(f"{'='*80}\n")
                    else:
                        errors += 1
                else:
                    # No work available
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    time.sleep(2)
                
                if errors > 5:
                    logger.warning("Too many errors, re-registering...")
                    self.register()
                    errors = 0
                    
            except KeyboardInterrupt:
                logger.info(f"\n\nShutting down...")
                logger.info(f"Final stats: {jobs_completed} jobs, {total_tokens} tokens")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                errors += 1
                time.sleep(5)

if __name__ == "__main__":
    MIAMiner().run()