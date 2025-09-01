#!/bin/bash
# Fixed vLLM installer with proper tool call handling

echo "🔧 Installing vLLM with Fixed Tool Call Support"
echo "============================================="

# Check if already installed
if [ -d "/data/qwen-awq-miner" ]; then
    echo "📁 Found existing installation at /data/qwen-awq-miner"
    cd /data/qwen-awq-miner
    
    # Stop existing miner
    if [ -f "miner.pid" ]; then
        echo "🛑 Stopping existing miner..."
        kill $(cat miner.pid) 2>/dev/null || true
        rm -f miner.pid
    fi
    
    # Backup existing miner
    if [ -f "mia_miner.py" ]; then
        cp mia_miner.py mia_miner.py.backup
        echo "✅ Backed up existing miner"
    fi
else
    echo "❌ No installation found at /data/qwen-awq-miner"
    echo "Run the universal installer first"
    exit 1
fi

# Create fixed miner with proper tool handling
cat > mia_miner.py << 'EOF'
#!/usr/bin/env python3
"""MIA Job Polling Miner with Fixed Tool Call Handling"""
import requests
import time
import logging
import sys
import json
import os
import socket
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('miner.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MIA_BACKEND_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend-production.up.railway.app")
VLLM_URL = "http://localhost:8000/v1/chat/completions"

class MIAMiner:
    def __init__(self):
        self.session = requests.Session()
        self.miner_id = None
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        
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
                self.miner_id = data.get("miner_id")
                logger.info(f"✅ Registered as miner {self.miner_id} ({self.miner_name})")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def get_work(self) -> dict:
        try:
            r = self.session.get(
                f"{MIA_BACKEND_URL}/get_work?miner_id={self.miner_id}",
                timeout=10
            )
            if r.status_code == 200:
                return r.json()
            return None
        except:
            return None
    
    def process_with_vllm(self, job: dict) -> dict:
        try:
            start_time = time.time()
            
            # Extract job details
            prompt = job.get("prompt", "")
            max_tokens = job.get("max_tokens", 500)
            temperature = job.get("temperature", 0.7)
            tools = job.get("tools", [])
            tool_choice = job.get("tool_choice", "auto")
            context = job.get("context", {})
            
            # Log the request
            logger.info("")
            logger.info(f"💬 USER MESSAGE:")
            logger.info(f"  {prompt}")
            
            if context:
                logger.info("")
                logger.info("🔍 CONTEXT:")
                for key, value in context.items():
                    value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    logger.info(f"  {key}: {value_str}")
            
            if tools:
                logger.info("")
                logger.info(f"🔧 TOOLS AVAILABLE: {len(tools)}")
                for tool in tools:
                    if tool.get("type") == "function":
                        func = tool.get("function", {})
                        logger.info(f"  - {func.get('name')}: {func.get('description', '')}")
            
            # Build messages
            messages = []
            
            # Add system message from context
            if context.get("system_prompt"):
                messages.append({
                    "role": "system",
                    "content": context["system_prompt"]
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            logger.info("")
            logger.info("📨 MESSAGES TO vLLM:")
            for msg in messages:
                content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                logger.info(f"  [{msg['role']}]: {content_preview}")
            
            # Prepare vLLM request
            vllm_request = {
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add tools if available
            if tools:
                vllm_request["tools"] = tools
                vllm_request["tool_choice"] = tool_choice
            
            # Call vLLM
            response = self.session.post(
                VLLM_URL,
                json=vllm_request,
                timeout=60
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code != 200:
                logger.error(f"vLLM error: {response.status_code}")
                return {
                    "response": f"Error: vLLM returned {response.status_code}",
                    "tokens_generated": 0,
                    "processing_time": elapsed_time
                }
            
            result = response.json()
            
            # Extract response
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content", "")
            usage = result.get("usage", {})
            
            logger.info("")
            logger.info("✅ RESPONSE FROM vLLM:")
            logger.info(f"  Content: {content[:100] + '...' if content and len(content) > 100 else content}")
            logger.info(f"  Tokens generated: {usage.get('completion_tokens', 0)}")
            logger.info(f"  Processing time: {elapsed_time:.2f}s")
            
            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.info("")
                logger.info("🔧 TOOL CALL DETECTED:")
                tool_call = tool_calls[0]
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                logger.info(f"  Function: {function_name}")
                logger.info(f"  Parameters: {json.dumps(function_args, indent=2)}")
                
                # For tool calls, response might be None but that's OK
                return {
                    "response": content if content is not None else "",
                    "tool_call": {
                        "name": function_name,
                        "parameters": function_args
                    },
                    "tokens_generated": usage.get("completion_tokens", 0),
                    "processing_time": elapsed_time
                }
            
            # Regular response
            return {
                "response": content if content is not None else "",
                "tokens_generated": usage.get("completion_tokens", 0),
                "processing_time": elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"response": f"Error: {str(e)}", "tokens_generated": 0, "processing_time": 0}
    
    def submit_result(self, request_id: str, result: dict) -> bool:
        try:
            logger.info("")
            logger.info("📤 SUBMITTING RESULT:")
            
            # Ensure all required fields exist
            if result.get("response") is None:
                result["response"] = ""
            
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
    
    def run(self):
        logger.info(f"🚀 Starting MIA Miner - {self.miner_name}")
        logger.info(f"Backend: {MIA_BACKEND_URL}")
        
        # Register with backend
        if not self.register():
            logger.error("Failed to register with backend")
            return
        
        # Write PID
        with open("miner.pid", "w") as f:
            f.write(str(os.getpid()))
        
        jobs_completed = 0
        total_tokens = 0
        errors = 0
        
        logger.info("⏳ Polling for jobs...")
        
        while True:
            try:
                job = self.get_work()
                if job and job.get("request_id"):
                    request_id = job["request_id"]
                    logger.info(f"📋 Got job: {request_id}")
                    
                    result = self.process_with_vllm(job)
                    
                    if self.submit_result(request_id, result):
                        jobs_completed += 1
                        total_tokens += result.get("tokens_generated", 0)
                        errors = 0
                        logger.info(f"📊 Stats: {jobs_completed} jobs, {total_tokens} tokens")
                    else:
                        errors += 1
                else:
                    time.sleep(2)
                
                if errors > 5:
                    logger.error("Too many errors, reconnecting...")
                    time.sleep(10)
                    if not self.register():
                        logger.error("Failed to re-register")
                        break
                    errors = 0
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(5)

if __name__ == "__main__":
    # Wait for vLLM to be ready
    logger.info("⏳ Waiting for vLLM to start...")
    for i in range(30):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                logger.info("✅ vLLM is ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        logger.error("❌ vLLM failed to start after 60 seconds")
        sys.exit(1)
    
    # Start miner
    miner = MIAMiner()
    miner.run()
EOF

# Make executable
chmod +x mia_miner.py

# Create restart script (keeping debug mode)
cat > restart_miner.sh << 'EOF'
#!/bin/bash
cd /data/qwen-awq-miner

# Stop existing miner
if [ -f miner.pid ]; then
    echo "Stopping miner..."
    kill $(cat miner.pid) 2>/dev/null || true
    rm -f miner.pid
fi

# Start new miner with debug logging to console
echo "Starting fixed miner with debug logging..."
source .venv/bin/activate

# Run in foreground with full logging
echo "📋 Starting miner with live debug output..."
echo "Press Ctrl+C to stop"
echo ""
python mia_miner.py
EOF

chmod +x restart_miner.sh

echo ""
echo "✅ Tool call fix installed!"
echo ""
echo "To apply the fix:"
echo "  cd /data/qwen-awq-miner"
echo "  ./restart_miner.sh"
echo ""
echo "To check logs:"
echo "  tail -f logs/miner_direct.log"