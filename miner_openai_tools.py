#!/usr/bin/env python3
"""
MIA GPU Miner with OpenAI Tools Support
Handles both traditional prompts and OpenAI-style tool calling
"""
import json
import logging
import os
import re
import socket
import time
from typing import Dict, List, Optional, Any
import requests
from vllm import LLM, SamplingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Initialize model
logger.info(f"Loading model: {MODEL_NAME}")
model = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    max_model_len=12288,  # 12k context
    quantization="awq",
    dtype="float16",
    enforce_eager=True
)

def format_messages_for_qwen(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Format messages in Qwen chat format with optional tools"""
    prompt = ""
    
    # Extract system message
    system_msg = "You are a helpful assistant."
    if messages and messages[0].get("role") == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    
    # Add tools to system message if provided
    if tools:
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            desc = f"- {func.get('name')}: {func.get('description')}"
            tool_descriptions.append(desc)
        
        system_msg += "\n\nYou have access to the following tools:\n"
        system_msg += "\n".join(tool_descriptions)
        system_msg += "\n\nWhen you need to use a tool, respond with a JSON object in this format:"
        system_msg += '\n{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool_name", "arguments": "{\\"param\\": \\"value\\"}"}}]}'
    
    # Start with system message
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    
    # Add conversation messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role == "tool":
            # Tool results are provided as user messages in Qwen
            prompt += f"<|im_start|>user\nTool result: {content}<|im_end|>\n"
    
    prompt += "<|im_start|>assistant\n"
    return prompt

def extract_tool_calls(response: str) -> Optional[List[Dict]]:
    """Extract tool calls from model response"""
    # Try to find JSON tool calls
    try:
        # Look for JSON object with tool_calls
        json_match = re.search(r'\{.*"tool_calls".*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "tool_calls" in data:
                return data["tool_calls"]
    except:
        pass
    
    # Try to find individual tool call patterns
    tool_calls = []
    
    # Pattern 1: function_name(arguments)
    func_pattern = r'(\w+)\s*\((.*?)\)'
    matches = re.findall(func_pattern, response)
    for name, args in matches:
        if name in ['search_menu_items', 'get_dish_details', 'filter_by_dietary']:
            try:
                # Parse arguments
                args_dict = {}
                if 'search_term' in args:
                    term_match = re.search(r'["\']([^"\']+)["\']', args)
                    if term_match:
                        args_dict['search_term'] = term_match.group(1)
                
                tool_calls.append({
                    "id": f"call_{len(tool_calls)+1}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args_dict)
                    }
                })
            except:
                pass
    
    return tool_calls if tool_calls else None

def generate_with_tools(messages: List[Dict], tools: Optional[List] = None, 
                       temperature: float = 0.7, max_tokens: int = 300) -> Dict:
    """Generate response with optional tool support"""
    
    # Format prompt
    prompt = format_messages_for_qwen(messages, tools)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        stop=["<|im_end|>", "<|endoftext|>"]
    )
    
    # Generate
    start_time = time.time()
    outputs = model.generate([prompt], sampling_params)
    generation_time = time.time() - start_time
    
    # Extract response
    response_text = outputs[0].outputs[0].text.strip()
    token_count = len(outputs[0].outputs[0].token_ids)
    
    logger.info(f"Generated {token_count} tokens in {generation_time:.2f}s")
    
    # Build OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": token_count,
            "total_tokens": len(prompt.split()) + token_count
        }
    }
    
    # Check for tool calls if tools were provided
    if tools:
        tool_calls = extract_tool_calls(response_text)
        if tool_calls:
            # Update response with tool calls
            response["choices"][0]["message"]["tool_calls"] = tool_calls
            # Remove tool call JSON from content
            clean_content = re.sub(r'\{.*"tool_calls".*\}', '', response_text, flags=re.DOTALL).strip()
            response["choices"][0]["message"]["content"] = clean_content
            logger.info(f"Detected {len(tool_calls)} tool calls")
    
    return response

# MIA Backend Integration
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')
miner_name = f"qwen-openai-{socket.gethostname()}"
miner_id = None

def register_miner():
    """Register with MIA backend"""
    global miner_id
    try:
        response = requests.post(
            f"{backend_url}/miners/register",
            json={"name": miner_name},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            miner_id = data.get('id')
            logger.info(f"✅ Registered as miner {miner_id}")
            return True
    except Exception as e:
        logger.error(f"❌ Registration failed: {e}")
    return False

def get_next_job():
    """Get next job from backend"""
    if not miner_id:
        return None
    
    try:
        response = requests.get(
            f"{backend_url}/miners/{miner_id}/job",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def submit_result(job_id: str, result: Dict):
    """Submit job result"""
    try:
        # Handle both traditional and OpenAI format
        if "choices" in result:
            # OpenAI format - extract content
            message = result["choices"][0]["message"]
            response_text = message.get("content", "")
            
            # Include tool calls if present
            tool_calls = message.get("tool_calls")
            if tool_calls:
                result_data = {
                    "response": response_text,
                    "tool_calls": tool_calls,
                    "format": "openai"
                }
            else:
                result_data = {"response": response_text}
        else:
            # Traditional format
            result_data = result
        
        response = requests.post(
            f"{backend_url}/miners/{miner_id}/job/{job_id}/complete",
            json=result_data,
            timeout=30
        )
        if response.status_code == 200:
            logger.info(f"✅ Submitted job {job_id}")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to submit job {job_id}: {e}")
    return False

def process_job(job: Dict):
    """Process a single job"""
    job_id = job.get('id')
    job_data = job.get('data', {})
    
    logger.info(f"Processing job {job_id}")
    
    try:
        # Check if this is an OpenAI-style request
        if "messages" in job_data:
            # OpenAI format
            messages = job_data.get("messages", [])
            tools = job_data.get("tools")
            temperature = job_data.get("temperature", 0.7)
            max_tokens = job_data.get("max_tokens", 300)
            
            logger.info(f"OpenAI format: {len(messages)} messages, {len(tools) if tools else 0} tools")
            
            result = generate_with_tools(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Traditional format
            prompt = job_data.get('message', '')
            params = {
                'temperature': job_data.get('temperature', 0.7),
                'max_tokens': job_data.get('max_tokens', 512)
            }
            
            # Convert to messages format
            messages = [{"role": "user", "content": prompt}]
            result = generate_with_tools(messages, None, **params)
            
            # Convert back to traditional format
            result = {
                'response': result["choices"][0]["message"]["content"],
                'tokens': result["usage"]["completion_tokens"]
            }
        
        # Submit result
        submit_result(job_id, result)
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        submit_result(job_id, {"error": str(e)})

def main():
    """Main miner loop"""
    logger.info(f"🚀 Starting OpenAI-compatible miner: {miner_name}")
    
    # Register with backend
    retry_count = 0
    while retry_count < 5:
        if register_miner():
            break
        retry_count += 1
        logger.info(f"Retrying registration in 5s... ({retry_count}/5)")
        time.sleep(5)
    
    if not miner_id:
        logger.error("Failed to register with backend")
        return
    
    # Main processing loop
    logger.info("✅ Ready to process jobs with OpenAI tool support")
    
    while True:
        try:
            job = get_next_job()
            if job:
                process_job(job)
            else:
                time.sleep(0.5)  # Short sleep when no jobs
        
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()