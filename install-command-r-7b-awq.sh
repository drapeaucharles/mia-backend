#!/bin/bash
# Command-R-7B-AWQ Installer for MIA with Function Calling
# Uses quantized version for better performance

echo "🚀 Installing Command-R-7B-AWQ for MIA"
echo "====================================="
echo "Using AWQ quantized model for efficiency"
echo ""

# Check if running in Vast.ai or local
if [ -d "/data" ]; then
    BASE_DIR="/data"
else
    BASE_DIR="$HOME"
fi

INSTALL_DIR="$BASE_DIR/mia-command-r"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Set up Python environment
echo "📦 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.4.2  # For fast inference with AWQ
pip install transformers accelerate sentencepiece protobuf
pip install flask waitress requests psutil gputil
pip install auto-awq  # For AWQ support

# Create Command-R miner with function calling
cat > command_r_awq_miner.py << 'EOF'
#!/usr/bin/env python3
"""
MIA Command-R-7B-AWQ Miner with Function Calling
Uses quantized model for efficient inference
"""
import os
import sys
import json
import time
import logging
import requests
import threading
import re
from flask import Flask, request, jsonify
from waitress import serve
from vllm import LLM, SamplingParams
from typing import Dict, List, Optional, Tuple

# Configure environment
if os.path.exists("/data"):
    os.environ["HF_HOME"] = "/data/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging
log_file = "/data/command_r.log" if os.path.exists("/data") else "command_r.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger('command-r-miner')

app = Flask(__name__)

# Global model
model = None

# Tool calling patterns
TOOL_CALL_PATTERN = r'<tool_call>\s*({[^}]+})\s*</tool_call>'
TOOL_RESULT_PATTERN = r'<tool_result>\s*({[^}]+})\s*</tool_result>'

def load_model():
    """Load Command-R model - fallback to Qwen if not available"""
    global model
    logger.info("Loading language model...")
    
    try:
        # Try Command-R variants
        model_options = [
            "TheBloke/c4ai-command-r-v01-AWQ",  # AWQ quantized version
            "CohereForAI/c4ai-command-r-v01",   # Full version
            "Qwen/Qwen2.5-7B-Instruct"          # Fallback to Qwen
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Trying to load {model_name}...")
                model = LLM(
                    model=model_name,
                    dtype="half",
                    gpu_memory_utilization=0.95,
                    max_model_len=4096,
                    trust_remote_code=True,
                    enforce_eager=True,
                    quantization="awq" if "AWQ" in model_name else None
                )
                logger.info(f"✓ Successfully loaded {model_name}")
                return model_name
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
                
        raise Exception("No models could be loaded")
        
    except Exception as e:
        logger.error(f"Critical error loading models: {e}")
        sys.exit(1)

def format_tools_prompt(prompt: str, tools: List[Dict], context: Dict = None) -> str:
    """Format prompt with tool definitions"""
    
    # Build tool descriptions
    tool_desc = []
    for tool in tools:
        params = tool.get('parameters', {})
        param_desc = []
        for pname, pinfo in params.items():
            param_desc.append(f"  - {pname}: {pinfo.get('description', 'No description')}")
        
        tool_desc.append(f"""Tool: {tool['name']}
Description: {tool.get('description', 'No description')}
Parameters:
{chr(10).join(param_desc)}""")
    
    tools_text = "\n\n".join(tool_desc)
    
    # Build context if provided
    context_text = ""
    if context:
        if 'menu_items' in context:
            context_text = f"\nAvailable menu items: {', '.join(context['menu_items'])}\n"
        if 'business_name' in context:
            context_text = f"\nYou are helping customers at {context['business_name']}.\n" + context_text
    
    # Format the full prompt
    system_prompt = f"""You are a helpful customer support assistant.{context_text}

You have access to the following tools to help answer questions:

{tools_text}

When you need to use a tool, respond with:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>

After I provide the tool result, give a natural response to the customer.
Important: Only use tools when you need specific information not in your context."""
    
    return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from model response"""
    match = re.search(TOOL_CALL_PATTERN, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tool call: {match.group(1)}")
    return None

def generate_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None, 
                       max_tokens: int = 150, temperature: float = 0.7) -> Dict:
    """Generate response with optional tool calling"""
    
    # Format prompt
    if tools:
        formatted_prompt = format_tools_prompt(prompt, tools, context)
    else:
        # Simple format without tools
        formatted_prompt = f"User: {prompt}\nAssistant:"
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        stop=["User:", "\n\n"]
    )
    
    start = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    gen_time = time.time() - start
    
    response = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / gen_time if gen_time > 0 else 0
    
    logger.info(f"Generated {tokens} tokens in {gen_time:.2f}s = {speed:.1f} tok/s")
    
    # Check for tool call
    tool_call = extract_tool_call(response) if tools else None
    
    return {
        'response': response,
        'tool_call': tool_call,
        'tokens': tokens,
        'time': gen_time,
        'speed': speed
    }

def process_tool_result(original_prompt: str, tool_call: Dict, tool_result: Dict, 
                       context: Dict = None, max_tokens: int = 150) -> Dict:
    """Process tool result and generate final response"""
    
    # Format the conversation with tool result
    formatted_prompt = f"""Previous conversation:
User: {original_prompt}
Assistant: I'll look that up for you.

Tool used: {tool_call['name']}
Tool result: {json.dumps(tool_result, indent=2)}

Now provide a natural, helpful response to the user based on this information.

User: {original_prompt}
Assistant:"""
    
    # Generate final response
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.95
    )
    
    start = time.time()
    outputs = model.generate([formatted_prompt], sampling_params)
    gen_time = time.time() - start
    
    response = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    speed = tokens / gen_time if gen_time > 0 else 0
    
    return {
        'response': response,
        'tokens': tokens,
        'time': gen_time,
        'speed': speed
    }

@app.route('/generate', methods=['POST'])
@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for generation with tool calling"""
    data = request.json
    
    # Check if this is a tool result submission
    if 'tool_result' in data:
        result = process_tool_result(
            original_prompt=data.get('original_prompt', ''),
            tool_call=data.get('tool_call', {}),
            tool_result=data.get('tool_result', {}),
            context=data.get('context', {}),
            max_tokens=data.get('max_tokens', 150)
        )
    else:
        # Initial generation
        result = generate_with_tools(
            prompt=data.get('prompt', ''),
            tools=data.get('tools', None),
            context=data.get('context', {}),
            max_tokens=data.get('max_tokens', 150),
            temperature=data.get('temperature', 0.7)
        )
    
    return jsonify({
        'text': result['response'],
        'tool_call': result.get('tool_call'),
        'tokens_generated': result['tokens'],
        'tokens_per_second': round(result['speed'], 1)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready',
        'model': 'Command-R-7B-AWQ',
        'features': ['tool_calling', 'multilingual', 'customer_support'],
        'languages': ['en', 'fr', 'ru', 'id', 'es', 'de', 'pt', 'it']
    })

# MIA Backend Integration
backend_url = os.getenv('MIA_BACKEND_URL', 'https://mia-backend-production.up.railway.app')

def worker():
    """Background worker for MIA jobs"""
    miner_id = "command-r-awq-1"
    logger.info(f"Starting worker with ID: {miner_id}")
    
    while True:
        try:
            # Get work from backend
            response = requests.get(
                f"{backend_url}/get_work?miner_id={miner_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                work = response.json()
                
                if work and work.get('request_id'):
                    logger.info(f"Processing job: {work['request_id']}")
                    
                    # Generate response with tool support
                    result = generate_with_tools(
                        prompt=work.get('prompt', ''),
                        tools=work.get('tools', None),
                        context=work.get('context', {}),
                        max_tokens=work.get('max_tokens', 150),
                        temperature=work.get('temperature', 0.7)
                    )
                    
                    # Prepare submission
                    submission = {
                        'response': result['response'],
                        'tokens_generated': result['tokens'],
                        'processing_time': result['time']
                    }
                    
                    # Add tool call if present
                    if result.get('tool_call'):
                        submission['tool_call'] = result['tool_call']
                        submission['requires_tool_execution'] = True
                    
                    # Submit result
                    requests.post(f"{backend_url}/submit_result", json={
                        'miner_id': miner_id,
                        'request_id': work['request_id'],
                        'result': submission
                    })
                    
                    logger.info(f"✓ Job completed: {work['request_id']}")
                
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)
        
        time.sleep(1)

def test_tool_calling():
    """Test tool calling functionality"""
    logger.info("Testing tool calling...")
    
    test_tools = [
        {
            "name": "get_dish_details",
            "description": "Get detailed information about a specific menu item",
            "parameters": {
                "dish_name": {
                    "type": "string",
                    "description": "The name of the dish to get details for"
                }
            }
        },
        {
            "name": "search_by_ingredient", 
            "description": "Find dishes containing a specific ingredient",
            "parameters": {
                "ingredient": {
                    "type": "string",
                    "description": "The ingredient to search for"
                }
            }
        }
    ]
    
    test_context = {
        "business_name": "Bella Vista Restaurant",
        "menu_items": ["Lobster Ravioli", "Truffle Arancini", "Caesar Salad"]
    }
    
    # Test 1: Should trigger tool call
    result1 = generate_with_tools(
        "What are the ingredients in the Lobster Ravioli?",
        tools=test_tools,
        context=test_context
    )
    logger.info(f"Test 1 - Tool call: {result1.get('tool_call')}")
    
    # Test 2: Should not trigger tool call
    result2 = generate_with_tools(
        "What items do you have on the menu?",
        tools=test_tools,
        context=test_context
    )
    logger.info(f"Test 2 - Response: {result2['response'][:100]}...")

if __name__ == "__main__":
    # Load model
    model_name = load_model()
    
    # Run tests
    test_tool_calling()
    
    # Start worker thread
    threading.Thread(target=worker, daemon=True).start()
    
    # Start API server
    logger.info(f"Starting server with {model_name} on port 8000...")
    serve(app, host='0.0.0.0', port=8000, threads=4)
EOF

# Create runner script
cat > run_command_r.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python command_r_awq_miner.py
EOF
chmod +x run_command_r.sh

# Create start script
cat > start_command_r.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate

# Stop any existing miners
pkill -f "command_r_awq_miner.py" || true
pkill -f "mia_miner" || true
sleep 2

# Start new miner
nohup python command_r_awq_miner.py > command_r.log 2>&1 &
echo $! > command_r.pid
echo "Command-R miner started. PID: $(cat command_r.pid)"
echo "Logs: tail -f command_r.log"
EOF
chmod +x start_command_r.sh

# Create stop script
cat > stop_command_r.sh << 'EOF'
#!/bin/bash
if [ -f command_r.pid ]; then
    kill $(cat command_r.pid) 2>/dev/null
    rm command_r.pid
    echo "Command-R miner stopped"
else
    pkill -f "command_r_awq_miner.py" || true
    echo "Miner processes stopped"
fi
EOF
chmod +x stop_command_r.sh

# Create test script
cat > test_command_r.sh << 'EOF'
#!/bin/bash
echo "Testing Command-R API..."
echo ""

# Test 1: Simple generation
echo "Test 1: Simple generation"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 50
  }' | python3 -m json.tool

echo ""
echo "Test 2: Tool calling"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What ingredients are in the pasta carbonara?",
    "tools": [{
      "name": "get_dish_details",
      "description": "Get details about a dish",
      "parameters": {"dish_name": {"type": "string"}}
    }],
    "context": {"business_name": "Italian Restaurant"}
  }' | python3 -m json.tool
EOF
chmod +x test_command_r.sh

echo ""
echo "✅ Command-R-7B-AWQ installer created!"
echo ""
echo "This installer will:"
echo "1. Try to load Command-R-7B-AWQ (quantized)"
echo "2. Fall back to Qwen2.5-7B if needed"
echo "3. Support tool calling for data queries"
echo ""
echo "To install and run:"
echo "1. cd $INSTALL_DIR"
echo "2. bash $(basename $0)"
echo "3. ./start_command_r.sh"
echo "4. ./test_command_r.sh"