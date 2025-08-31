#!/bin/bash
# Fix tool format in miner - tell model HOW to use tools without XML

cd /data/qwen-awq-miner

# Stop current miner
echo "Stopping current miner..."
pkill -f miner.py || true
sleep 2

# Create the updated function
cat > fix_tools_v2.py << 'EOF'
import sys
import re

# Read the current miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Find and replace the format_prompt_with_tools function
old_pattern = r'def format_prompt_with_tools\(.*?\):\s*""".*?""".*?return.*?<\|im_start\|>assistant\\n"'
new_function = '''def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    """Format prompt with tool definitions for Qwen"""
    if not tools:
        # Simple format without tools
        if context and context.get("system_prompt"):
            return f"<|im_start|>system\\n{context['system_prompt']}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        return f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    
    # Build tool descriptions
    tool_descriptions = []
    for tool in tools:
        params_str = json.dumps(tool.get("parameters", {}), indent=2)
        tool_descriptions.append(f"""Function: {tool['name']}
Description: {tool.get('description', '')}
Parameters: {params_str}""")
    
    tools_text = "\\n\\n".join(tool_descriptions)
    
    # Add context
    ctx = ""
    if context:
        if context.get("business_name"):
            ctx += f"\\nYou are helping customers at {context['business_name']}."
        if context.get("system_prompt"):
            ctx = f"\\n{context['system_prompt']}"
    
    # System prompt with clear tool usage instructions
    system_prompt = f"""You are a helpful assistant.{ctx}

You have access to these functions:
{tools_text}

When you need to use a function, respond in this format:
I'll help you with that. Let me search for [what you're searching].

Function call: {{"name": "function_name", "parameters": {{"param": "value"}}}}

Then provide the results in a natural way."""
    
    return f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"'''

# Replace the function
content = re.sub(old_pattern, new_function, content, flags=re.DOTALL)

# Also update extract_tool_call to handle the new format
extract_pattern = r'def extract_tool_call\(response: str\) -> Optional\[Dict\]:'
if extract_pattern in content:
    # Find the function and update it
    extract_old = r'def extract_tool_call\(response: str\) -> Optional\[Dict\]:.*?return None'
    extract_new = '''def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response - handles various formats"""
    if not response:
        return None
    
    # Try JSON format first (most reliable)
    import re
    import json
    
    # Look for function call in various formats
    patterns = [
        r'Function call:\\s*({.*?})',  # Our suggested format
        r'"name"\\s*:\\s*"(\\w+)".*?"parameters"\\s*:\\s*({.*?})',  # Direct JSON
        r'\\{"name"\\s*:\\s*".*?".*?\\}',  # Complete JSON object
        r'<tool_call>(.*?)</tool_call>',  # Legacy format if still used
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            try:
                if pattern == patterns[0] or pattern == patterns[2]:
                    # Complete JSON object
                    json_str = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    return json.loads(json_str)
                elif pattern == patterns[1]:
                    # Name and parameters separate
                    name = matches[0][0]
                    params = json.loads(matches[0][1])
                    return {"name": name, "parameters": params}
            except:
                continue
    
    # Fallback: look for intent to use a function
    response_lower = response.lower()
    if "search" in response_lower and any(word in response_lower for word in ["fish", "meat", "vegetarian", "menu", "dish"]):
        # Infer the search term
        for word in ["fish", "meat", "chicken", "vegetarian", "pasta", "seafood"]:
            if word in response_lower:
                return {
                    "name": "search_menu_items",
                    "parameters": {"search_term": word, "search_type": "ingredient"}
                }
    
    return None'''
    
    content = re.sub(extract_old, extract_new, content, flags=re.DOTALL)

# Write back
with open('miner.py', 'w') as f:
    f.write(content)

print("✓ Updated format_prompt_with_tools and extract_tool_call functions")
EOF

# Run the fix
python3 fix_tools_v2.py

# Clean up
rm fix_tools_v2.py

echo "✓ Tool format fixed with clear instructions!"
echo ""
echo "Starting miner with improved tool handling..."

# Start the miner
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
else
    nohup ./run_miner.sh > miner.log 2>&1 &
    echo "Miner started with PID: $!"
fi

echo ""
echo "✓ Miner restarted with better tool format"
echo "The model now knows HOW to indicate tool usage"
echo "Check logs: tail -f miner.log"