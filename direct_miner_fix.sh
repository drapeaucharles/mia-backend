#!/bin/bash
# Direct fix for miner tool format

cd /data/qwen-awq-miner

# Stop current miner
echo "Stopping current miner..."
pkill -f miner.py || true
sleep 2

# Backup current miner
cp miner.py miner_backup_$(date +%Y%m%d_%H%M%S).py

# Create a Python script to do precise replacement
cat > fix_miner_direct.py << 'EOF'
#!/usr/bin/env python3
import re

# Read miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Find the format_prompt_with_tools function
# Use a more precise pattern
func_start = content.find('def format_prompt_with_tools(')
if func_start == -1:
    print("ERROR: Could not find format_prompt_with_tools function")
    exit(1)

# Find the end of the function (next def or class)
func_end = content.find('\ndef ', func_start + 1)
if func_end == -1:
    func_end = content.find('\nclass ', func_start + 1)
if func_end == -1:
    func_end = len(content)

# Extract the old function
old_func = content[func_start:func_end]
print(f"Found function: {len(old_func)} chars")

# Create the new function
new_func = '''def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
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

To use a function, you MUST respond in exactly this format:
I'll search for that.
Function call: {{"name": "search_menu_items", "parameters": {{"search_term": "fish", "search_type": "ingredient"}}}}

The JSON must be on its own line starting with "Function call:"."""
    
    return f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
'''

# Replace the function
new_content = content[:func_start] + new_func + content[func_end:]

# Write it back
with open('miner.py', 'w') as f:
    f.write(new_content)

print("✓ Updated format_prompt_with_tools function")

# Now update extract_tool_call if it exists
if 'def extract_tool_call(' in content:
    # Read again with updated content
    with open('miner.py', 'r') as f:
        content = f.read()
    
    # Find extract_tool_call
    extract_start = content.find('def extract_tool_call(')
    if extract_start != -1:
        extract_end = content.find('\ndef ', extract_start + 1)
        if extract_end == -1:
            extract_end = content.find('\nclass ', extract_start + 1)
        if extract_end == -1:
            extract_end = len(content)
            
        # New extract function
        new_extract = '''def extract_tool_call(response: str) -> Optional[Dict]:
    """Extract tool call from response"""
    if not response:
        return None
    
    import re
    import json
    
    # Look for our specific format
    match = re.search(r'Function call:\s*({.*?})', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Also check for JSON anywhere in response
    try:
        # Find any JSON object that looks like a tool call
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, response)
        if matches:
            return json.loads(matches[0])
    except:
        pass
    
    return None
'''
        
        # Replace it
        new_content = content[:extract_start] + new_extract + content[extract_end:]
        
        with open('miner.py', 'w') as f:
            f.write(new_content)
        
        print("✓ Updated extract_tool_call function")

print("✓ All updates complete")
EOF

# Run the fix
python3 fix_miner_direct.py

# Verify the fix
echo -e "\n=== Verifying fix ===\n"
if grep -q "Function call:" miner.py; then
    echo "✓ miner.py now contains 'Function call:' instruction"
    grep -A 5 "Function call:" miner.py | head -10
else
    echo "✗ ERROR: Fix may have failed"
fi

# Clean up
rm fix_miner_direct.py

echo -e "\n✓ Fix complete!"
echo "Starting miner..."

# Start the miner
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
else
    nohup ./run_miner.sh > miner.log 2>&1 &
    echo "Miner started with PID: $!"
fi

echo ""
echo "✓ Miner restarted with proper tool format"
echo "Test it again - tools should work now!"