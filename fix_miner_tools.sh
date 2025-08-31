#!/bin/bash
# Fix tool format in existing miner - minimal change

cd /data/qwen-awq-miner

# Stop current miner
echo "Stopping current miner..."
pkill -f miner.py || true
sleep 2

# Backup current miner
cp miner.py miner_backup_$(date +%Y%m%d_%H%M%S).py

# Update the format_prompt_with_tools function
cat > fix_tools.py << 'EOF'
import sys
import re

# Read the current miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Find and replace the format_prompt_with_tools function
old_pattern = r'def format_prompt_with_tools\(.*?\):\s*""".*?""".*?return.*?<\|im_start\|>assistant\\n"'
new_function = '''def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    """Format prompt with tool definitions for Qwen - OpenAI style"""
    if not tools:
        # Simple format without tools
        if context and context.get("system_prompt"):
            return f"<|im_start|>system\\n{context['system_prompt']}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        return f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    
    # Build tool descriptions WITHOUT asking for XML tags
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
    
    # System prompt - just describe tools, don't ask for tags
    system_prompt = f"""You are a helpful assistant.{ctx}

You have access to these functions:
{tools_text}

When you need to use a function, call it with the appropriate parameters."""
    
    return f"<|im_start|>system\\n{system_prompt}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"'''

# Replace the function
content = re.sub(old_pattern, new_function, content, flags=re.DOTALL)

# Write back
with open('miner.py', 'w') as f:
    f.write(content)

print("✓ Updated format_prompt_with_tools function")
EOF

# Run the fix
python3 fix_tools.py

# Clean up
rm fix_tools.py

echo "✓ Tool format fixed!"
echo ""
echo "Starting miner with fixed tool format..."

# Start the miner
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
else
    nohup ./run_miner.sh > miner.log 2>&1 &
    echo "Miner started with PID: $!"
fi

echo ""
echo "✓ Miner restarted with proper tool format"
echo "Check logs: tail -f miner.log"