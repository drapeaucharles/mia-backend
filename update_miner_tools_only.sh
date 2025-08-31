#!/bin/bash
# Minimal update to existing miner - ONLY fix tool format

cd /data/qwen-awq-miner

# Backup current miner
cp miner.py miner_backup.py

# Create a patch for the miner to use OpenAI tool format
cat > tool_format_patch.py << 'EOF'
# This patch updates format_prompt_with_tools to NOT ask for XML tags

def format_prompt_with_tools(prompt: str, tools: List[Dict] = None, context: Dict = None) -> str:
    """Format prompt with tool definitions for Qwen - OpenAI style"""
    if not tools:
        # Simple format without tools
        if context and context.get("system_prompt"):
            return f"<|im_start|>system\n{context['system_prompt']}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Build tool descriptions WITHOUT asking for XML tags
    tool_descriptions = []
    for tool in tools:
        params_str = json.dumps(tool.get("parameters", {}), indent=2)
        tool_descriptions.append(f"""Function: {tool['name']}
Description: {tool.get('description', '')}
Parameters: {params_str}""")
    
    tools_text = "\n\n".join(tool_descriptions)
    
    # Add context
    ctx = ""
    if context:
        if context.get("business_name"):
            ctx += f"\nYou are helping customers at {context['business_name']}."
        if context.get("system_prompt"):
            ctx = f"\n{context['system_prompt']}"
    
    # System prompt - just describe tools, don't ask for tags
    system_prompt = f"""You are a helpful assistant.{ctx}

You have access to these functions:
{tools_text}

When you need to use a function, naturally indicate which function and with what parameters."""
    
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
EOF

echo "✓ Created tool format patch"
echo ""
echo "To apply this patch:"
echo "1. Edit miner.py and replace the format_prompt_with_tools function"
echo "2. The ONLY change is removing the XML tag instructions"
echo "3. Everything else stays the same"
echo ""
echo "The miner will still:"
echo "- Poll MIA backend for jobs"
echo "- Process tools when provided"
echo "- Extract tool calls from responses"
echo "- Work exactly as before, just without asking for XML tags"