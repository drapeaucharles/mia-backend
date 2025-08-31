#!/bin/bash
# Minimal fix - only update tool instruction in prompt

cd /data/qwen-awq-miner

# Create a simple patch
cat > patch_tools.py << 'EOF'
import re

# Read miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Only replace the tool instruction part
# Find the line that says "To use a tool, respond with:"
if 'To use a tool, respond with:' in content:
    # Replace just the instruction part
    content = content.replace(
        '''To use a tool, respond with:
<tool_call>
{"name": "tool_name", "parameters": {"param": "value"}}
</tool_call>''',
        '''To use a tool, call it naturally by saying something like:
"I'll search for that information" followed by:
{"name": "tool_name", "parameters": {"param": "value"}}'''
    )
    
    print("✓ Updated tool instruction")
else:
    print("✗ Could not find tool instruction to update")

# Write back
with open('miner.py', 'w') as f:
    f.write(content)
EOF

# Apply patch
python3 patch_tools.py
rm patch_tools.py

# Restart
pkill -f miner.py && ./start_miner.sh

echo "✓ Applied minimal fix - removed XML tags from instruction only"
echo "Everything else remains the same"