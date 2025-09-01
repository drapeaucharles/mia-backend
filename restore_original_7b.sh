#!/bin/bash
# Restore the ACTUAL original setup - 7B model

cd /data/qwen-awq-miner

# Fix miner.py to use the correct 7B model
echo "Restoring to Qwen2.5-7B-Instruct-AWQ..."
sed -i 's/Qwen2.5-32B-Instruct-AWQ/Qwen2.5-7B-Instruct-AWQ/g' miner.py

# Also fix the tool format while we're at it
python3 << 'EOF'
with open('miner.py', 'r') as f:
    content = f.read()

# Fix model name
content = content.replace('Qwen2.5-32B-Instruct-AWQ', 'Qwen2.5-7B-Instruct-AWQ')

# Fix tool format
content = content.replace(
    'To use a tool, respond with:\n<tool_call>\n{"name": "tool_name", "parameters": {"param": "value"}}\n</tool_call>',
    'To use a tool, respond with:\nI\'ll use the appropriate tool.\n{"name": "tool_name", "parameters": {"param": "value"}}'
)

with open('miner.py', 'w') as f:
    f.write(content)
EOF

echo "✅ Restored to original 7B model"
echo "✅ Fixed tool format (no XML tags)"
echo ""
echo "This is what you had working before I messed it up."
echo "Start with: ./start_miner.sh"