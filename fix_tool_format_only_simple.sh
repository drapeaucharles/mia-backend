#!/bin/bash
# Fix ONLY the tool format in existing miner - nothing else

cd /data/qwen-awq-miner

echo "🔧 Fixing tool format in existing miner"
echo "======================================"

# Backup current miner.py
cp miner.py miner_backup_$(date +%Y%m%d_%H%M%S).py

# Fix the tool instruction format
python3 << 'EOF'
# Read miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Replace XML format with natural format
old_text = '''To use a tool, respond with:
<tool_call>
{"name": "tool_name", "parameters": {"param": "value"}}
</tool_call>'''

new_text = '''To use a tool, respond with:
I'll use the appropriate tool.
{"name": "tool_name", "parameters": {"param": "value"}}'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('miner.py', 'w') as f:
        f.write(content)
    print("✅ Fixed tool format - removed XML tags")
else:
    print("⚠️  Tool format already fixed or different")

EOF

# Restart miner
echo -e "\nRestarting miner..."
pkill -f miner.py
./start_miner.sh

echo -e "\n✅ Done! The miner should now return tool_call: true"
echo "Check logs: tail -f miner.log"