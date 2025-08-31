#!/bin/bash
# Minimal fix - ONLY change the tool instruction format
# This preserves everything else to avoid breaking the miner

cd /data/qwen-awq-miner

# Stop miner
echo "Stopping miner..."
pkill -f miner.py || true
sleep 2

# Backup
cp miner.py miner_backup_$(date +%Y%m%d_%H%M%S).py

# Create the fix script
cat > fix_tool_format.py << 'EOF'
#!/usr/bin/env python3
import re

# Read miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Find the exact text to replace
old_instruction = '''To use a tool, respond with:
<tool_call>
{"name": "tool_name", "parameters": {"param": "value"}}
</tool_call>'''

new_instruction = '''To use a tool, respond in this format:
I'll use the appropriate tool.
{"name": "tool_name", "parameters": {"param": "value"}}'''

# Simple replacement
if old_instruction in content:
    content = content.replace(old_instruction, new_instruction)
    print("✓ Updated tool instruction format")
    
    # Write back
    with open('miner.py', 'w') as f:
        f.write(content)
else:
    print("✗ Could not find the exact tool instruction to replace")
    print("The miner may already be updated or has a different format")

EOF

# Apply the fix
python3 fix_tool_format.py
rm fix_tool_format.py

# Verify
echo -e "\n=== Verifying change ==="
if grep -q '<tool_call>' miner.py; then
    echo "✗ XML tags still present"
else
    echo "✓ XML tags removed"
fi

# Restart
echo -e "\nRestarting miner..."
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
else
    nohup ./run_miner.sh > miner.log 2>&1 &
fi

echo -e "\n✓ Minimal fix applied - only changed tool instruction format"
echo "The miner should now return tool_call: true when tools are provided"