#!/bin/bash
# Simple debug for job content

cd /data/qwen-awq-miner

# Restore from backup first
if [ -f "miner_backup_*.py" ]; then
    LATEST_BACKUP=$(ls -t miner_backup_*.py | head -1)
    echo "Restoring from $LATEST_BACKUP"
    cp "$LATEST_BACKUP" miner.py
fi

# Add simple debugging
cat > add_debug.py << 'EOF'
# Read miner.py
with open('miner.py', 'r') as f:
    lines = f.readlines()

# Find the line and add debug after it
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if 'logger.info(f"Processing job: {work[' in line and 'request_id' in line:
        # Add debug lines with proper indentation
        indent = ' ' * 20  # Match the indentation
        new_lines.append(indent + '# Debug job content\n')
        new_lines.append(indent + 'prompt_preview = work.get("prompt", "")[:50] + "..."\n')
        new_lines.append(indent + 'has_tools = "tools" in work\n')
        new_lines.append(indent + 'logger.info(f"DEBUG - Prompt: {prompt_preview}")\n')
        new_lines.append(indent + 'logger.info(f"DEBUG - Has tools: {has_tools}")\n')
        new_lines.append(indent + 'if has_tools:\n')
        new_lines.append(indent + '    tool_names = [t.get("name", "?") for t in work.get("tools", [])]\n')
        new_lines.append(indent + '    logger.info(f"DEBUG - Tool names: {tool_names}")\n')

# Write back
with open('miner.py', 'w') as f:
    f.writelines(new_lines)

print("✓ Added debugging")
EOF

python3 add_debug.py
rm add_debug.py

echo "✓ Debug added successfully"
echo ""
echo "Restarting miner..."

# Restart
pkill -f miner.py
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
else
    nohup ./run_miner.sh > miner.log 2>&1 &
fi

echo ""
echo "✓ Miner restarted"
echo "Watch logs with: tail -f miner.log | grep DEBUG"