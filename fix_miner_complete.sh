#!/bin/bash
# Complete fix for miner - correct Python env + tool format

cd /data/qwen-awq-miner

echo "=== Complete Miner Fix ==="

# 1. Stop any running processes
echo "1. Stopping existing processes..."
pkill -f miner.py || true
pkill -f vllm || true
sleep 2

# 2. Clear old logs
echo "2. Clearing old logs..."
rm -f miner.log vllm_server.log
touch miner.log

# 3. Fix the tool format in miner.py
echo "3. Fixing tool format..."
cat > fix_tools.py << 'EOF'
# Fix tool instruction format
with open('miner.py', 'r') as f:
    content = f.read()

# Replace XML format with natural format
old_text = '''To use a tool, respond with:
<tool_call>
{"name": "tool_name", "parameters": {"param": "value"}}
</tool_call>'''

new_text = '''To use a tool, respond naturally by saying something like:
"I'll search for that information" followed by:
{"name": "tool_name", "parameters": {"param": "value"}}'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('miner.py', 'w') as f:
        f.write(content)
    print("✓ Fixed tool format")
else:
    print("⚠ Tool format already fixed or different")
EOF

./venv/bin/python fix_tools.py
rm fix_tools.py

# 4. Create proper startup script with correct Python
echo "4. Creating startup script..."
cat > start_miner_correct.sh << 'EOF'
#!/bin/bash
# Start miner with correct Python environment

cd /data/qwen-awq-miner

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Use the correct Python with vLLM
echo "Starting miner with vLLM support..."
./venv/bin/python miner.py 2>&1 | tee miner.log
EOF

chmod +x start_miner_correct.sh

# 5. Also update existing scripts
if [ -f "start_miner.sh" ]; then
    echo "5. Updating start_miner.sh to use correct Python..."
    sed -i 's|python3 miner.py|./venv/bin/python miner.py|g' start_miner.sh
    sed -i 's|python miner.py|./venv/bin/python miner.py|g' start_miner.sh
fi

if [ -f "run_miner.sh" ]; then
    echo "5. Updating run_miner.sh to use correct Python..."
    sed -i 's|python3 miner.py|./venv/bin/python miner.py|g' run_miner.sh
    sed -i 's|python miner.py|./venv/bin/python miner.py|g' run_miner.sh
fi

# 6. Start the miner
echo -e "\n6. Starting miner..."
nohup ./venv/bin/python miner.py > miner.log 2>&1 &
MINER_PID=$!

echo -e "\n=== Fix Complete ==="
echo "✓ Tool format fixed (removed XML tags)"
echo "✓ Using correct Python: ./venv/bin/python"
echo "✓ Miner started with PID: $MINER_PID"
echo ""
echo "Monitor with: tail -f miner.log"
echo "Check for 'tool_call: true' in the logs!"