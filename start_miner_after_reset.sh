#!/bin/bash
# Start miner after container reset

cd /data/qwen-awq-miner

echo "=== Starting Miner After Reset ==="

# 1. Test CUDA is working
echo "1. Testing CUDA..."
source venv/bin/activate
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    x = torch.zeros(1).cuda()
    print('✓ CUDA is working!')
else:
    print('✗ CUDA still not available')
"

# 2. Check if tool format fix was applied
echo -e "\n2. Checking tool format..."
if grep -q '<tool_call>' miner.py; then
    echo "✗ Old XML format still present - applying fix..."
    python -c "
with open('miner.py', 'r') as f:
    content = f.read()
content = content.replace(
    'To use a tool, respond with:\\n<tool_call>\\n{\"name\": \"tool_name\", \"parameters\": {\"param\": \"value\"}}\\n</tool_call>',
    'To use a tool, respond naturally by saying something like:\\n\"I\\'ll search for that information\" followed by:\\n{\"name\": \"tool_name\", \"parameters\": {\"param\": \"value\"}}'
)
with open('miner.py', 'w') as f:
    f.write(content)
print('✓ Fixed tool format')
"
else
    echo "✓ Tool format already fixed"
fi

# 3. Start the miner
echo -e "\n3. Starting miner..."
export CUDA_VISIBLE_DEVICES=0
nohup python miner.py > miner.log 2>&1 &
MINER_PID=$!

echo -e "\n=== Miner Started ==="
echo "PID: $MINER_PID"
echo ""
echo "Monitor with: tail -f miner.log"
echo "Look for:"
echo "  - 'Model loaded successfully'"
echo "  - 'tool_call: true' when testing tools"

# 4. Wait and show initial logs
sleep 5
echo -e "\n=== Initial Logs ==="
tail -20 miner.log