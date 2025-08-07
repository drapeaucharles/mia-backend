#!/bin/bash

# Fix the 422 submit error

echo "Fixing submit error in miners..."

# Fix miner_final.py if it exists
if [ -f "/opt/mia-gpu-miner/miner_final.py" ]; then
    echo "Fixing miner_final.py..."
    sed -i 's|"result": result.get("text", ""),.*"processing_time": 1.0|"result": {"response": result.get("text", ""), "tokens_generated": len(result.get("text", "").split()), "processing_time": 1.0}|' /opt/mia-gpu-miner/miner_final.py
fi

# Fix gpu_miner_fixed.py if it exists  
if [ -f "/opt/mia-gpu-miner/gpu_miner_fixed.py" ]; then
    echo "Fixing gpu_miner_fixed.py..."
    
    # Create a temporary Python script to fix the format
    cat > /tmp/fix_miner.py << 'PYEOF'
import re

with open('/opt/mia-gpu-miner/gpu_miner_fixed.py', 'r') as f:
    content = f.read()

# Find and fix the submit_result call
old_pattern = r'"result": \{\s*"success": True,\s*"response": result\.get\("text", ""\),\s*"tokens_generated": result\.get\("tokens_generated", 0\),\s*"model": result\.get\("model", "[^"]+"\)\s*\}'

new_text = '"result": {\n                                        "response": result.get("text", ""),\n                                        "tokens_generated": result.get("tokens_generated", 0),\n                                        "model": result.get("model", "Mistral-7B-OpenOrca-GPTQ")\n                                    }'

# Fix the pattern
content = re.sub(old_pattern, new_text, content)

# Also check for the simpler format
if '"result": result.get("text", "")' in content:
    content = content.replace(
        '"result": result.get("text", "")',
        '"result": {"response": result.get("text", ""), "tokens_generated": result.get("tokens_generated", 0)}'
    )

with open('/opt/mia-gpu-miner/gpu_miner_fixed.py', 'w') as f:
    f.write(content)

print("Fixed gpu_miner_fixed.py")
PYEOF

    python3 /tmp/fix_miner.py
fi

# Restart the miner if it's running
echo "Restarting miner..."
pkill -f "miner_final.py|gpu_miner_fixed.py" || true
sleep 2

echo "✅ Fixed! The submit error should be resolved."
echo ""
echo "To restart your miner:"
echo "cd /opt/mia-gpu-miner && source venv/bin/activate && python gpu_miner_fixed.py"