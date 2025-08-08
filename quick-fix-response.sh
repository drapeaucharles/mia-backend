#!/bin/bash

# Quick fix for response extraction in running miners
echo "Applying quick fix for response extraction..."

# Create a fixed generation function
cat > /tmp/fix_generation.py << 'EOF'
import sys
import os

# Find the miner file
miner_files = [
    "/opt/mia-gpu-miner/mia_miner_unified.py",
    "/opt/mia-gpu-miner/mia_miner_unified_fixed.py",
    os.path.expanduser("~/mia-gpu-miner/mia_miner_unified.py"),
    os.path.expanduser("~/mia-gpu-miner/mia_miner_unified_fixed.py")
]

miner_file = None
for f in miner_files:
    if os.path.exists(f):
        miner_file = f
        break

if not miner_file:
    print("Error: No miner file found")
    sys.exit(1)

print(f"Fixing: {miner_file}")

# Read the file
with open(miner_file, 'r') as f:
    content = f.read()

# Find and replace the response extraction
import re

# Pattern to find the decode section
pattern = r'(\s+)# Decode response\s*\n\s+full_response = tokenizer\.decode.*?\n.*?response = response\.replace\("<\|im_end\|>", ""\)\.strip\(\)'
replacement = r'''\1# Decode the generated tokens only (not including the input)
\1generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
\1response = tokenizer.decode(generated_ids, skip_special_tokens=True)
\1
\1# Clean up any remaining special tokens that might have slipped through
\1response = response.replace("<|im_end|>", "").strip()
\1response = response.replace("<|im_start|>", "").strip()'''

# Try to replace
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

if new_content == content:
    # Try alternative pattern
    pattern2 = r'# Decode response.*?response\.replace\("<\|im_end\|>", ""\)\.strip\(\)'
    replacement2 = '''# Decode the generated tokens only (not including the input)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any remaining special tokens that might have slipped through
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>", "").strip()'''
    
    new_content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

if new_content != content:
    # Backup original
    backup_file = miner_file + '.backup'
    with open(backup_file, 'w') as f:
        f.write(content)
    print(f"Backup saved to: {backup_file}")
    
    # Write fixed version
    with open(miner_file, 'w') as f:
        f.write(new_content)
    print("✓ Fixed response extraction!")
else:
    print("Warning: Could not find the pattern to replace")
    print("Manual fix may be needed")
EOF

# Run the fix
python3 /tmp/fix_generation.py

# Clean up
rm -f /tmp/fix_generation.py

# Restart service if it exists
if systemctl is-active --quiet mia-miner; then
    echo "Restarting mia-miner service..."
    sudo systemctl restart mia-miner
    echo "✓ Service restarted"
elif systemctl is-active --quiet mia-gpu-miner; then
    echo "Restarting mia-gpu-miner service..."
    sudo systemctl restart mia-gpu-miner
    echo "✓ Service restarted"
else
    echo ""
    echo "Please restart your miner manually to apply the fix:"
    echo "  - If running in screen: exit and restart"
    echo "  - If running with nohup: kill the process and restart"
fi

echo ""
echo "✓ Response extraction fix applied!"
echo ""
echo "The model should now return complete responses:"
echo "  'Hello' -> 'Hello! How can I assist you today?'"
echo "  (Instead of: 'to assist you with any questions...')"