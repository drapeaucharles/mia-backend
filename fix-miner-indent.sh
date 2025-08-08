#!/bin/bash

# Fix indentation error in miner script

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Fixing indentation error in miner script...${NC}"

cd /data/mia-gpu-miner

# Create a fixed version of the miner
cat > fix_indentation.py << 'EOF'
#!/usr/bin/env python3
"""Fix indentation in miner script"""

# Read the current miner script
with open('mia_miner_unified.py', 'r') as f:
    lines = f.readlines()

# Find and fix the indentation issue around line 81
fixed_lines = []
for i, line in enumerate(lines):
    # Check for the problematic try block
    if line.strip() == 'try:' and i > 0:
        # Get the indentation of the previous non-empty line
        prev_indent = 0
        for j in range(i-1, -1, -1):
            if lines[j].strip():
                prev_indent = len(lines[j]) - len(lines[j].lstrip())
                break
        
        # Ensure proper indentation
        if len(line) - len(line.lstrip()) != prev_indent + 4:
            line = ' ' * (prev_indent + 4) + 'try:\n'
    
    fixed_lines.append(line)

# Write the fixed version
with open('mia_miner_unified_fixed.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Fixed indentation issues")
print("✓ Created mia_miner_unified_fixed.py")
EOF

python3 fix_indentation.py

# Backup the broken version
cp mia_miner_unified.py mia_miner_unified_broken.py

# Replace with fixed version
mv mia_miner_unified_fixed.py mia_miner_unified.py

# Make it executable
chmod +x mia_miner_unified.py

echo -e "${GREEN}✓ Indentation fixed!${NC}"
echo ""
echo "Now restart the miner:"
echo "  ./stop_miner.sh"
echo "  ./start_miner.sh"