#!/bin/bash

# Update miner with the simplest language fix

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Updating miner with simple language fix...${NC}"
echo ""

# Check if we're in the right directory
if [ -d "/data/mia-gpu-miner" ]; then
    cd /data/mia-gpu-miner
elif [ -d "$HOME/mia-gpu-miner" ]; then
    cd $HOME/mia-gpu-miner
else
    echo -e "${RED}Error: Miner directory not found${NC}"
    exit 1
fi

# Activate venv
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Copy the new miners
echo -e "${YELLOW}Copying new miners...${NC}"
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_simple_fix.py ./
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_robust.py ./

chmod +x mia_miner_concurrent_simple_fix.py
chmod +x mia_miner_concurrent_robust.py

# Stop existing miner
echo -e "${YELLOW}Stopping existing miner...${NC}"
./stop_miner.sh 2>/dev/null || true
sleep 2

echo ""
echo -e "${GREEN}✓ Language fix miners installed!${NC}"
echo ""
echo -e "${YELLOW}Option 1: Simple Fix (RECOMMENDED)${NC}"
echo "Let the AI model detect language itself"
echo "Run: python3 mia_miner_concurrent_simple_fix.py"
echo ""
echo -e "${YELLOW}Option 2: Robust Detection${NC}"
echo "Multiple layers of language enforcement"
echo "Run: python3 mia_miner_concurrent_robust.py"
echo ""
echo -e "${GREEN}The simple fix just tells the model to:${NC}"
echo "• Respond in the same language as the user"
echo "• Default to English if unsure"
echo "• Never mix languages"
echo ""
echo "This should work because the AI model that can"
echo "understand the question surely knows what language it's in!"
echo ""
echo -e "${GREEN}To run in background:${NC}"
echo "  nohup python3 mia_miner_concurrent_simple_fix.py > /data/miner.log 2>&1 &"