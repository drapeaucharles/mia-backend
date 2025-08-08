#!/bin/bash

# Test concurrent miner with dynamic VRAM management

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Testing Concurrent Miner with Dynamic VRAM Management${NC}"
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

# Install GPUtil if not present
pip install gputil 2>/dev/null || true

# Stop existing miner
echo -e "${YELLOW}Stopping existing miner...${NC}"
./stop_miner.sh 2>/dev/null || true
sleep 2

# Start concurrent miner
echo -e "${YELLOW}Starting concurrent miner...${NC}"
echo ""
echo "Features:"
echo "• Dynamic VRAM management"
echo "• Concurrent job processing" 
echo "• Adaptive polling (0.5-2 seconds)"
echo "• Performance tracking"
echo ""

# Run the concurrent miner
python3 mia_miner_concurrent.py