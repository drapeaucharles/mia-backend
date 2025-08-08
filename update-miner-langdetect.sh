#!/bin/bash

# Update miner with proper language detection

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Updating miner with proper language detection...${NC}"
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

# Install langdetect
echo -e "${YELLOW}Installing langdetect library...${NC}"
pip install langdetect

# Copy the new miner
echo -e "${YELLOW}Copying new miner with language detection...${NC}"
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_langdetect.py ./mia_miner_concurrent_langdetect.py
chmod +x mia_miner_concurrent_langdetect.py

# Stop existing miner
echo -e "${YELLOW}Stopping existing miner...${NC}"
./stop_miner.sh 2>/dev/null || true
sleep 2

echo ""
echo -e "${GREEN}✓ Miner updated with proper language detection!${NC}"
echo ""
echo "The new miner uses the langdetect library to accurately detect"
echo "the language of user input and respond in the same language."
echo ""
echo "To start the updated miner:"
echo "  python3 mia_miner_concurrent_langdetect.py"
echo ""
echo "Or to run in background:"
echo "  nohup python3 mia_miner_concurrent_langdetect.py > /data/miner.log 2>&1 &"
echo ""
echo "Language support includes:"
echo "• English, Spanish, French, German, Italian"
echo "• Portuguese, Russian, Japanese, Korean, Chinese"
echo "• Arabic, Hindi, Dutch, Polish, Turkish"
echo "• Vietnamese, Thai, Indonesian, Malay"