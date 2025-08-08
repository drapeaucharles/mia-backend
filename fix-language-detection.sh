#!/bin/bash

# Fix language detection in all miners

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Fixing language detection in miners...${NC}"

cd /data/mia-gpu-miner

# Update concurrent miner
if [ -f "mia_miner_concurrent.py" ]; then
    echo -e "${YELLOW}Updating concurrent miner...${NC}"
    
    # Fix the system message
    sed -i 's/You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses in multiple languages./You are MIA, a helpful AI assistant. IMPORTANT: Always respond in the SAME LANGUAGE as the user'\''s message. If the user writes in English, respond in English. If they write in Spanish, respond in Spanish. Match the user'\''s language exactly./g' mia_miner_concurrent.py
fi

# Update vLLM AWQ miner
if [ -f "mia_miner_vllm_awq.py" ]; then
    echo -e "${YELLOW}Updating vLLM AWQ miner...${NC}"
    
    sed -i 's/You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses in multiple languages./You are MIA, a helpful AI assistant. IMPORTANT: Always respond in the SAME LANGUAGE as the user'\''s message. If the user writes in English, respond in English. If they write in Spanish, respond in Spanish. Match the user'\''s language exactly./g' mia_miner_vllm_awq.py
fi

# Update simple miner
if [ -f "mia_miner_simple.py" ]; then
    echo -e "${YELLOW}Updating simple miner...${NC}"
    
    sed -i 's/You are MIA, a helpful multilingual assistant./You are MIA, a helpful AI assistant. IMPORTANT: Always respond in the SAME LANGUAGE as the user'\''s message. Match the user'\''s language exactly./g' mia_miner_simple.py
fi

# Update fast miner
if [ -f "mia_miner_fast.py" ]; then
    echo -e "${YELLOW}Updating fast miner...${NC}"
    
    sed -i 's/You are MIA, a helpful multilingual assistant./You are MIA, a helpful AI assistant. IMPORTANT: Always respond in the SAME LANGUAGE as the user'\''s message. Match the user'\''s language exactly./g' mia_miner_fast.py
fi

echo ""
echo -e "${GREEN}✓ Language detection fixed!${NC}"
echo ""
echo "Now restart your miner:"
echo "  ./stop_miner.sh"
echo "  ./start_miner.sh  # or python3 mia_miner_concurrent.py"
echo ""
echo "The AI will now respond in the same language as the user!"