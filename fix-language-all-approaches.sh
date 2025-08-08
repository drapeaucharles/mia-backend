#!/bin/bash

# Comprehensive language detection fix with multiple approaches

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Language Detection Fix - All Approaches         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
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

# Copy all new miners
echo -e "${YELLOW}Copying language detection miners...${NC}"
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_simple_detect.py ./ 2>/dev/null || true
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_prefix.py ./ 2>/dev/null || true
cp /home/charles-drapeau/Documents/Project/mia-backend/mia_miner_concurrent_langdetect.py ./ 2>/dev/null || true

chmod +x mia_miner_concurrent_*.py

# Stop existing miner
echo -e "${YELLOW}Stopping existing miner...${NC}"
./stop_miner.sh 2>/dev/null || true
sleep 2

echo ""
echo -e "${GREEN}✓ Language detection miners installed!${NC}"
echo ""
echo -e "${BLUE}Available approaches:${NC}"
echo ""
echo -e "${YELLOW}1. Simple Keyword Detection (Recommended)${NC}"
echo "   - No external dependencies"
echo "   - Fast and reliable"
echo "   - Run: python3 mia_miner_concurrent_simple_detect.py"
echo ""
echo -e "${YELLOW}2. Language Prefix Approach${NC}"
echo "   - Forces language with response prefix"
echo "   - Good for stubborn models"
echo "   - Run: python3 mia_miner_concurrent_prefix.py"
echo ""
echo -e "${YELLOW}3. Advanced Detection (langdetect)${NC}"
echo "   - Most accurate detection"
echo "   - Requires: pip install langdetect"
echo "   - Run: python3 mia_miner_concurrent_langdetect.py"
echo ""
echo -e "${GREEN}Quick test commands:${NC}"
echo ""
echo "Test English:"
echo '  curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '"'"'{"prompt":"Hello, how are you?","max_tokens":50}'"'"
echo ""
echo "Test Spanish:"
echo '  curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '"'"'{"prompt":"Hola, ¿cómo estás?","max_tokens":50}'"'"
echo ""
echo "Test French:"
echo '  curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '"'"'{"prompt":"Bonjour, comment allez-vous?","max_tokens":50}'"'"
echo ""
echo -e "${GREEN}To run in background:${NC}"
echo "  nohup python3 mia_miner_concurrent_simple_detect.py > /data/miner.log 2>&1 &"
echo ""