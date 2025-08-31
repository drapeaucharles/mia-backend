#!/bin/bash
# Update script for MIA miner with OpenAI tools support

echo "=== MIA Miner Update Script ==="
echo "This will stop your current miner, update, and start with OpenAI tools support"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Find the installation directory
if [ -d "/data/mia-gpu-miner" ]; then
    MINER_DIR="/data/mia-gpu-miner"
    BACKEND_DIR="/data/mia-backend"
elif [ -d "$HOME/mia-gpu-miner" ]; then
    MINER_DIR="$HOME/mia-gpu-miner"
    BACKEND_DIR="$HOME/mia-backend"
else
    echo -e "${RED}Error: Cannot find miner installation directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Found miner at: $MINER_DIR${NC}"

# Step 1: Stop current miner
echo -e "\n${YELLOW}Step 1: Stopping current miner...${NC}"
if [ -f "$MINER_DIR/stop_miner.sh" ]; then
    cd "$MINER_DIR" && ./stop_miner.sh
else
    # Try to kill any running miner process
    pkill -f "miner.py" || true
    pkill -f "vllm.entrypoints" || true
fi
sleep 2

# Step 2: Update mia-backend repository
echo -e "\n${YELLOW}Step 2: Updating mia-backend repository...${NC}"
if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${YELLOW}Cloning mia-backend repository...${NC}"
    cd "$(dirname "$BACKEND_DIR")"
    git clone https://github.com/drapeaucharles/mia-backend.git
fi

cd "$BACKEND_DIR"
git fetch origin
git checkout master
git pull origin master

# Step 3: Copy new scripts to miner directory
echo -e "\n${YELLOW}Step 3: Installing OpenAI tools support...${NC}"
if [ -f "$BACKEND_DIR/start_vllm_openai_server.sh" ]; then
    cp "$BACKEND_DIR/start_vllm_openai_server.sh" "$MINER_DIR/"
    chmod +x "$MINER_DIR/start_vllm_openai_server.sh"
fi

if [ -f "$BACKEND_DIR/miner_openai_tools.py" ]; then
    cp "$BACKEND_DIR/miner_openai_tools.py" "$MINER_DIR/"
fi

# Step 4: Start vLLM with OpenAI API
echo -e "\n${YELLOW}Step 4: Starting vLLM with OpenAI API support...${NC}"
cd "$MINER_DIR"

# Check if venv exists
if [ ! -d "venv" ] && [ -d "/data/venv" ]; then
    ln -s /data/venv venv
fi

# Start the OpenAI-compatible server
if [ -f "start_vllm_openai_server.sh" ]; then
    echo -e "${GREEN}Starting vLLM OpenAI server...${NC}"
    ./start_vllm_openai_server.sh
else
    echo -e "${RED}Error: start_vllm_openai_server.sh not found${NC}"
    echo "Falling back to regular miner..."
    if [ -f "start_miner.sh" ]; then
        ./start_miner.sh
    fi
fi

echo -e "\n${GREEN}Update complete!${NC}"
echo -e "To check if the server is running:"
echo -e "  curl http://localhost:8000/v1/models"
echo -e "\nTo view logs:"
echo -e "  tail -f $MINER_DIR/vllm_server.log"