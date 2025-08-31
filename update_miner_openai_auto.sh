#!/bin/bash
# Auto-detect and update MIA miner with OpenAI tools support

echo "=== MIA Miner Auto-Update Script ==="
echo "This will stop your current miner, update, and start with OpenAI tools support"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Auto-detect miner directory
echo -e "${YELLOW}Detecting miner installation...${NC}"
MINER_DIR=""

# Check common locations
for dir in "/data/qwen-awq-miner" "/data/mia-gpu-miner" "$HOME/mia-gpu-miner" "$HOME/qwen-awq-miner"; do
    if [ -d "$dir" ] && [ -f "$dir/miner.py" ]; then
        MINER_DIR="$dir"
        break
    fi
done

if [ -z "$MINER_DIR" ]; then
    echo -e "${RED}Error: Cannot find miner installation directory${NC}"
    echo "Checked: /data/qwen-awq-miner, /data/mia-gpu-miner, ~/mia-gpu-miner, ~/qwen-awq-miner"
    exit 1
fi

echo -e "${GREEN}Found miner at: $MINER_DIR${NC}"

# Step 1: Stop current miner
echo -e "\n${YELLOW}Step 1: Stopping current miner...${NC}"
cd "$MINER_DIR"

# Try multiple stop methods
if [ -f "stop_miner.sh" ]; then
    ./stop_miner.sh
elif [ -f "stop.sh" ]; then
    ./stop.sh
else
    # Manually kill processes
    echo "No stop script found, killing processes manually..."
    pkill -f "miner.py" || true
    pkill -f "vllm.entrypoints" || true
    pkill -f "qwen.*miner" || true
fi
sleep 2

# Step 2: Update/Clone mia-backend repository
echo -e "\n${YELLOW}Step 2: Setting up mia-backend repository...${NC}"
BACKEND_DIR="/data/mia-backend"

if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${YELLOW}Cloning mia-backend repository...${NC}"
    cd /data
    git clone https://github.com/drapeaucharles/mia-backend.git
fi

cd "$BACKEND_DIR"
git fetch origin
git checkout master
git pull origin master

# Step 3: Copy OpenAI tools scripts
echo -e "\n${YELLOW}Step 3: Installing OpenAI tools support...${NC}"

# Copy the vLLM OpenAI server script
if [ -f "$BACKEND_DIR/start_vllm_openai_server.sh" ]; then
    cp "$BACKEND_DIR/start_vllm_openai_server.sh" "$MINER_DIR/"
    chmod +x "$MINER_DIR/start_vllm_openai_server.sh"
    echo -e "${GREEN}✓ Installed start_vllm_openai_server.sh${NC}"
fi

# Copy the OpenAI tools miner
if [ -f "$BACKEND_DIR/miner_openai_tools.py" ]; then
    cp "$BACKEND_DIR/miner_openai_tools.py" "$MINER_DIR/"
    echo -e "${GREEN}✓ Installed miner_openai_tools.py${NC}"
fi

# Step 4: Update configuration if needed
echo -e "\n${YELLOW}Step 4: Updating configuration...${NC}"
cd "$MINER_DIR"

# Check if venv exists and link if needed
if [ ! -d "venv" ]; then
    if [ -d "/data/venv" ]; then
        ln -s /data/venv venv
    elif [ -d "../venv" ]; then
        ln -s ../venv venv
    fi
fi

# Step 5: Start vLLM with OpenAI API
echo -e "\n${YELLOW}Step 5: Starting vLLM with OpenAI API support...${NC}"

if [ -f "start_vllm_openai_server.sh" ]; then
    echo -e "${GREEN}Starting vLLM OpenAI server...${NC}"
    echo -e "${YELLOW}This will start the server with:${NC}"
    echo "  - OpenAI-compatible API on port 8000"
    echo "  - Qwen2.5-7B-Instruct-AWQ model"
    echo "  - 12k context window"
    echo "  - Tool calling support"
    echo ""
    ./start_vllm_openai_server.sh
else
    echo -e "${RED}Error: start_vllm_openai_server.sh not found${NC}"
    echo -e "${YELLOW}Falling back to regular miner...${NC}"
    if [ -f "start_miner.sh" ]; then
        ./start_miner.sh
    elif [ -f "run_miner.sh" ]; then
        ./run_miner.sh
    else
        echo -e "${RED}No start script found!${NC}"
    fi
fi

echo -e "\n${GREEN}=== Update Complete! ===${NC}"
echo -e "\nTo verify the OpenAI server is running:"
echo -e "  ${YELLOW}curl http://localhost:8000/v1/models${NC}"
echo -e "\nTo check server logs:"
echo -e "  ${YELLOW}tail -f $MINER_DIR/vllm_server.log${NC}"
echo -e "\nTo check if it's processing:"
echo -e "  ${YELLOW}nvidia-smi${NC}"