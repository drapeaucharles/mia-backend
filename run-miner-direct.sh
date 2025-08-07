#!/bin/bash

# Direct miner runner - no systemd required

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}MIA Miner Direct Runner${NC}"
echo "======================="

# Check if installation exists
if [ ! -d "/opt/mia-gpu-miner/venv" ]; then
    echo -e "${RED}Error: Miner not installed at /opt/mia-gpu-miner${NC}"
    echo "Please run the installer first."
    exit 1
fi

cd /opt/mia-gpu-miner
source venv/bin/activate

# Check if model is downloaded
echo -e "${YELLOW}Checking for Mistral 7B model...${NC}"
python -c "
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', local_files_only=True)
    print('✓ Model already downloaded')
except:
    print('Model not found, downloading now...')
    print('This is a 14GB download and will take 10-20 minutes.')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', torch_dtype='auto', low_cpu_mem_usage=True)
    print('✓ Model downloaded successfully!')
"

# Start vLLM server in background
echo -e "\n${YELLOW}Starting vLLM server...${NC}"
if pgrep -f "vllm_server.py" > /dev/null; then
    echo "vLLM server already running"
else
    nohup python vllm_server.py > vllm.log 2>&1 &
    echo $! > vllm.pid
    echo "vLLM server started (PID: $(cat vllm.pid))"
    echo "Waiting 30 seconds for model to load..."
    sleep 30
fi

# Start miner
echo -e "\n${YELLOW}Starting GPU miner...${NC}"
echo "Logs will be saved to miner.log"
echo ""
echo "Commands:"
echo "  View logs: tail -f /opt/mia-gpu-miner/miner.log"
echo "  Stop miner: pkill -f gpu_miner.py"
echo "  Stop vLLM: pkill -f vllm_server.py"
echo ""

# Run miner in foreground so you can see output
python gpu_miner.py 2>&1 | tee miner.log