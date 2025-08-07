#!/bin/bash

# MIA GPU Miner Minimal Setup - For servers with limited disk space
# Cleans up aggressively during installation

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}MIA GPU Miner - Minimal Space Setup${NC}"
echo "===================================="

# Check initial space
echo -e "\n${YELLOW}Current disk usage:${NC}"
df -h /

# Clean up first
echo -e "\n${YELLOW}[1/9] Cleaning up disk space...${NC}"
sudo apt-get clean
sudo apt-get autoremove -y
sudo rm -rf /var/cache/apt/archives/*
sudo rm -rf ~/.cache/pip
sudo rm -rf /tmp/*
sudo journalctl --vacuum-time=1d

# Check space after cleanup
echo -e "\n${YELLOW}Disk space after cleanup:${NC}"
df -h /

# Minimum space check (need at least 20GB free)
AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
AVAILABLE_GB=$(echo $AVAILABLE_SPACE | sed 's/G//')

if (( $(echo "$AVAILABLE_GB < 20" | bc -l) )); then
    echo -e "${RED}ERROR: Not enough disk space. Need at least 20GB free.${NC}"
    echo -e "Available: ${AVAILABLE_GB}GB"
    exit 1
fi

# Detect sudo
if [ "$EUID" -eq 0 ]; then
    SUDO_CMD=""
else
    SUDO_CMD="sudo"
fi

# Step 2: Quick system update
echo -e "\n${YELLOW}[2/9] Quick system update...${NC}"
$SUDO_CMD apt-get update -qq

# Step 3: Check NVIDIA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA drivers not found!${NC}"
    echo "Install with: sudo apt-get install nvidia-driver-535"
    echo "Then reboot and run this script again."
    exit 1
else
    echo -e "\n${GREEN}[3/9] NVIDIA drivers found${NC}"
    nvidia-smi || true
fi

# Step 4: Install minimal dependencies
echo -e "\n${YELLOW}[4/9] Installing minimal dependencies...${NC}"
$SUDO_CMD apt-get install -y -qq \
    python3 python3-venv python3-dev python3-pip \
    git curl wget build-essential

# Step 5: Setup directory
INSTALL_DIR="/opt/mia-gpu-miner"
echo -e "\n${YELLOW}[5/9] Creating installation directory...${NC}"
$SUDO_CMD rm -rf "$INSTALL_DIR"
$SUDO_CMD mkdir -p "$INSTALL_DIR"
$SUDO_CMD chown -R $USER:$USER "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Step 6: Create venv
echo -e "\n${YELLOW}[6/9] Setting up Python environment...${NC}"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools --no-cache-dir

# Step 7: Install packages one by one with cleanup
echo -e "\n${YELLOW}[7/9] Installing AI packages (space-efficient mode)...${NC}"

# Install PyTorch
echo "Installing PyTorch..."
pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install critical packages only
echo "Installing core packages..."
pip install --no-cache-dir \
    numpy==1.24.3 \
    transformers==4.36.2 \
    vllm==0.2.7

echo "Installing server packages..."
pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    requests==2.31.0 \
    psutil==5.9.6

# Clean pip cache after each major install
rm -rf ~/.cache/pip

# Step 8: Create minimal scripts
echo -e "\n${YELLOW}[8/9] Creating server scripts...${NC}"

# Minimal vLLM server
cat > "$INSTALL_DIR/vllm_server.py" << 'EOF'
#!/usr/bin/env python3
import os
import torch
from vllm import LLM, SamplingParams
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 500
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    text: str
    model: str

# Use smaller model for space constraints
MODEL = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
llm = None

@app.on_event("startup")
async def startup():
    global llm
    print(f"[vLLM] Starting up...")
    print(f"[vLLM] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[vLLM] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[vLLM] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"[vLLM] Loading model: {MODEL}")
    print(f"[vLLM] This is a 4GB GPTQ quantized model")
    print(f"[vLLM] First time download may take 5-10 minutes...")
    
    try:
        # GPTQ quantized model uses less disk space
        llm = LLM(
            model=MODEL,
            quantization="gptq",
            dtype="float16",
            gpu_memory_utilization=0.9,
            download_dir="/opt/mia-gpu-miner/models",
            trust_remote_code=True
        )
        print(f"[vLLM] ✓ Model loaded!")
        print(f"[vLLM] Ready to serve requests on port 8000")
    except Exception as e:
        print(f"[vLLM] ❌ Failed to load model: {e}")
        raise

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL}

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not llm:
        raise HTTPException(503, "Model not loaded")
    
    # Format prompt for chat
    formatted_prompt = f"User: {req.prompt}\nAssistant:"
    
    params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens
    )
    
    outputs = llm.generate([formatted_prompt], params)
    
    return GenerateResponse(
        text=outputs[0].outputs[0].text,
        model=MODEL
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Minimal miner
cat > "$INSTALL_DIR/gpu_miner.py" << 'EOF'
#!/usr/bin/env python3
import os
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIA_URL = os.getenv("MIA_BACKEND_URL", "https://mia-backend.up.railway.app")
VLLM_URL = "http://localhost:8000"

class MIAMiner:
    def __init__(self):
        self.miner_id = None
        self.session = requests.Session()
    
    def wait_vllm(self):
        for i in range(60):
            try:
                r = requests.get(f"{VLLM_URL}/health")
                if r.status_code == 200:
                    logger.info("vLLM ready!")
                    return True
            except:
                pass
            time.sleep(5)
        return False
    
    def register(self):
        try:
            ip = requests.get('https://api.ipify.org').text
            logger.info(f"Public IP: {ip}")
            
            data = {
                "wallet_address": os.getenv("WALLET_ADDRESS", "0x" + "0"*40),
                "endpoint_url": f"http://{ip}:8000",
                "model": "Mistral-7B-OpenOrca-GPTQ",
                "max_tokens": 4096
            }
            logger.info(f"Registering with: {MIA_URL}/miner/register")
            
            r = self.session.post(f"{MIA_URL}/miner/register", json=data, timeout=30)
            logger.info(f"Registration response: {r.status_code}")
            
            if r.status_code != 200:
                logger.error(f"Registration failed with status {r.status_code}: {r.text}")
                return False
                
            response_data = r.json()
            if "miner_id" not in response_data:
                logger.error(f"Invalid response format: {response_data}")
                return False
                
            self.miner_id = response_data["miner_id"]
            logger.info(f"✓ Registered successfully! Miner ID: {self.miner_id}")
            return True
        except Exception as e:
            logger.error(f"Register failed with exception: {e}")
            return False
    
    def run(self):
        if not self.wait_vllm():
            return
        
        while not self.register():
            time.sleep(30)
        
        while True:
            try:
                r = self.session.get(f"{MIA_URL}/miner/{self.miner_id}/work")
                if r.status_code == 200:
                    work = r.json()
                    if work.get("request_id"):
                        logger.info(f"Processing {work['request_id']}")
                        
                        gen = self.session.post(
                            f"{VLLM_URL}/generate",
                            json={"prompt": work["prompt"]}
                        )
                        
                        self.session.post(
                            f"{MIA_URL}/miner/{self.miner_id}/submit",
                            json={
                                "request_id": work["request_id"],
                                "result": {
                                    "success": True,
                                    "response": gen.json()["text"]
                                }
                            }
                        )
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(10)
            
            time.sleep(5)

if __name__ == "__main__":
    MIAMiner().run()
EOF

# Start script with logging
cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

cd /opt/mia-gpu-miner
source venv/bin/activate

echo -e "${YELLOW}Starting MIA GPU Miner...${NC}"
echo "=============================="

# Kill existing processes
echo -e "\n${YELLOW}[1/4] Stopping any existing processes...${NC}"
pkill -f vllm_server.py || true
pkill -f gpu_miner.py || true
sleep 2

# Start vLLM
echo -e "\n${YELLOW}[2/4] Starting vLLM server...${NC}"
echo "This may take 5-10 minutes on first run to download the model (4GB)"
echo "You can monitor progress in another terminal with:"
echo "  tail -f /opt/mia-gpu-miner/vllm.log"
echo ""

python vllm_server.py > vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"

# Monitor startup
echo -e "\n${YELLOW}[3/4] Waiting for model to load...${NC}"
echo -n "Progress: "
for i in {1..60}; do
    if grep -q "Model loaded!" vllm.log 2>/dev/null; then
        echo -e "\n${GREEN}✓ Model loaded successfully!${NC}"
        break
    elif grep -q "Downloading" vllm.log 2>/dev/null; then
        echo -n "📥"
    else
        echo -n "."
    fi
    sleep 5
done

# Check if vLLM is still running
if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo -e "\n${RED}❌ vLLM server crashed! Check vllm.log for errors${NC}"
    tail -20 vllm.log
    exit 1
fi

# Start miner
echo -e "\n${YELLOW}[4/4] Starting GPU miner...${NC}"
echo "Miner logs will appear below:"
echo "==============================="
python gpu_miner.py 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
EOF
chmod +x "$INSTALL_DIR/start.sh"

# Step 9: Download quantized model (smaller)
echo -e "\n${YELLOW}[9/9] Downloading quantized model (4GB instead of 14GB)...${NC}"
cd "$INSTALL_DIR"
source venv/bin/activate

python3 << 'EOF'
from transformers import AutoTokenizer
print("Downloading tokenizer for quantized model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/Mistral-7B-OpenOrca-GPTQ",
        cache_dir="/opt/mia-gpu-miner/models"
    )
    print("✓ Tokenizer ready!")
    print("Note: The full model will download on first run (4GB)")
except Exception as e:
    print(f"Warning: {e}")
EOF

# Cleanup
rm -rf ~/.cache/pip

# Done
echo -e "\n${GREEN}✓ Minimal setup complete!${NC}"
echo -e "\nTo start:"
echo -e "  ${YELLOW}cd /opt/mia-gpu-miner && ./start.sh${NC}"
echo -e "\nThis uses a quantized model (4GB) to save space."
echo -e "The model will download on first run."

# Final space check
echo -e "\n${YELLOW}Final disk usage:${NC}"
df -h /