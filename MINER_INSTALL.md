# MIA GPU Miner Installation Guide

## Quick Install (Recommended)

```bash
# Method 1: Direct download and run
wget https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/setup-gpu-miner.sh
chmod +x setup-gpu-miner.sh
./setup-gpu-miner.sh

# Method 2: Using curl to file
curl -o setup-miner.sh https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/setup-gpu-miner.sh
chmod +x setup-miner.sh
./setup-miner.sh
```

## Manual Installation

If the automatic installer fails, follow these steps:

### 1. Check Prerequisites

```bash
# Check GPU
nvidia-smi

# Check Python
python3 --version

# Install basic dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl wget
```

### 2. Create Installation Directory

```bash
mkdir -p ~/mia-gpu-miner
cd ~/mia-gpu-miner
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

### 4. Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install vLLM and Dependencies

```bash
pip install vllm transformers accelerate sentencepiece protobuf
pip install requests psutil gpustat py-cpuinfo uvicorn fastapi
```

### 6. Download Scripts

```bash
# Download vLLM server
wget https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/setup-gpu-miner.sh -O setup.sh

# Extract the Python scripts from it (or download from repo)
# The setup script contains embedded Python files that need to be extracted
```

### 7. Download Mistral 7B Model

```bash
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading Mistral 7B Instruct (14GB)...")
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', torch_dtype='auto')
print("Download complete!")
EOF
```

### 8. Configure Environment

```bash
# Create .env file
cat > .env << EOF
MIA_API_URL=https://mia-backend-production.up.railway.app
MINER_NAME=gpu-miner-$(hostname)
VLLM_URL=http://localhost:8000
POLL_INTERVAL=5
EOF
```

### 9. Start the Miner

```bash
# Start vLLM server in background
python vllm_server.py &

# Wait for it to load
sleep 30

# Start miner
python gpu_miner.py
```

## Troubleshooting

### Curl Error "Failed writing body"

This happens when the pipe is broken. Solutions:
1. Download to a file first: `curl -o setup.sh URL`
2. Use wget instead: `wget URL`
3. Clone the entire repository

### Python Version Issues

If python3.10 is not available:
```bash
# Use system Python 3
sudo apt install python3-venv python3-pip

# Or install Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv
```

### GPU Memory Issues

Mistral 7B requires ~16GB VRAM. If you have less:
- Wait for quantized model support (coming soon)
- Use a smaller model
- Reduce max sequence length

### vLLM Installation Fails

Try specific version:
```bash
pip install vllm==0.2.7
```

Or build from source:
```bash
pip install git+https://github.com/vllm-project/vllm.git
```

## Getting Help

1. Run diagnostics: `bash test-python.sh`
2. Check GPU: `nvidia-smi`
3. Check logs: `sudo journalctl -u mia-gpu-miner -f`
4. Report issues: https://github.com/drapeaucharles/mia-backend/issues