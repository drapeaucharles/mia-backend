# Running MIA GPU Miner on Your Server

## Prerequisites
- NVIDIA GPU with 16GB+ VRAM
- Ubuntu 20.04/22.04
- CUDA 11.8+
- Python 3.10+

## Quick Start (One-Line Install)

```bash
# SSH into your GPU server
ssh your-server

# Run the installer
bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner.sh)
```

## Manual Installation Steps

### 1. Clone the repository
```bash
git clone https://github.com/drapeaucharles/mia-backend.git
cd mia-backend
```

### 2. Run the setup script
```bash
cd mia-miner
chmod +x setup-gpu-miner.sh
./setup-gpu-miner.sh
```

### 3. Configure environment
```bash
# Edit the .env file
nano ~/mia-gpu-miner/.env

# Set these values:
MIA_API_URL=https://mia-backend-production.up.railway.app
MINER_NAME=your-unique-miner-name
```

### 4. Start the miner service
```bash
# For systemd systems:
sudo systemctl start mia-gpu-miner
sudo systemctl enable mia-gpu-miner

# For non-systemd systems:
cd ~/mia-gpu-miner
./start_miner.sh
```

## Direct Run (Without systemd)

If your server doesn't have systemd:

```bash
# Download and run the direct runner
cd ~
wget https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/run-miner-direct.sh
chmod +x run-miner-direct.sh

# Run with your settings
export MIA_API_URL="https://mia-backend-production.up.railway.app"
export MINER_NAME="my-gpu-miner"
./run-miner-direct.sh
```

## Monitoring Your Miner

### Check miner status
```bash
# For systemd:
sudo systemctl status mia-gpu-miner

# View logs:
sudo journalctl -u mia-gpu-miner -f

# For direct run:
# Logs will appear in terminal
```

### Monitor GPU usage
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Or use nvtop if installed
nvtop
```

### Check vLLM server
```bash
# Test if vLLM is running
curl http://localhost:8000/v1/models

# Check vLLM logs
tail -f ~/mia-gpu-miner/vllm.log
```

### Verify miner registration
```bash
# Check if your miner appears in the backend
curl https://mia-backend-production.up.railway.app/api/miners | python3 -m json.tool
```

## Troubleshooting

### If miner fails to start:
```bash
# Check for GPU
nvidia-smi

# Check Python version
python3 --version

# Check CUDA
nvcc --version

# Manually test vLLM
cd ~/mia-gpu-miner
source venv/bin/activate
python vllm_server.py
```

### If model download fails:
```bash
# The installer now uses Mistral v0.2 which is not gated
# If you still have issues, check internet connection:
ping huggingface.co

# Or manually download:
cd ~/mia-gpu-miner
source venv/bin/activate
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
```

## Managing Your Miner

### Stop the miner
```bash
sudo systemctl stop mia-gpu-miner
```

### Restart the miner
```bash
sudo systemctl restart mia-gpu-miner
```

### Update the miner
```bash
cd ~/mia-backend
git pull
cd mia-miner
./setup-gpu-miner.sh
```

## Expected Output

When running correctly, you should see:
```
Starting MIA GPU Miner...
✓ vLLM server started on http://localhost:8000
✓ Loading Mistral-7B-Instruct-v0.2...
✓ Model loaded successfully
✓ Registered with MIA backend as: your-miner-name
✓ Status: idle
Polling for jobs every 5 seconds...
```

## Performance Tips

1. **Dedicated GPU**: Ensure no other processes use the GPU
2. **Network**: Stable internet connection for job polling
3. **Memory**: Keep 32GB+ RAM available
4. **Storage**: SSD recommended for model loading

Ready to start? Run the one-line installer on your GPU server!