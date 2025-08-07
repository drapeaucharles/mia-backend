# MIA GPU Miner

A GPU-powered miner client for the MIA (Decentralized AI Customer Support Assistant) network. This miner runs Mistral 7B Instruct locally on GPU to process multilingual customer support prompts.

## 🚀 Features

- **Local GPU Inference**: Runs Mistral 7B Instruct model directly on GPU
- **vLLM Integration**: High-performance inference server for efficient token generation
- **Multilingual Support**: Handles prompts in English, French, Indonesian, and more
- **Auto-Registration**: Automatically registers with MIA backend
- **Real-time Status Updates**: Reports GPU status (idle/busy) to backend
- **One-Line Installation**: Complete setup with a single command

## 📋 System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **CUDA**: Version 11.8 or higher
- **OS**: Ubuntu 20.04/22.04 or similar Linux distribution
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space for model files
- **Python**: 3.10 or higher

## 🔧 Quick Start

### One-Line Install

Run this command on your GPU server:

```bash
bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner.sh)
```

Or with environment variables to skip prompts:

```bash
export MIA_API_URL="https://mia-backend-production.up.railway.app"
export MINER_NAME="my-gpu-miner"
bash <(curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner.sh)
```

This will:
1. Check for NVIDIA GPU and drivers
2. Install Python dependencies
3. Install PyTorch with CUDA support
4. Install vLLM inference server
5. Download Mistral 7B Instruct model (~14GB)
6. Configure and start the miner service
7. Register with MIA backend

## 📊 Monitoring

View miner logs:
```bash
sudo journalctl -u mia-gpu-miner -f
```

Check miner status:
```bash
sudo systemctl status mia-gpu-miner
```

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

Check vLLM server:
```bash
curl http://localhost:8000/
```

## 🛠️ Management Commands

Stop the miner:
```bash
sudo systemctl stop mia-gpu-miner
```

Start the miner:
```bash
sudo systemctl start mia-gpu-miner
```

Restart the miner:
```bash
sudo systemctl restart mia-gpu-miner
```

## 📁 File Structure

```
mia-gpu-miner/
├── venv/                # Python virtual environment
├── vllm_server.py      # vLLM inference server
├── gpu_miner.py        # Main miner client
├── start_miner.sh      # Startup script
├── .env                # Environment configuration
├── vllm.log           # vLLM server logs
└── vllm.pid           # vLLM process ID
```

## 🔄 How It Works

1. **Model Server**: vLLM loads Mistral 7B Instruct on GPU at startup
2. **Registration**: Miner registers with backend, reporting GPU specs
3. **Job Polling**: Every 5 seconds, checks backend for new prompts
4. **Inference**: When job received, sends prompt to vLLM server
5. **Response**: Generated text sent back to backend with token count
6. **Status Updates**: Reports idle/busy status in real-time

## 🌍 Multilingual Support

Mistral 7B Instruct supports multiple languages including:
- English
- French
- Spanish
- German
- Italian
- Portuguese
- Indonesian
- And many more...

The model automatically detects and responds in the appropriate language.

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MIA_API_URL` | MIA backend URL | `https://mia-backend-production.up.railway.app` |
| `MINER_NAME` | Unique miner identifier | `gpu-miner-{hostname}` |
| `VLLM_URL` | Local vLLM server URL | `http://localhost:8000` |
| `POLL_INTERVAL` | Seconds between job checks | `5` |

## 🚧 Troubleshooting

### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Install drivers if needed
sudo apt update
sudo apt install nvidia-driver-525
```

### Out of GPU memory
- Ensure no other processes are using GPU
- Check minimum 16GB VRAM available
- Consider using quantized model (future feature)

### vLLM server not starting
```bash
# Check vLLM logs
tail -f ~/mia-gpu-miner/vllm.log

# Test vLLM manually
cd ~/mia-gpu-miner
source venv/bin/activate
python vllm_server.py
```

### Cannot connect to backend
```bash
# Test backend connectivity
curl https://mia-backend-production.up.railway.app/health

# Check firewall
sudo ufw status
```

## 🔐 Security

- Miners use unique auth keys for identification
- All communication with backend via HTTPS
- Model runs locally - no data sent to third parties
- IP addresses tracked for security monitoring

## 📈 Performance

Expected performance with Mistral 7B:
- **RTX 3090 (24GB)**: ~50-100 tokens/second
- **RTX 4090 (24GB)**: ~100-150 tokens/second
- **A100 (40GB)**: ~200-300 tokens/second

Actual performance depends on prompt length and complexity.

## 🤝 Contributing

To improve the miner:
1. Fork the repository
2. Create a feature branch
3. Test on GPU hardware
4. Submit pull request

## 📞 Support

For issues or questions:
- Check logs first: `sudo journalctl -u mia-gpu-miner -f`
- Verify GPU requirements are met
- Ensure backend is accessible
- Report issues to the MIA repository

## 📄 License

Part of the MIA project. See main repository for license details.