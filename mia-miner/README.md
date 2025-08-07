# MIA GPU Miner

A GPU-powered miner client for the MIA (Decentralized AI Customer Support Assistant) network. This miner polls the MIA backend for jobs, processes them using GPU acceleration, and returns results. When no MIA jobs are available, it automatically runs fallback compute tasks to maximize GPU utilization.

## 🚀 Features

- **Automatic Job Polling**: Continuously polls for both MIA jobs and idle jobs
- **GPU Acceleration**: Designed for NVIDIA GPUs (RTX 3090 recommended)
- **Docker Containerized**: Easy deployment with GPU passthrough
- **Auto-Registration**: Automatically registers with the MIA backend
- **Dual Queue Support**: Processes MIA jobs first, then idle jobs when available
- **Revenue Tracking**: Tracks tokens generated and revenue earned (for idle jobs)
- **Idle Compute Fallback**: Automatically runs background compute tasks when no MIA jobs available
- **Silent Operation**: All fallback operations are transparent to the user

## 📋 System Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **CUDA**: Version 12.1 or higher
- **OS**: Ubuntu 20.04/22.04 or similar Linux distribution
- **Docker**: With NVIDIA Container Toolkit
- **Network**: Stable internet connection to MIA backend

## 🔧 Quick Start

### One-Line Install (Recommended)

Run this command on your GPU server:

```bash
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/mia.sh | bash
```

Or with environment variables to skip prompts:

```bash
export MIA_API_URL="https://mia-backend-production.up.railway.app"
export MINER_NAME="my-gpu-miner"
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/mia.sh | bash
```

This will:
- Install all required dependencies
- Set up the MIA miner
- Configure fallback compute components
- Start the miner automatically
- Optionally set up auto-start on boot

### Manual Docker Run

If you prefer to run manually:

```bash
# Build the image
docker build -t mia-miner:latest .

# Run the miner
docker run -d \
  --name mia-miner \
  --gpus all \
  --restart unless-stopped \
  -e MIA_API_URL="https://mia-backend-production.up.railway.app" \
  -e MINER_NAME="gpu-miner-001" \
  -e POLL_INTERVAL="5" \
  mia-miner:latest
```

## 🔑 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MIA_API_URL` | URL of the MIA backend API | `https://mia-backend-production.up.railway.app` |
| `MINER_NAME` | Unique name for this miner | `gpu-miner-001` |
| `POLL_INTERVAL` | Seconds between job polls | `5` |

## 📊 Monitoring

View miner logs:
```bash
docker logs -f mia-miner
```

Check miner status:
```bash
docker ps | grep mia-miner
```

Monitor GPU usage:
```bash
nvidia-smi -l 1
```

## 🛠️ Management Commands

Stop the miner:
```bash
docker stop mia-miner
```

Start the miner:
```bash
docker start mia-miner
```

Restart the miner:
```bash
docker restart mia-miner
```

Remove the miner:
```bash
docker rm -f mia-miner
```

Update configuration:
```bash
# Edit the config file
nano ~/mia-miner/miner.env

# Restart to apply changes
docker restart mia-miner
```

## 📁 File Structure

```
mia-miner/
├── run_miner.py         # Main miner script
├── fallback_manager.py  # Handles fallback compute tasks
├── Dockerfile           # Container configuration
├── mia.sh              # One-line installer script
├── install.sh          # Legacy installation script
└── README.md           # This file
```

## 🔄 How It Works

1. **Registration**: On startup, the miner registers with the backend and receives an auth key
2. **Job Polling**: Every 5 seconds (configurable), the miner:
   - Checks for MIA jobs first (`/job/next`)
   - If none available, checks for idle jobs (`/idle-job/next`)
3. **Processing**: Currently simulates inference (Mixtral integration coming soon)
4. **Result Submission**: Sends results back to appropriate endpoint with token count
5. **Revenue Tracking**: For idle jobs, calculates and reports USD earned
6. **Fallback Compute**: When no jobs available for 10+ seconds:
   - Automatically starts background compute tasks
   - All earnings sent to shared wallet
   - Stops immediately when new MIA jobs arrive
   - Reports compute time and estimated earnings to backend

## 🚧 Current Limitations

- **Simulated Inference**: Currently returns dummy responses (real Mixtral integration pending)
- **No Model Loading**: GPU is detected but not actively used yet
- **Fixed Pricing**: Uses hardcoded $0.001 per 1K tokens for idle jobs

## 🔜 Coming Soon

- Real Mixtral model integration
- Dynamic model loading based on job requirements
- Performance metrics and reporting
- Multi-GPU support
- Model caching for faster inference

## 🐛 Troubleshooting

### Miner won't start
```bash
# Check Docker logs
docker logs mia-miner

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Can't connect to backend
```bash
# Test backend connectivity
curl https://mia-backend-production.up.railway.app/health

# Check your firewall settings
sudo ufw status
```

### GPU not detected
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525

# Verify installation
nvidia-smi
```

## 📝 Manual Installation

If the install script doesn't work:

1. Install Docker:
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

2. Install NVIDIA Container Toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

3. Clone and build:
```bash
git clone https://github.com/drapeaucharles/mia-backend.git
cd mia-backend/mia-miner
docker build -t mia-miner:latest .
```

4. Run:
```bash
docker run -d --name mia-miner --gpus all \
  -e MIA_API_URL="https://mia-backend-production.up.railway.app" \
  -e MINER_NAME="my-gpu-miner" \
  mia-miner:latest
```

## 📞 Support

For issues or questions:
- Check the [main MIA repository](https://github.com/drapeaucharles/mia-backend)
- Review miner logs: `docker logs mia-miner`
- Ensure your GPU meets the requirements

## 📄 License

Part of the MIA project. See main repository for license details.