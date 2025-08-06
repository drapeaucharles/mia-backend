#!/bin/bash

# MIA Miner Installation Script
# This script installs Docker (if needed) and sets up the MIA miner

set -e

echo "==================================="
echo "     MIA GPU Miner Installer       "
echo "==================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   echo "Please run as a regular user with sudo privileges"
   exit 1
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ Warning: nvidia-smi not found. GPU may not be properly configured.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Docker is installed
echo ""
echo "Checking for Docker..."
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✓ Docker installed successfully${NC}"
    echo -e "${YELLOW}Note: You may need to log out and back in for group changes to take effect${NC}"
else
    echo -e "${GREEN}✓ Docker is already installed${NC}"
fi

# Check for NVIDIA Container Toolkit
echo ""
echo "Checking for NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Install nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Restart Docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
else
    echo -e "${GREEN}✓ NVIDIA Container Toolkit is ready${NC}"
fi

# Get configuration from user
echo ""
echo "==================================="
echo "        Miner Configuration        "
echo "==================================="
echo ""

# Get MIA API URL
read -p "Enter MIA Backend URL [https://mia-backend-production.up.railway.app]: " MIA_API_URL
MIA_API_URL=${MIA_API_URL:-https://mia-backend-production.up.railway.app}

# Get Miner Name
DEFAULT_MINER_NAME="gpu-miner-$(hostname)-$(date +%s)"
read -p "Enter Miner Name [$DEFAULT_MINER_NAME]: " MINER_NAME
MINER_NAME=${MINER_NAME:-$DEFAULT_MINER_NAME}

# Get Poll Interval
read -p "Enter Poll Interval in seconds [5]: " POLL_INTERVAL
POLL_INTERVAL=${POLL_INTERVAL:-5}

# Create local directory for miner
MINER_DIR="$HOME/mia-miner"
mkdir -p "$MINER_DIR"

# Save configuration
CONFIG_FILE="$MINER_DIR/miner.env"
cat > "$CONFIG_FILE" << EOF
MIA_API_URL=$MIA_API_URL
MINER_NAME=$MINER_NAME
POLL_INTERVAL=$POLL_INTERVAL
EOF

echo ""
echo "Configuration saved to: $CONFIG_FILE"

# Build or pull Docker image
echo ""
echo "Setting up MIA miner Docker image..."

# Check if we have the Dockerfile locally
if [ -f "Dockerfile" ] && [ -f "run_miner.py" ]; then
    echo "Building Docker image from local files..."
    docker build -t mia-miner:latest .
else
    echo "Pulling pre-built Docker image..."
    # In production, this would pull from a registry
    echo -e "${YELLOW}Note: In production, image would be pulled from registry${NC}"
    echo "For now, please ensure Dockerfile and run_miner.py are in current directory"
    exit 1
fi

# Create systemd service for auto-start (optional)
echo ""
read -p "Create systemd service for auto-start? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/tmp/mia-miner.service"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=MIA GPU Miner
After=docker.service
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10
User=$USER
ExecStart=/usr/bin/docker run --rm --name mia-miner --gpus all --env-file $CONFIG_FILE mia-miner:latest
ExecStop=/usr/bin/docker stop mia-miner

[Install]
WantedBy=multi-user.target
EOF

    sudo mv "$SERVICE_FILE" /etc/systemd/system/mia-miner.service
    sudo systemctl daemon-reload
    sudo systemctl enable mia-miner.service
    
    echo -e "${GREEN}✓ Systemd service created${NC}"
fi

# Start the miner
echo ""
echo "==================================="
echo "          Starting Miner           "
echo "==================================="
echo ""

# Stop any existing miner
docker stop mia-miner 2>/dev/null || true

# Start new miner
echo "Starting MIA miner..."
docker run -d \
    --name mia-miner \
    --gpus all \
    --restart unless-stopped \
    --env-file "$CONFIG_FILE" \
    mia-miner:latest

# Check if started successfully
sleep 3
if docker ps | grep -q mia-miner; then
    echo -e "${GREEN}✓ MIA miner started successfully!${NC}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:    docker logs -f mia-miner"
    echo "  Stop miner:   docker stop mia-miner"
    echo "  Start miner:  docker start mia-miner"
    echo "  Remove miner: docker rm -f mia-miner"
    echo ""
    echo "Configuration file: $CONFIG_FILE"
else
    echo -e "${RED}✗ Failed to start miner${NC}"
    echo "Check logs with: docker logs mia-miner"
    exit 1
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"