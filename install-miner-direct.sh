#!/bin/bash

# Direct installation script for MIA GPU Miner
# This downloads the setup script first, then runs it

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}MIA GPU Miner Installer${NC}"
echo "========================"

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download the setup script
echo -e "${YELLOW}Downloading setup script...${NC}"
wget -q https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/mia-miner/setup-gpu-miner.sh -O setup-gpu-miner.sh || {
    echo -e "${RED}Failed to download setup script${NC}"
    echo "Alternative: Download manually from:"
    echo "https://github.com/drapeaucharles/mia-backend/blob/master/mia-miner/setup-gpu-miner.sh"
    exit 1
}

# Make it executable
chmod +x setup-gpu-miner.sh

# Run the setup
echo -e "${YELLOW}Starting installation...${NC}"
./setup-gpu-miner.sh

# Cleanup
cd /
rm -rf "$TEMP_DIR"