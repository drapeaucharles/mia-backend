#!/bin/bash
# Run miner directly without systemd

echo "Running MIA GPU Miner (Direct Mode)..."

# Set environment variables
export MIA_API_URL="${MIA_API_URL:-https://mia-backend-production.up.railway.app}"
export MINER_NAME="${MINER_NAME:-gpu-miner-$(hostname)}"

echo "Configuration:"
echo "- API URL: $MIA_API_URL"
echo "- Miner Name: $MINER_NAME"
echo ""

# Download the direct runner if not exists
if [ ! -f "$HOME/run-miner-direct.sh" ]; then
    echo "Downloading direct runner..."
    curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/run-miner-direct.sh -o "$HOME/run-miner-direct.sh"
    chmod +x "$HOME/run-miner-direct.sh"
fi

# Run the miner
cd $HOME
./run-miner-direct.sh