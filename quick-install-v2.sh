#!/bin/bash
# Quick installer that downloads and runs the universal installer v2

# Download the universal installer
wget -O /tmp/universal-miner-installer-v2.sh https://gist.githubusercontent.com/drapeaucharles/YOUR_GIST_ID/raw/universal-miner-installer-v2.sh

# Make it executable and run
chmod +x /tmp/universal-miner-installer-v2.sh
bash /tmp/universal-miner-installer-v2.sh