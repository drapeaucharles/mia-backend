#!/bin/bash
# Quick update to new auto-registering miner

cd /data/qwen-awq-miner

# Stop old miner
pkill -f miner.py

# Download new miner directly
echo "Updating miner to auto-registering version..."
curl -sS https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-final.sh > /tmp/installer.sh

# Extract just the miner.py part
sed -n '/cat > miner.py << '\''SCRIPT'\''/,/^SCRIPT$/p' /tmp/installer.sh | sed '1d;$d' > miner.py
chmod +x miner.py

# Update start script
sed -n '/cat > start_miner.sh << '\''SCRIPT'\''/,/^SCRIPT$/p' /tmp/installer.sh | sed '1d;$d' > start_miner.sh
chmod +x start_miner.sh

echo "✅ Updated! Now run: ./start_miner.sh"