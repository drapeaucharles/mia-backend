#!/bin/bash
# Complete installer: Base vLLM + Heartbeat architecture
# This runs the base installer first, then adds heartbeat

echo "🚀 Installing Complete vLLM with Heartbeat/Push Architecture"
echo "========================================================="
echo ""

# First run the universal installer
echo "📦 Step 1: Installing base vLLM (universal installer)..."
echo "----------------------------------------------------"
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-final.sh | bash

# Check if base installation succeeded
if [ ! -d "/data/qwen-awq-miner" ]; then
    echo "❌ Base installation failed!"
    exit 1
fi

echo ""
echo "✅ Base installation complete!"
echo ""
sleep 2

# Now run the tool fix update
echo "📦 Step 2: Applying tool call fixes..."
echo "------------------------------------"
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-tool-fix.sh | bash

echo ""
echo "✅ Tool fixes applied!"
echo ""
sleep 2

# Finally run the heartbeat installer
echo "📦 Step 3: Adding heartbeat/push architecture..."
echo "-----------------------------------------------"
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-heartbeat.sh | bash

echo ""
echo "✅ Complete installation finished!"
echo ""
echo "📋 What's installed:"
echo "   - Base vLLM with working wheel"
echo "   - Original polling miner (mia_miner.py)"
echo "   - New heartbeat miner (mia_miner_heartbeat.py)"
echo ""
echo "🚀 To start the heartbeat miner:"
echo "   cd /data/qwen-awq-miner && ./start_heartbeat_miner.sh"
echo ""
echo "📊 To start the original polling miner:"
echo "   cd /data/qwen-awq-miner && ./restart_miner.sh"
echo ""