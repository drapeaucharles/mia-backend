#\!/bin/bash
# Fix the model path in the running bore miner

echo "🔧 Fixing model path in bore miner..."

cd /data/qwen-awq-miner

# Stop current miner
if [ -f heartbeat_bore.pid ]; then
    kill $(cat heartbeat_bore.pid) 2>/dev/null || true
    rm -f heartbeat_bore.pid
fi

# Fix the model path
sed -i 's|"/data/models/Qwen2.5-14B-Instruct-AWQ"|"Qwen/Qwen2.5-7B-Instruct-AWQ"|g' mia_miner_heartbeat_bore.py

# Restart with venv
source .venv/bin/activate
./start_heartbeat_bore.sh

echo "✅ Fixed and restarted\!"
echo "Check logs: tail -f heartbeat_bore.out"
