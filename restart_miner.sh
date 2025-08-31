#!/bin/bash
# Simple restart script for existing qwen-awq-miner

cd /data/qwen-awq-miner

# Stop current miner
echo "Stopping current miner..."
pkill -f miner.py || true
sleep 2

# Start miner again
echo "Starting miner..."
if [ -f "start_miner.sh" ]; then
    ./start_miner.sh
elif [ -f "run_miner.sh" ]; then
    nohup ./run_miner.sh > miner.log 2>&1 &
    echo "Miner started. Check logs: tail -f miner.log"
else
    echo "Error: No start script found"
fi

echo "Done!"