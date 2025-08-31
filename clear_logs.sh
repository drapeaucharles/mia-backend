#!/bin/bash
# Clear old logs in the miner directory

cd /data/qwen-awq-miner

echo "=== Clearing Old Logs ==="

# Show current log sizes
echo "Current logs:"
ls -lh *.log 2>/dev/null || echo "No log files found"

# Clear/remove logs
echo -e "\nClearing logs..."
rm -f miner.log vllm_server.log 2>/dev/null
echo "✓ Removed old log files"

# Create empty log files
touch miner.log vllm_server.log
echo "✓ Created fresh log files"

echo -e "\n=== Logs Cleared ==="
echo "Fresh start - no old errors in the logs!"