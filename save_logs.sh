#!/bin/bash
# Save logs to files for easier viewing

cd /data/qwen-awq-miner

echo "=== Saving Logs ==="

# Save recent miner log
if [ -f "miner.log" ]; then
    tail -n 1000 miner.log > recent_miner.log
    echo "✓ Saved last 1000 lines to recent_miner.log"
fi

# Save diagnostic output
echo "Running diagnostic and saving to file..."
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/diagnose_cuda_issue.sh 2>&1 | tee diagnostic_output.txt

echo -e "\n=== Logs Saved ==="
echo "View with:"
echo "  cat recent_miner.log"
echo "  cat diagnostic_output.txt"
echo ""
echo "Or download them:"
echo "  scp root@your-vps-ip:/data/qwen-awq-miner/diagnostic_output.txt ."