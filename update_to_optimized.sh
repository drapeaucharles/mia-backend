#!/bin/bash
# Update MIA miner to optimized version for 60+ tokens/sec

echo "MIA Backend Optimization Update for RTX 3090"
echo "==========================================="

# Backup current miner
echo "1. Backing up current miner..."
cp mia_miner_production.py mia_miner_production.backup.py

# Copy optimized version
echo "2. Installing optimized miner..."
cp mia_miner_optimized.py mia_miner_production.py

# Check if using systemd service
if systemctl is-enabled mia-miner >/dev/null 2>&1; then
    echo "3. Restarting MIA miner service..."
    sudo systemctl stop mia-miner
    sleep 2
    sudo systemctl start mia-miner
    sleep 5
    sudo systemctl status mia-miner --no-pager
else
    echo "3. No systemd service found. Please restart your miner manually."
fi

echo ""
echo "4. Testing new configuration..."
curl -X POST http://localhost:8000/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, how are you?",
        "max_tokens": 50,
        "temperature": 0
    }' | python3 -m json.tool

echo ""
echo "✓ Update complete!"
echo ""
echo "Expected improvements:"
echo "- GPU memory utilization: 85% → 95%"
echo "- Max sequences: 8 → 16"
echo "- Context length: 2048 → 4096"
echo "- Expected speed: 28 → 60+ tokens/sec"
echo ""
echo "To monitor performance:"
echo "- Check logs: journalctl -u mia-miner -f"
echo "- Test speed: curl http://localhost:8000/health"