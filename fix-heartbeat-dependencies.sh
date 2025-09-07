#!/bin/bash
# Fix missing dependencies for heartbeat miner

echo "🔧 Fixing Heartbeat Miner Dependencies"
echo "====================================="
echo ""

cd /data/qwen-awq-miner

# Activate the virtual environment (it's .venv not venv)
if [ -f ".venv/bin/activate" ]; then
    echo "✅ Found virtual environment at .venv"
    source .venv/bin/activate
else
    echo "❌ No virtual environment found at .venv!"
    # Try to find any Python environment
    if [ -f "venv/bin/activate" ]; then
        echo "✅ Found virtual environment at venv"
        source venv/bin/activate
    else
        echo "📦 No virtual environment found, using system Python"
        # Install with system pip
        pip install flask waitress aiohttp || pip3 install flask waitress aiohttp
    fi
fi

# Install missing dependencies
echo "📦 Installing Flask and aiohttp..."
pip install flask waitress aiohttp

# Verify installations
echo ""
echo "🔍 Verifying installations:"
python -c "import flask; print(f'✅ Flask {flask.__version__}')" 2>/dev/null || echo "❌ Flask failed"
python -c "import waitress; print('✅ Waitress installed')" 2>/dev/null || echo "❌ Waitress failed"
python -c "import aiohttp; print(f'✅ aiohttp {aiohttp.__version__}')" 2>/dev/null || echo "❌ aiohttp failed"

echo ""
echo "🚀 Restarting heartbeat miner..."

# Stop old miner if running
if [ -f "heartbeat_miner.pid" ]; then
    kill $(cat heartbeat_miner.pid) 2>/dev/null || true
    rm -f heartbeat_miner.pid
fi

# Start miner again
nohup python3 mia_miner_heartbeat.py > heartbeat_miner.out 2>&1 &
echo $! > heartbeat_miner.pid

echo "✅ Heartbeat miner restarted with PID $(cat heartbeat_miner.pid)"
echo ""
echo "⏳ Waiting for miner to start..."
sleep 5

# Check if it's running
if ps -p $(cat heartbeat_miner.pid) > /dev/null; then
    echo "✅ Miner is running!"
    echo ""
    echo "📊 Last few log lines:"
    tail -5 heartbeat_miner.out
else
    echo "❌ Miner failed to start. Check logs:"
    tail -20 heartbeat_miner.out
fi