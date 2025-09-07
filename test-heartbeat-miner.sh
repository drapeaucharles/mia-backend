#!/bin/bash
# Test script to verify heartbeat miner is working

echo "🧪 Testing Heartbeat Miner"
echo "========================="
echo ""

# Check if miner is running
if [ -f "/data/qwen-awq-miner/heartbeat_miner.pid" ]; then
    PID=$(cat /data/qwen-awq-miner/heartbeat_miner.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Heartbeat miner is running (PID: $PID)"
    else
        echo "❌ PID file exists but process is not running"
        exit 1
    fi
else
    echo "❌ No heartbeat miner PID file found"
    exit 1
fi

echo ""
echo "📡 Testing local endpoints:"
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
HEALTH=$(curl -s http://localhost:5000/health 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "   ✅ Health endpoint responding"
    echo "   Response: $HEALTH"
else
    echo "   ❌ Health endpoint not responding"
fi

echo ""
echo "2. Testing vLLM backend..."
VLLM=$(curl -s http://localhost:8000/v1/models 2>/dev/null | head -c 100)
if [ $? -eq 0 ]; then
    echo "   ✅ vLLM is running"
    echo "   Models: $VLLM..."
else
    echo "   ❌ vLLM not responding"
fi

echo ""
echo "3. Checking logs..."
if [ -f "/data/qwen-awq-miner/heartbeat_miner.out" ]; then
    echo "   Last 10 lines of heartbeat miner log:"
    echo "   ------------------------------------"
    tail -10 /data/qwen-awq-miner/heartbeat_miner.out | sed 's/^/   /'
fi

echo ""
echo "📊 Summary:"
echo "   - Miner PID: $PID"
echo "   - Flask endpoint: http://localhost:5000"
echo "   - vLLM endpoint: http://localhost:8000"
echo "   - Log file: /data/qwen-awq-miner/heartbeat_miner.out"
echo ""
echo "💡 To view live logs: tail -f /data/qwen-awq-miner/heartbeat_miner.out"
echo "💡 To stop miner: kill $PID"