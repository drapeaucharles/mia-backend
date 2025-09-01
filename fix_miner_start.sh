#!/bin/bash
# Fix the miner start script to wait for vLLM properly

cd /data/qwen-awq-miner

# Update start_miner.sh to wait for vLLM
cat > start_miner.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

# First ensure vLLM is running
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM..."
    ./start_vllm.sh
    echo "Waiting for vLLM to be ready..."
    
    # Wait up to 60 seconds for vLLM to start
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "✓ vLLM is ready!"
            break
        fi
        printf "."
        sleep 1
    done
    echo ""
else
    echo "vLLM already running (PID $(cat vllm.pid))"
fi

# Activate virtual environment
source .venv/bin/activate

# Install requests if needed
pip install requests 2>/dev/null || true

# Set environment
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export MIA_BACKEND_URL=${MIA_BACKEND_URL:-https://mia-backend-production.up.railway.app}
export MINER_ID=${MINER_ID:-1}

echo "Starting polling miner (ID: $MINER_ID)..."
python miner.py 2>&1 | tee -a miner.log
EOF

chmod +x start_miner.sh

echo "✅ Fixed! Now run: ./start_miner.sh"