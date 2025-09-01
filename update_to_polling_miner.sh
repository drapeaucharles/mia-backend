#!/bin/bash
# Update existing installation to use job polling miner

echo "Updating to Job Polling Miner"
echo "============================="
echo ""

# Go to existing installation
cd /data/qwen-awq-miner

# Backup current miner if exists
if [ -f "miner.py" ]; then
    echo "Backing up current miner.py..."
    cp miner.py miner_backup_$(date +%Y%m%d_%H%M%S).py
fi

# Copy new polling miner
echo "Installing new job polling miner..."
cp /home/charles-drapeau/Documents/Project/MIA_project/mia-backend/miner_job_polling.py miner.py

# Make sure it's executable
chmod +x miner.py

# Update start_miner.sh to use the polling miner
echo "Updating start_miner.sh..."
cat > start_miner.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

# First ensure vLLM is running
if [[ ! -f vllm.pid ]] || ! kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "Starting vLLM server first..."
    ./start_vllm.sh
    echo "Waiting for vLLM to start..."
    sleep 10
fi

# Activate virtual environment
source .venv/bin/activate

# Set environment
export HF_HOME=/data/cache/hf
export TRANSFORMERS_CACHE=/data/cache/hf
export MIA_BACKEND_URL=${MIA_BACKEND_URL:-https://mia-backend-production.up.railway.app}

# Get miner ID from environment or use default
if [ -z "$MINER_ID" ]; then
    echo "WARNING: MINER_ID not set, using default (1)"
    echo "Set it with: export MINER_ID=your_actual_id"
    export MINER_ID=1
fi

# Start the polling miner
echo "Starting MIA job polling miner..."
echo "Backend: $MIA_BACKEND_URL"
echo "Miner ID: $MINER_ID"
echo ""

python miner.py 2>&1 | tee -a miner.log
EOF
chmod +x start_miner.sh

# Create stop script if doesn't exist
if [ ! -f "stop_miner.sh" ]; then
    cat > stop_miner.sh << 'EOF'
#!/bin/bash
echo "Stopping miner..."
pkill -f "miner.py"
echo "Miner stopped"

# Note: vLLM server is kept running
echo "vLLM server is still running. To stop it: ./vllm_manage.sh stop"
EOF
    chmod +x stop_miner.sh
fi

# Create simple management script
cat > miner_manage.sh << 'EOF'
#!/bin/bash
case "$1" in
    start)
        ./start_miner.sh
        ;;
    stop)
        ./stop_miner.sh
        ;;
    restart)
        ./stop_miner.sh
        sleep 2
        ./start_miner.sh
        ;;
    status)
        if pgrep -f "miner.py" > /dev/null; then
            echo "✓ Miner is running"
            echo "Recent logs:"
            tail -5 miner.log
        else
            echo "✗ Miner is not running"
        fi
        
        if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
            echo "✓ vLLM is running (PID $(cat vllm.pid))"
        else
            echo "✗ vLLM is not running"
        fi
        ;;
    logs)
        tail -f miner.log
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
EOF
chmod +x miner_manage.sh

echo ""
echo "✅ Update complete!"
echo ""
echo "The new miner:"
echo "- Uses job polling (no API server needed)"
echo "- Polls jobs from MIA backend"
echo "- Sends to local vLLM server"
echo "- Supports tool calling with Hermes parser"
echo ""
echo "To use:"
echo "1. Set your miner ID: export MINER_ID=your_id"
echo "2. Start the miner: cd /data/qwen-awq-miner && ./start_miner.sh"
echo ""
echo "Or use the management script:"
echo "  ./miner_manage.sh start    # Start miner"
echo "  ./miner_manage.sh status   # Check status"
echo "  ./miner_manage.sh logs     # View logs"
echo "  ./miner_manage.sh stop     # Stop miner"