#!/bin/bash
# Safe vLLM OpenAI installer that respects existing miner setup
# Does NOT kill processes or change virtualenvs by default

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== vLLM OpenAI Server Installer ===${NC}"
echo "This installer respects your existing miner setup"
echo ""

# Base directory
BASE_DIR="/data/qwen-awq-miner"
cd "$BASE_DIR"

# Check for existing virtualenv - use .venv first, then venv
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
    echo -e "${GREEN}Using existing virtualenv at .venv${NC}"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
    echo -e "${GREEN}Using existing virtualenv at venv${NC}"
else
    echo -e "${RED}Error: No virtualenv found at .venv or venv${NC}"
    echo "Please ensure your miner is properly installed first"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Create the vLLM start script
cat > start_vllm_openai.sh << 'EOF'
#!/bin/bash
# Start vLLM OpenAI server with proper settings

BASE_DIR="/data/qwen-awq-miner"
cd "$BASE_DIR"

# Use existing virtualenv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtualenv found"
    exit 1
fi

# Check if already running
if [ -f "vllm.pid" ] && kill -0 $(cat vllm.pid) 2>/dev/null; then
    echo "vLLM server is already running with PID: $(cat vllm.pid)"
    echo "Use ./stop_vllm_openai.sh to stop it first"
    exit 1
fi

# Set environment for optimal performance
export VLLM_ATTENTION_BACKEND="XFORMERS"  # Fast default
export CUDA_VISIBLE_DEVICES="0"

echo "Starting vLLM OpenAI server..."
echo "Model: Qwen/Qwen2.5-7B-Instruct-AWQ"
echo "Context: 12,288 tokens"
echo "Tool parser: hermes"
echo "Attention: xFormers"

# Start vLLM with proper settings
nohup python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
    --quantization awq \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12288 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > logs/vllm.out 2>&1 &

# Save PID
echo $! > vllm.pid

echo "vLLM server starting with PID: $(cat vllm.pid)"
echo "Logs: tail -f logs/vllm.out"
echo ""
echo "Waiting for server to be ready..."

# Wait for server to start
for i in {1..30}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ vLLM OpenAI server is ready!"
        echo ""
        echo "Endpoints:"
        echo "  - http://localhost:8000/v1/models"
        echo "  - http://localhost:8000/v1/chat/completions"
        echo "  - http://localhost:8000/v1/completions"
        exit 0
    fi
    sleep 2
    echo -n "."
done

echo ""
echo "⚠️  Server may still be starting. Check logs: tail -f logs/vllm.out"
EOF

# Create stop script
cat > stop_vllm_openai.sh << 'EOF'
#!/bin/bash
# Stop vLLM OpenAI server (only the one we started)

BASE_DIR="/data/qwen-awq-miner"
cd "$BASE_DIR"

if [ -f "vllm.pid" ]; then
    PID=$(cat vllm.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping vLLM server (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "Process still running, sending SIGKILL..."
            kill -9 $PID
        fi
        rm -f vllm.pid
        echo "vLLM server stopped"
    else
        echo "vLLM server not running (stale PID file)"
        rm -f vllm.pid
    fi
else
    echo "No vLLM server PID file found"
fi
EOF

# Create logs viewing script
cat > logs_vllm_openai.sh << 'EOF'
#!/bin/bash
# View vLLM server logs

BASE_DIR="/data/qwen-awq-miner"
LOG_FILE="$BASE_DIR/logs/vllm.out"

if [ -f "$LOG_FILE" ]; then
    echo "=== vLLM Server Logs ==="
    echo "File: $LOG_FILE"
    echo "Press Ctrl+C to exit"
    echo ""
    tail -f "$LOG_FILE"
else
    echo "No log file found at: $LOG_FILE"
    echo "The server may not have been started yet"
fi
EOF

# Create status script
cat > status_vllm_openai.sh << 'EOF'
#!/bin/bash
# Check vLLM server status

BASE_DIR="/data/qwen-awq-miner"
cd "$BASE_DIR"

echo "=== vLLM OpenAI Server Status ==="
echo ""

# Check PID
if [ -f "vllm.pid" ]; then
    PID=$(cat vllm.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "✅ Server is running (PID: $PID)"
        
        # Check process details
        echo ""
        echo "Process info:"
        ps -p $PID -o pid,vsz,rss,pcpu,pmem,cmd | tail -n 1
        
        # Check API endpoint
        echo ""
        echo "API status:"
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "✅ API is responding"
            echo ""
            echo "Models available:"
            curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null | grep -E '"id":|"object":'
        else
            echo "❌ API is not responding"
        fi
    else
        echo "❌ Server is not running (stale PID file)"
    fi
else
    echo "❌ Server is not running (no PID file)"
fi

# Check logs
if [ -f "logs/vllm.out" ]; then
    echo ""
    echo "Recent log entries:"
    tail -n 5 logs/vllm.out
fi
EOF

# Make all scripts executable
chmod +x start_vllm_openai.sh stop_vllm_openai.sh logs_vllm_openai.sh status_vllm_openai.sh

echo -e "${GREEN}✅ Installation complete!${NC}"
echo ""
echo "Scripts created:"
echo "  - ${YELLOW}./start_vllm_openai.sh${NC} - Start the vLLM OpenAI server"
echo "  - ${YELLOW}./stop_vllm_openai.sh${NC} - Stop the vLLM server (only ours)"
echo "  - ${YELLOW}./logs_vllm_openai.sh${NC} - View server logs"
echo "  - ${YELLOW}./status_vllm_openai.sh${NC} - Check server status"
echo ""
echo "To start the server now:"
echo "  ${GREEN}./start_vllm_openai.sh${NC}"
echo ""
echo "Note: This installer did NOT stop any existing processes"