#!/bin/bash
# Stop the job poller (but keep vLLM running)

cd /data/qwen-awq-miner

if [[ -f job_poller.pid ]]; then
    PID=$(cat job_poller.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping job poller (PID $PID)..."
        kill -TERM "$PID"
        rm -f job_poller.pid
        echo "✅ Job poller stopped"
    else
        echo "Job poller not running (stale PID)"
        rm -f job_poller.pid
    fi
else
    echo "No job poller PID file found"
fi

echo ""
echo "Note: vLLM server is still running"
echo "To stop vLLM: cd /data/qwen-awq-miner && ./vllm_manage.sh stop"