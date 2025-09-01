#!/bin/bash
# Check status of both vLLM and job poller

cd /data/qwen-awq-miner

echo "🔍 System Status"
echo "==============="

# Check vLLM
if [[ -f vllm.pid ]] && kill -0 "$(cat vllm.pid)" 2>/dev/null; then
    echo "✅ vLLM server: Running (PID $(cat vllm.pid))"
    ps -p "$(cat vllm.pid)" -o pid,vsz,rss,comm | tail -1
else
    echo "❌ vLLM server: Not running"
fi

# Check job poller
if [[ -f job_poller.pid ]] && kill -0 "$(cat job_poller.pid)" 2>/dev/null; then
    echo "✅ Job poller: Running (PID $(cat job_poller.pid))"
    ps -p "$(cat job_poller.pid)" -o pid,vsz,rss,comm | tail -1
else
    echo "❌ Job poller: Not running"
fi

echo ""
echo "📋 Commands:"
echo "- Start job poller: bash /home/charles-drapeau/Documents/Project/MIA_project/mia-backend/start_job_poller.sh"
echo "- Stop job poller: bash /home/charles-drapeau/Documents/Project/MIA_project/mia-backend/stop_job_poller.sh"
echo "- View poller logs: tail -f /data/qwen-awq-miner/logs/job_poller.log"
echo "- View vLLM logs: tail -f /data/qwen-awq-miner/logs/vllm.out"