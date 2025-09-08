#!/bin/bash
echo "🔍 Checking vLLM status on GPU..."
echo ""
echo "1. Check if vLLM process is running:"
echo "   ps aux | grep vllm"
echo ""
echo "2. Check if vLLM is listening on port 8000:"
echo "   curl -s http://localhost:8000/v1/models"
echo ""
echo "3. Test vLLM directly:"
echo '   curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{"model": "/data/models/Qwen2.5-14B-Instruct-AWQ", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'"'"''
echo ""
echo "Run these commands on your GPU VPS to debug"