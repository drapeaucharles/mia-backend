#!/bin/bash
# Demo script showing tool calling flow with MIA backend

echo "MIA Backend Tool Calling Demo"
echo "============================="
echo ""

# Check if backend is running
echo "1. Checking backend health..."
if curl -s http://localhost:8080/ | grep -q "healthy"; then
    echo "✓ Backend is healthy"
else
    echo "✗ Backend not running. Please start it first."
    exit 1
fi

echo ""
echo "2. Submitting a job with tool calling..."
echo ""

# Submit a chat request with tools
RESPONSE=$(curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What fish dishes do you have on the menu?",
    "context": {
      "business_name": "The Seafood Grill",
      "system_prompt": "You are a helpful restaurant assistant. Use the search_menu_items tool to find menu items."
    },
    "tools": [{
      "type": "function",
      "function": {
        "name": "search_menu_items",
        "description": "Search menu items by ingredient, category, or name",
        "parameters": {
          "type": "object",
          "properties": {
            "search_term": {
              "type": "string",
              "description": "What to search for"
            },
            "search_type": {
              "type": "string",
              "enum": ["ingredient", "category", "name"],
              "description": "Type of search"
            }
          },
          "required": ["search_term", "search_type"]
        }
      }
    }],
    "tool_choice": "auto"
  }')

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo "Job created: $JOB_ID"

echo ""
echo "3. Waiting for a miner to process the job..."
echo "(Make sure job poller is running: bash start_job_poller.sh)"
echo ""

# Poll for result
for i in {1..30}; do
    RESULT=$(curl -s http://localhost:8080/job/$JOB_ID/result)
    STATUS=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
    
    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo "✓ Job completed!"
        echo ""
        echo "Result:"
        echo "$RESULT" | python3 -m json.tool
        
        # Check if tool was called
        if echo "$RESULT" | grep -q "tool_call"; then
            echo ""
            echo "✓ Tool call detected!"
            echo "The AI correctly identified it should search for fish dishes."
        fi
        
        exit 0
    fi
    
    printf "."
    sleep 1
done

echo ""
echo "✗ Timeout waiting for result"
echo "Make sure:"
echo "1. vLLM is running: cd /data/qwen-awq-miner && ./vllm_manage.sh status"
echo "2. Job poller is running: bash start_job_poller.sh"