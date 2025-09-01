#!/bin/bash
# Fix test script to use venv

cd /data/qwen-awq-miner

# Update test_vllm.sh to activate venv first
cat > test_vllm.sh << 'SCRIPT'
#!/usr/bin/env bash
set -Eeuo pipefail

# Activate venv first
source .venv/bin/activate

BASE="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
AUTH="Authorization: Bearer ${OPENAI_API_KEY:-sk-LOCAL}"

echo "🧪 Testing vLLM API"
echo "=================="

# Test 1: Check models endpoint
echo -e "\n📋 GET /models:"
curl -sS "$BASE/models" -H "$AUTH" | python -m json.tool | head -20

# Test 2: Simple completion (no tools)
echo -e "\n💬 Chat completion (no tools):"
curl -sS "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "$AUTH" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Say hello in 5 words"}],
        "max_tokens": 50
    }' | python -m json.tool | grep -E "(content|role)" || echo "Failed"

# Test 3: Tool calling with auto mode
echo -e "\n🔧 Tool calling (auto mode):"
curl -sS "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "$AUTH" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "What fish dishes do you have?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_menu_items",
                "description": "Search menu items by ingredient, category, or name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {"type": "string"},
                        "search_type": {"type": "string", "enum": ["ingredient", "category", "name"]}
                    },
                    "required": ["search_term", "search_type"]
                }
            }
        }],
        "tool_choice": "auto",
        "temperature": 0
    }' | python -m json.tool | grep -A10 "tool_calls" || echo "No tool calls"

echo -e "\n✅ Tests complete"
SCRIPT
chmod +x test_vllm.sh

echo "✅ Fixed test script to use venv"