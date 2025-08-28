#!/bin/bash

echo "Testing Qwen Miner"
echo "=================="
echo ""

# Test health endpoint
echo "1. Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

echo ""
echo "2. Generation test:"
time curl -s -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 100
  }' | python3 -m json.tool

echo ""
echo "3. Local endpoint test:"
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }' | python3 -m json.tool