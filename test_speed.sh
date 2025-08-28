#!/bin/bash
# Test MIA backend speed

echo "Testing MIA Backend Speed"
echo "========================"
echo ""

# Test 1: Simple query (greedy decoding for max speed)
echo "Test 1: Simple query with greedy decoding (fastest)"
echo "---------------------------------------------------"
time curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_tokens": 200,
    "temperature": 0
  }' | python3 -m json.tool

echo ""
echo ""

# Test 2: Restaurant query
echo "Test 2: Restaurant query (your use case)"
echo "----------------------------------------"
time curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are Maria, a friendly server at Bella Vista Restaurant.\n\nOur complete menu:\n\nMain Courses:\n• Ribeye Steak ($38.99) [beef, herbs, butter]\n• Grilled Chicken ($24.99) [chicken breast, herbs, lemon]\n• Lamb Chops ($35.99) [lamb, rosemary, garlic]\n\nCustomer: I want to eat steak or things similar\nAssistant:",
    "max_tokens": 150,
    "temperature": 0.7
  }' | python3 -m json.tool

echo ""
echo ""

# Test 3: Check health/performance stats
echo "Test 3: Health check with performance stats"
echo "------------------------------------------"
curl -X GET http://localhost:8000/health | python3 -m json.tool

echo ""
echo ""
echo "Look for 'tokens_per_second' in the response!"