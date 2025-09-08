#!/usr/bin/env python3
import requests
import json
import time

print("🚀 Testing GPU with new bore.pub:53103...\n")

# Test direct connection to GPU
print("1️⃣ Testing direct push to GPU...")
try:
    response = requests.post(
        "http://bore.pub:53103/process",
        json={
            "request_id": "test-new-port",
            "prompt": "Hello, what pizzas do you have on the menu?",
            "messages": [{"role": "user", "content": "Hello, what pizzas do you have on the menu?"}],
            "temperature": 0.7,
            "max_tokens": 200
        },
        timeout=30
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        if result.get('success'):
            print(f"✅ AI Response: {result.get('response', 'No response')[:300]}...")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"Tokens generated: {result.get('tokens_generated', 0)}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"Error: {e}")

print("\n2️⃣ Testing via backend /chat endpoint...")
start = time.time()
response = requests.post(
    "https://mia-backend-production.up.railway.app/chat",
    json={
        "message": "What vegetarian options do you have?",
        "client_id": "test-bore-2",
        "restaurant_id": "1"
    }
)
elapsed = time.time() - start

print(f"Status: {response.status_code}")
print(f"Time: {elapsed:.2f}s")
if response.status_code == 200:
    data = response.json()
    ai_response = data.get('response', 'No response')
    if ai_response and ai_response != 'No response':
        print(f"✅ AI Response: {ai_response[:300]}...")
    else:
        print("⚠️ Empty response from backend")