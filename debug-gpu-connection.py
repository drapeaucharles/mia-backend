#!/usr/bin/env python3
"""Debug GPU connection issues"""
import requests
import json

print("🔍 Debugging GPU Connection\n")

# 1. Check what backend sees
print("1️⃣ What backend sees about GPU:")
response = requests.get("https://mia-backend-production.up.railway.app/metrics/gpus")
data = response.json()
print(json.dumps(data, indent=2))

# 2. Test direct connection to GPU
print("\n2️⃣ Testing direct connection to GPU at bore.pub:45413...")
try:
    response = requests.get("http://bore.pub:45413/health", timeout=5)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# 3. Test pushing work directly to GPU (bypassing backend)
print("\n3️⃣ Testing direct work push to GPU...")
try:
    response = requests.post(
        "http://bore.pub:45413/process",
        json={
            "request_id": "test-direct",
            "prompt": "Hello, what's on the menu?",
            "messages": [{"role": "user", "content": "Hello, what's on the menu?"}],
            "temperature": 0.7,
            "max_tokens": 200
        },
        timeout=30
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result.get('success')}")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# 4. Check backend logs (if we had access)
print("\n4️⃣ Backend should be pushing to URL stored in heartbeat")
print("Check Railway logs for any errors when pushing to GPU")