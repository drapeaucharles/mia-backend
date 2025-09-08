#!/usr/bin/env python3
import requests

# Make a test request that will log the GPU URL being used
print("🔍 Checking what GPU URL the backend is using...")

# This should trigger logging on the backend
response = requests.post(
    "https://mia-backend-production.up.railway.app/chat/direct",
    json={"message": "Test to check GPU URL", "context": {}},
    timeout=10
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

print("\n💡 The backend logs will show which URL it tried to use")
print("Check Railway logs to see the exact URL being attempted")

# Also check current bore.pub port on GPU
print("\n📌 Current bore.pub URL on your GPU: bore.pub:53103")
print("If backend is using old port (45413), the heartbeat isn't updating properly")