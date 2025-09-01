#!/usr/bin/env python3
"""Test what's being submitted to backend"""
import requests
import json

# Test the submit format
backend_url = "https://mia-backend-production.up.railway.app"

# What the miner sends
test_data = {
    "miner_id": 1,
    "request_id": "test-123",
    "result": {
        "response": "This is a test response that should be stored correctly",
        "tokens_generated": 10,
        "processing_time": 0.5,
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ"
    }
}

print("Testing submit_result format...")
print(f"Sending: {json.dumps(test_data, indent=2)}")

try:
    response = requests.post(
        f"{backend_url}/submit_result",
        json=test_data,
        timeout=10
    )
    
    print(f"\nResponse status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Now check what was stored
    print("\nChecking stored result...")
    check = requests.get(f"{backend_url}/job/test-123/result")
    print(f"Stored result: {json.dumps(check.json(), indent=2) if check.status_code == 200 else check.text}")
    
except Exception as e:
    print(f"Error: {e}")