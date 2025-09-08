#!/usr/bin/env python3
"""
Test the push architecture with bore.pub GPU
"""
import requests
import time
import json

def test_push_architecture():
    print("🚀 Testing Push Architecture with Bore.pub GPU\n")
    
    # Check GPU status first
    print("1️⃣ Checking GPU status...")
    response = requests.get("https://mia-backend-production.up.railway.app/metrics/gpus")
    data = response.json()
    
    if data['total_gpus'] == 0:
        print("❌ No GPUs registered!")
        return
        
    gpu = data['gpus'][0]
    print(f"✅ GPU {gpu['id']} is {gpu['status']}")
    print(f"   Last heartbeat: {gpu['seconds_since_heartbeat']}s ago\n")
    
    # Wait for Railway deployment if needed
    print("2️⃣ Waiting for backend to update (if just deployed)...")
    time.sleep(10)
    
    # Test the direct endpoint
    print("\n3️⃣ Testing direct GPU push...")
    
    test_messages = [
        "Hello! What's on your menu today?",
        "Do you have any vegetarian pizzas?",
        "What's your most popular dish?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: {message}")
        start = time.time()
        
        try:
            response = requests.post(
                "https://mia-backend-production.up.railway.app/chat",
                json={
                    "message": message,
                    "client_id": f"push-test-{i}",
                    "restaurant_id": "1",
                    "context": {}
                },
                timeout=30
            )
            
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get('response', 'No response')
                
                if ai_response and ai_response != 'No response':
                    print(f"✅ Success in {elapsed:.2f}s")
                    print(f"   AI: {ai_response[:150]}...")
                else:
                    print(f"⚠️ Empty response in {elapsed:.2f}s")
            else:
                print(f"❌ Error {response.status_code}: {response.text[:100]}")
                
        except requests.Timeout:
            print("❌ Request timed out")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("📊 Summary:")
    print(f"- Backend URL: https://mia-backend-production.up.railway.app")
    print(f"- GPU URL: Check your bore.pub URL in the logs")
    print(f"- Architecture: Push-based (heartbeat + direct communication)")
    print("="*50)

if __name__ == "__main__":
    test_push_architecture()