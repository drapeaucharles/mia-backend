#!/usr/bin/env python3
"""
Test GPU connectivity for heartbeat architecture
"""
import requests
import json
import sys

def test_gpu_connectivity():
    # First check GPU status
    print("🔍 Checking GPU status...")
    
    try:
        response = requests.get("https://mia-backend-production.up.railway.app/metrics/gpus")
        data = response.json()
        
        print(f"\n📊 GPU Metrics:")
        print(f"Total GPUs: {data['total_gpus']}")
        print(f"Available: {data['available_gpus']}")
        
        if data['total_gpus'] == 0:
            print("\n❌ No GPUs registered!")
            return
            
        gpu = data['gpus'][0]
        print(f"\nGPU Details:")
        print(f"ID: {gpu['id']}")
        print(f"Name: {gpu['name']}")
        print(f"Status: {gpu['status']}")
        print(f"Last heartbeat: {gpu['seconds_since_heartbeat']}s ago")
        
        # Extract IP from name (format: hostname_backend)
        gpu_id = gpu['id']
        
        # Test direct push
        print(f"\n🚀 Testing direct push to GPU...")
        
        test_message = {"message": "Hello from test", "context": {}}
        
        try:
            response = requests.post(
                "https://mia-backend-production.up.railway.app/chat/direct",
                json=test_message,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Response: {result.get('response', 'No response')[:100]}...")
            else:
                print(f"❌ Failed with status: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.Timeout:
            print("❌ Request timed out - GPU might not be reachable from backend")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")

# Also test from the GPU side
print("""
💡 To test from GPU side, run this on your VPS:

curl http://localhost:5000/health

If that works, the issue is likely that Railway can't reach your GPU's port 5000.
You may need to:
1. Use a reverse proxy (ngrok, cloudflared)
2. Or use the polling miner which doesn't require incoming connections
""")

if __name__ == "__main__":
    test_gpu_connectivity()