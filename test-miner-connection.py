#!/usr/bin/env python3
"""Test script to verify miner can connect to backend"""

import requests
import json

# Test backend health
backend_url = "https://mia-backend-production.up.railway.app"

try:
    # Check health endpoint
    response = requests.get(f"{backend_url}/health", timeout=5)
    print(f"✓ Backend health check: {response.status_code}")
    if response.status_code == 200:
        print(f"  Response: {response.json()}")
    
    # Check miner endpoints
    response = requests.get(f"{backend_url}/api/miners", timeout=5)
    print(f"\n✓ Miners endpoint: {response.status_code}")
    if response.status_code == 200:
        miners = response.json()
        print(f"  Active miners: {len(miners)}")
        for miner in miners:
            print(f"  - {miner.get('name', 'Unknown')}: {miner.get('status', 'Unknown')}")
    
    print("\n✓ Backend is accessible and ready for miners!")
    
except requests.exceptions.RequestException as e:
    print(f"✗ Error connecting to backend: {e}")
    print("\nPlease check:")
    print("- Internet connection")
    print("- Backend URL is correct")
    print("- Firewall settings")