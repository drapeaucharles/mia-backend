#!/usr/bin/env python3
"""
Test tool calling with the MIA backend
"""
import requests
import json
import time
import sys

# Backend URL
BACKEND_URL = "http://localhost:8080"

def test_tool_calling():
    """Test the tool calling flow"""
    
    # 1. Submit a job with tools
    print("1. Submitting job with tools...")
    
    tools = [{
        "type": "function",
        "function": {
            "name": "search_menu_items",
            "description": "Search menu items by ingredient, category, or name",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "What to search for"},
                    "search_type": {"type": "string", "enum": ["ingredient", "category", "name"]}
                },
                "required": ["search_term", "search_type"]
            }
        }
    }]
    
    chat_request = {
        "message": "What fish dishes do you have?",
        "tools": tools,
        "tool_choice": "auto",
        "context": {
            "business_name": "The Seafood Restaurant"
        }
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=chat_request)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
        
        job_data = response.json()
        job_id = job_data['job_id']
        print(f"Job created: {job_id}")
        
        # 2. Simulate miner getting work
        print("\n2. Miner polling for work...")
        # Note: In real scenario, miner would have a registered ID
        # For testing, we'll check the queue status
        
        queue_check = requests.get(f"{BACKEND_URL}/status")
        if queue_check.status_code == 200:
            status = queue_check.json()
            print(f"Queue length: {status.get('queue_length', 'unknown')}")
        
        # 3. Poll for result
        print("\n3. Polling for result...")
        max_attempts = 30
        for i in range(max_attempts):
            result_response = requests.get(f"{BACKEND_URL}/result/{job_id}")
            
            if result_response.status_code == 200:
                result = result_response.json()
                if result.get('status') == 'completed':
                    print("\nResult received:")
                    print(f"Response: {result.get('response')}")
                    
                    # Check for tool calls
                    if 'tool_call' in result:
                        print(f"\nTool call detected:")
                        print(f"  Name: {result['tool_call']['name']}")
                        print(f"  Parameters: {json.dumps(result['tool_call']['parameters'], indent=2)}")
                    
                    return
                elif result.get('status') == 'failed':
                    print(f"Job failed: {result.get('error')}")
                    return
            
            time.sleep(1)
            sys.stdout.write('.')
            sys.stdout.flush()
        
        print("\nTimeout waiting for result")
        
    except Exception as e:
        print(f"Error: {e}")

def test_simple_chat():
    """Test simple chat without tools"""
    print("\nTesting simple chat (no tools)...")
    
    chat_request = {
        "message": "Hello, how are you?",
        "context": {
            "business_name": "Test Restaurant"
        }
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/chat", json=chat_request)
        if response.status_code == 200:
            job_data = response.json()
            print(f"Job created: {job_data['job_id']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("MIA Backend Tool Calling Test")
    print("=" * 40)
    
    # Test connection
    try:
        health = requests.get(f"{BACKEND_URL}/")
        if health.status_code == 200:
            print("✓ Backend is healthy")
        else:
            print("✗ Backend not responding")
            sys.exit(1)
    except:
        print("✗ Cannot connect to backend at", BACKEND_URL)
        sys.exit(1)
    
    # Run tests
    test_simple_chat()
    print("\n" + "=" * 40)
    test_tool_calling()