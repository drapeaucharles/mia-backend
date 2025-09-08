#!/usr/bin/env python3
"""Test Pinggy connection methods"""
import subprocess
import time
import re

print("🔍 Testing Pinggy connection methods...")

# Method 1: Basic curl
print("\n1. Testing basic curl method:")
proc = subprocess.Popen(
    "curl -s https://pinggy.io/start | sh -s -- -p 5000 -t",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(5)
stdout, stderr = proc.communicate()
print(f"stdout: {stdout[:200]}")
print(f"stderr: {stderr[:200]}")

# Method 2: Direct pinggy command
print("\n2. Testing direct command (if pinggy installed):")
result = subprocess.run("which pinggy", shell=True, capture_output=True, text=True)
if result.returncode == 0:
    print("pinggy command found")
else:
    print("pinggy command not found")

# Method 3: SSH tunnel (original method)
print("\n3. SSH tunnel method requires SSH key")

print("\n📌 Recommended: Use localhost.run instead")
print("It's simpler and works immediately:")
print("ssh -R 80:localhost:5000 localhost.run")