#!/bin/bash

# Test script to check Python availability

echo "Checking Python installations..."
echo "================================"

# Check default python3
if command -v python3 &> /dev/null; then
    echo "python3: $(python3 --version)"
else
    echo "python3: NOT FOUND"
fi

# Check specific versions
for ver in 3.8 3.9 3.10 3.11; do
    if command -v python$ver &> /dev/null; then
        echo "python$ver: $(python$ver --version)"
    else
        echo "python$ver: NOT FOUND"
    fi
done

echo ""
echo "Checking available packages..."
echo "=============================="

# Test what packages are available
echo "Testing apt-cache for python packages:"
apt-cache search python3 | grep -E "^python3\.[0-9]+-venv" | head -10

echo ""
echo "Current OS Info:"
echo "================"
lsb_release -a 2>/dev/null || cat /etc/os-release | head -5

echo ""
echo "Recommendation:"
echo "==============="
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
    echo "Use: python3 (version $PY_VER)"
    echo "Install: sudo apt-get install python3-venv python3-pip"
else
    echo "No Python 3 found. Install with:"
    echo "sudo apt-get update && sudo apt-get install python3 python3-venv python3-pip"
fi