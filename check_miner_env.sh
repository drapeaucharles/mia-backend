#!/bin/bash
# Check the actual miner environment setup

cd /data/qwen-awq-miner

echo "=== Checking Miner Environment ==="

echo -e "\n1. Directory contents:"
ls -la | grep -E "(venv|\.venv|start|run|miner)" | head -20

echo -e "\n2. Python environments found:"
find . -maxdepth 2 -name "python*" -type f -executable 2>/dev/null | grep -E "(bin/python|Scripts/python)" | head -10

echo -e "\n3. Checking startup scripts:"
for script in start*.sh run*.sh; do
    if [ -f "$script" ]; then
        echo -e "\n--- $script ---"
        head -20 "$script" | grep -E "(python|venv|activate|miner\.py)"
    fi
done

echo -e "\n4. Checking which Python has vLLM:"
# Try different Python paths
for py in python3 python /usr/bin/python3 ./venv/bin/python ./.venv/bin/python; do
    if command -v $py &> /dev/null; then
        echo -n "Testing $py: "
        $py -c "import vllm; print('HAS vLLM')" 2>/dev/null || echo "NO vLLM"
    fi
done

echo -e "\n5. Checking original miner.py header:"
head -50 miner.py | grep -E "(import|vllm|model)" | head -20

echo -e "\n6. Current process:"
ps aux | grep -E "(miner|vllm)" | grep -v grep

echo -e "\n=== End Environment Check ==="