#!/bin/bash
# Clean up disk space and install miner properly

echo "🧹 Cleaning up disk space"
echo "========================"

# Check current disk usage
echo "Current disk usage:"
df -h /data

# Stop any running processes
pkill -f miner.py
pkill -f vllm

# Clean up Hugging Face cache
echo -e "\n🗑️ Cleaning Hugging Face cache..."
rm -rf /data/huggingface/hub/models--*/blobs/*
rm -rf /data/huggingface/hub/models--*/snapshots/*
rm -rf ~/.cache/huggingface/*

# Clean up incomplete downloads
echo "🗑️ Cleaning incomplete downloads..."
find /data -name "*.incomplete" -delete
find /data -name "*.tmp" -delete
find /data -name "*.downloading" -delete

# Remove old backups
echo "🗑️ Removing old backups..."
rm -f /data/qwen-awq-miner/miner_backup_*.py

# Clean pip cache
echo "🗑️ Cleaning pip cache..."
pip cache purge 2>/dev/null || true

# Show space after cleanup
echo -e "\nDisk usage after cleanup:"
df -h /data

# Ask before proceeding
echo -e "\n⚠️  The Qwen 32B model needs about 20GB of space"
echo "Do you have enough space now? (check above)"
echo ""
echo "Options:"
echo "1. Use smaller 7B model instead (needs ~4GB)"
echo "2. Continue with 32B model"
echo "3. Exit"
read -p "Choose [1/2/3]: " choice

case $choice in
    1)
        echo -e "\n📦 Installing with smaller 7B model..."
        cd /data/qwen-awq-miner
        
        # Update miner.py to use 7B model
        sed -i 's/Qwen2.5-32B-Instruct-AWQ/Qwen2.5-7B-Instruct-AWQ/g' miner.py
        sed -i 's/max_model_len=12000/max_model_len=4096/g' miner.py
        
        echo "✅ Updated to use 7B model"
        echo "Start with: ./start_miner.sh"
        ;;
    2)
        echo -e "\n⚠️  Continuing with 32B model..."
        echo "Make sure you have at least 20GB free space!"
        cd /data/qwen-awq-miner
        ./start_miner.sh
        ;;
    *)
        echo "Exiting..."
        exit 0
        ;;
esac