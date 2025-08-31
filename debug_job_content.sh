#!/bin/bash
# Add debugging to miner to see job content

cd /data/qwen-awq-miner

# Create a patch to add job logging
cat > add_job_debug.py << 'EOF'
import re

# Read miner.py
with open('miner.py', 'r') as f:
    content = f.read()

# Find where job is processed
process_pattern = r'if work and work\.get\(.request_id.\):'
if re.search(process_pattern, content):
    # Add logging after job is received
    new_content = re.sub(
        r'(if work and work\.get\(.request_id.\):.*?logger\.info\(f"Processing job: {work\[.request_id.\]}"\))',
        r'\1\n                    \n                    # DEBUG: Log job content\n                    logger.info(f"Job content: prompt={work.get(\'prompt\', \'\')[:50]}...")\n                    logger.info(f"Job has tools: {\'tools\' in work}")\n                    if \'tools\' in work:\n                        logger.info(f"Number of tools: {len(work.get(\'tools\', []))}")\n                        logger.info(f"Tools: {[t.get(\'name\') for t in work.get(\'tools\', [])]}")',
        content,
        flags=re.DOTALL
    )
    
    # Write back
    with open('miner.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Added job debugging")
else:
    print("✗ Could not find job processing code")
EOF

# Apply the patch
python3 add_job_debug.py
rm add_job_debug.py

echo "✓ Debug logging added"
echo ""
echo "Restart your miner to see detailed job content:"
echo "  pkill -f miner.py && ./start_miner.sh"
echo ""
echo "Then check logs to see what jobs contain:"
echo "  tail -f miner.log | grep -E '(Job content|Job has tools|Tools:)'"