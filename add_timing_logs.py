#!/usr/bin/env python3
import re
import sys

try:
    with open('mia_miner_unified.py', 'r') as f:
        content = f.read()

    # Add import for time if not present
    if "import time" not in content:
        content = "import time\n" + content

    # Add timing to process_job method
    process_job_pattern = r'(def process_job\(self, job\):.*?try:.*?)(self\.update_status\("busy"\).*?)(logger\.info\(f"Processing job: {job\[\'request_id\'\]}"?\).*?)(start_time = time\.time\(\))'
    
    if "# Time each step" not in content:
        # Add detailed timing
        content = re.sub(
            r'(start_time = time\.time\(\))(.*?)(r = requests\.post\(.*?timeout=120.*?\))',
            r'\1\2\n            # Time the generation request\n            gen_start = time.time()\n            \3\n            gen_time = time.time() - gen_start\n            logger.info(f"Generation API call took: {gen_time:.2f}s")',
            content,
            flags=re.DOTALL
        )

    # Add timing to generate endpoint
    if "gen_start = time.time()" not in content:
        content = re.sub(
            r'(def generate\(\):.*?try:)(.*?)(data = request\.json)',
            r'\1\n        gen_start = time.time()\2\3',
            content,
            flags=re.DOTALL
        )
        
        # Add log after generation
        content = re.sub(
            r'(tokens_generated = len\(outputs\[0\]\) - len\(inputs\.input_ids\[0\]\))',
            r'\1\n        gen_time = time.time() - gen_start\n        logger.info(f"Model generation took: {gen_time:.2f}s for {tokens_generated} tokens ({tokens_generated/gen_time:.1f} tok/s)")',
            content
        )

    # Add timing for tokenization
    content = re.sub(
        r'(# Tokenize with proper settings)(.*?)(inputs = tokenizer\(.*?\))',
        r'\1\n        tok_start = time.time()\2\3\n        logger.info(f"Tokenization took: {time.time() - tok_start:.3f}s")',
        content,
        flags=re.DOTALL
    )

    with open('mia_miner_unified.py', 'w') as f:
        f.write(content)

    print("✓ Added detailed timing logs to mia_miner_unified.py")
    print("Restart your miner to see timing breakdown")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)