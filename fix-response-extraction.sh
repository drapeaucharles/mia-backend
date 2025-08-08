#!/bin/bash

# Fix script for response extraction issue in MIA miners
# The responses are being cut off at the beginning

echo "Fixing MIA miner response extraction..."

# Detect installation directory
if [ -d "/opt/mia-gpu-miner" ]; then
    INSTALL_DIR="/opt/mia-gpu-miner"
    echo "Found installation at /opt/mia-gpu-miner"
elif [ -d "$HOME/mia-gpu-miner" ]; then
    INSTALL_DIR="$HOME/mia-gpu-miner"
    echo "Found installation at ~/mia-gpu-miner"
else
    echo "Error: No miner installation found"
    exit 1
fi

cd "$INSTALL_DIR"

# Create a test script to debug the issue
cat > "$INSTALL_DIR/test_generation.py" << 'EOF'
#!/usr/bin/env python3
import requests
import json
import sys

def test_generation(prompt="Hello"):
    """Test the generation endpoint"""
    url = "http://localhost:8000/generate"
    
    print(f"Testing with prompt: '{prompt}'")
    print("-" * 50)
    
    try:
        response = requests.post(
            url,
            json={"prompt": prompt, "max_tokens": 50},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data.get('text', 'No text in response')}")
            print(f"Tokens generated: {data.get('tokens_generated', 0)}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    test_generation(prompt)
EOF

chmod +x "$INSTALL_DIR/test_generation.py"

# Update the miner with better response extraction
cat > "$INSTALL_DIR/fix_response.py" << 'EOF'
import os
import sys

# Read the current miner file
miner_file = "mia_miner_unified.py"
if os.path.exists("mia_miner_unified_fixed.py"):
    miner_file = "mia_miner_unified_fixed.py"

with open(miner_file, 'r') as f:
    content = f.read()

# Fix the response extraction logic
old_extraction = '''        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in full_response:
            response = full_response.split("<|im_start|>assistant")[-1].strip()
        else:
            response = full_response[len(formatted_prompt):].strip()
        # Clean up any remaining tokens
        response = response.replace("<|im_end|>", "").strip()'''

new_extraction = '''        # Decode response - skip special tokens but keep the structure
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Debug logging
        logger.debug(f"Full response: {full_response[:500]}...")
        
        # Extract only the assistant's response
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in full_response:
            # Split and get everything after the assistant marker
            parts = full_response.split(assistant_marker)
            if len(parts) > 1:
                response = parts[-1].strip()
            else:
                response = full_response[len(formatted_prompt):].strip()
        else:
            # Fallback: try to extract after the prompt
            response = full_response[len(formatted_prompt):].strip()
        
        # Clean up any remaining special tokens
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>", "").strip()
        
        # Remove any incomplete sentences at the start (common with some tokenizers)
        # This helps when the model starts mid-word
        if response and not response[0].isupper() and not response.startswith('"'):
            # Find the first capital letter or punctuation
            for i, char in enumerate(response):
                if char.isupper() or char in '.!?"':
                    response = response[i:]
                    break
        
        logger.info(f"Extracted response: {response[:100]}...")'''

# Replace the extraction logic
content = content.replace(old_extraction, new_extraction)

# Also add debug logging to the generation endpoint
if "logger.info(f\"Formatted prompt: {formatted_prompt[:200]}...\")" not in content:
    content = content.replace(
        'formatted_prompt = f"""<|im_start|>system',
        'logger.info(f"Processing prompt: {prompt}")\n        formatted_prompt = f"""<|im_start|>system'
    )

# Write the fixed file
with open(miner_file, 'w') as f:
    f.write(content)

print("✓ Fixed response extraction logic")
EOF

# Run the fix
cd "$INSTALL_DIR"
source venv/bin/activate
python fix_response.py

# Clean up
rm -f fix_response.py

# Restart the service
if [ -f "/etc/systemd/system/mia-miner.service" ]; then
    echo "Restarting systemd service..."
    sudo systemctl restart mia-miner
    echo "✓ Service restarted"
else
    echo ""
    echo "Please restart your miner manually to apply the fix."
fi

echo ""
echo "✓ Response extraction has been fixed!"
echo ""
echo "To test the fix:"
echo "  $INSTALL_DIR/test_generation.py \"Hello\""
echo "  $INSTALL_DIR/test_generation.py \"What is 2+2?\""
echo ""
echo "The responses should now start properly without being cut off."