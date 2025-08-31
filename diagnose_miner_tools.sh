#!/bin/bash
# Diagnose what's happening with tool prompts in the miner

cd /data/qwen-awq-miner

# Create a diagnostic script
cat > diagnose_tools.py << 'EOF'
#!/usr/bin/env python3
"""Diagnose tool handling in the miner"""
import json
from typing import List, Dict, Optional

# Import the format function from miner
try:
    from miner import format_prompt_with_tools
    print("✓ Successfully imported format_prompt_with_tools")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    # Try to read and eval the function
    with open('miner.py', 'r') as f:
        content = f.read()
        if 'def format_prompt_with_tools' in content:
            print("✓ Function exists in miner.py")
        else:
            print("✗ Function not found in miner.py")
    exit(1)

# Test the format function
print("\n=== Testing Tool Prompt Format ===\n")

# Test prompt
test_prompt = "What fish dishes do you have?"
test_tools = [{
    "name": "search_menu_items",
    "description": "Search for menu items by ingredient, category, or name",
    "parameters": {
        "search_term": {"type": "string", "description": "The term to search for"},
        "search_type": {"type": "string", "enum": ["ingredient", "category", "name"]}
    }
}]
test_context = {
    "business_name": "Bella Vista Restaurant",
    "system_prompt": "You are Maria, a helpful server."
}

# Format the prompt
formatted = format_prompt_with_tools(test_prompt, test_tools, test_context)

print("Formatted prompt:")
print("-" * 80)
print(formatted)
print("-" * 80)

# Check if it contains the right instructions
if "Function call:" in formatted:
    print("\n✓ Contains 'Function call:' instruction")
else:
    print("\n✗ Missing 'Function call:' instruction")

if "<tool_call>" in formatted:
    print("⚠️  Still contains XML tag instruction (old format)")
else:
    print("✓ No XML tags (good)")

# Check for extract_tool_call function
print("\n=== Checking extract_tool_call ===")
try:
    from miner import extract_tool_call
    print("✓ Successfully imported extract_tool_call")
    
    # Test extraction
    test_responses = [
        'Function call: {"name": "search_menu_items", "parameters": {"search_term": "fish"}}',
        'I\'ll search for fish dishes. Function call: {"name": "search_menu_items", "parameters": {"search_term": "fish", "search_type": "ingredient"}}',
        'Let me check our fish options for you.',
        '<tool_call>{"name": "search_menu_items", "parameters": {"search_term": "fish"}}</tool_call>'
    ]
    
    for i, resp in enumerate(test_responses):
        result = extract_tool_call(resp)
        print(f"\nTest {i+1}: {resp[:50]}...")
        print(f"Extracted: {result}")
        
except ImportError:
    print("✗ Failed to import extract_tool_call")
EOF

# Run the diagnostic
echo "Running diagnostic..."
python3 diagnose_tools.py

# Clean up
rm diagnose_tools.py

echo -e "\n=== Checking miner.py directly ===\n"
# Check if the function was actually updated
if grep -q "Function call:" miner.py; then
    echo "✓ miner.py contains 'Function call:' instruction"
else
    echo "✗ miner.py missing 'Function call:' instruction"
    echo ""
    echo "Checking what format_prompt_with_tools actually contains:"
    grep -A 20 "def format_prompt_with_tools" miner.py | head -30
fi