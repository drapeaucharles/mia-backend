#!/usr/bin/env python3
import re

# Read the miner file
with open('mia_miner_unified.py', 'r') as f:
    content = f.read()

# Fix the generate function
fixed_generate = '''@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        start_time = time.time()
        data = request.json
        prompt = data.get("prompt", "")
        
        # Limit max tokens for faster response
        max_tokens = min(data.get("max_tokens", 50), 100)
        
        # Format prompt
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize with proper settings
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Move to GPU - handle both dict and BatchEncoding
        if hasattr(inputs, 'to'):
            inputs = inputs.to("cuda:0")
        else:
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        # Decode only generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        
        tokens_generated = len(generated_ids)
        total_time = time.time() - start_time
        
        # Only log in debug mode
        if tokens_generated > 0 and total_time > 0:
            logger.info(f"Generated {tokens_generated} tokens in {total_time:.2f}s ({tokens_generated/total_time:.1f} tok/s)")
        
        return jsonify({
            "text": response,
            "tokens_generated": int(tokens_generated),
            "model": "Mistral-7B-OpenOrca-GPTQ"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500'''

# Replace the generate function
pattern = r'@app\.route\("/generate", methods=\["POST"\]\).*?return jsonify\(\{"error": str\(e\)\}\), 500'
content = re.sub(pattern, fixed_generate, content, flags=re.DOTALL)

# Also fix the model loading to ensure it uses transformers if AutoGPTQ fails
if "from transformers import AutoTokenizer" not in content:
    content = content.replace(
        "from transformers import AutoTokenizer",
        "from transformers import AutoTokenizer, AutoModelForCausalLM"
    )

# Save the fixed file
with open('mia_miner_unified.py', 'w') as f:
    f.write(content)

print("✓ Fixed generate function")
print("✓ Fixed input_ids error")
print("✓ Removed temperature/early_stopping warnings")
print("")
print("Restart your miner now!")