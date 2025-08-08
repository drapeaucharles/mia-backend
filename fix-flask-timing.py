#!/usr/bin/env python3
import re

# Read the miner file
with open('mia_miner_unified.py', 'r') as f:
    content = f.read()

# Find and update the generate function with detailed timing
generate_function = '''@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        total_start = time.time()
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 500)
        
        # Log what we're generating
        logger.info(f"Generate request - max_tokens: {max_tokens}, prompt length: {len(prompt)}")
        
        # Use proper ChatML format
        format_start = time.time()
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        logger.info(f"Formatting took: {time.time() - format_start:.3f}s")
        
        # Tokenize with proper settings
        tok_start = time.time()
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        # Move to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logger.info(f"Tokenization took: {time.time() - tok_start:.3f}s")
        
        # Generate response
        gen_start = time.time()
        with torch.no_grad():
            # Log generation parameters
            logger.info(f"Generating with max_new_tokens={max_tokens}")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        gen_time = time.time() - gen_start
        
        # Decode the generated tokens only (not including the input)
        decode_start = time.time()
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up any remaining special tokens that might have slipped through
        response = response.replace("<|im_end|>", "").strip()
        response = response.replace("<|im_start|>", "").strip()
        decode_time = time.time() - decode_start
        
        tokens_generated = len(generated_ids)
        total_time = time.time() - total_start
        
        logger.info(f"Generation breakdown - Total: {total_time:.2f}s, Gen: {gen_time:.2f}s ({tokens_generated} tokens, {tokens_generated/gen_time:.1f} tok/s), Decode: {decode_time:.3f}s")
        
        return jsonify({
            "text": response,
            "tokens_generated": tokens_generated,
            "model": "Mistral-7B-OpenOrca-GPTQ"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500'''

# Replace the generate function
pattern = r'@app\.route\("/generate", methods=\["POST"\]\).*?except Exception as e:.*?return jsonify\(\{"error": str\(e\)\}\), 500'
content = re.sub(pattern, generate_function, content, flags=re.DOTALL)

# Save the updated file
with open('mia_miner_unified.py', 'w') as f:
    f.write(content)

print("✓ Added detailed timing to generate endpoint")
print("Restart miner to see where the 11 seconds is spent")