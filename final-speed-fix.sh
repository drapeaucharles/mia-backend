#!/bin/bash

echo "Applying final speed optimizations..."

cd ~/mia-gpu-miner || cd /opt/mia-gpu-miner
source venv/bin/activate

# Set environment variable to fix tokenizer warning
export TOKENIZERS_PARALLELISM=false

# Install ExLlamaV2 for fastest GPTQ inference
echo "Installing ExLlamaV2 for maximum speed..."
pip install exllamav2

# Create a test script to verify speed
cat > test_exllama.py << 'EOF'
import torch
import time
from transformers import AutoTokenizer

print("Testing inference backends...")

# Test 1: Try ExLlamaV2
try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
    print("✓ ExLlamaV2 available - this is the fastest backend!")
    use_exllama = True
except:
    print("✗ ExLlamaV2 not available")
    use_exllama = False

# Test 2: Check current speed with transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

print("\nLoading model with current setup...")
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-OpenOrca-GPTQ",
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    revision="main",
    low_cpu_mem_usage=True
)

# Benchmark
prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Warmup
with torch.no_grad():
    _ = model.generate(input_ids=inputs.input_ids, max_new_tokens=10, do_sample=False)

# Test
print("\nBenchmarking...")
times = []
for i in range(3):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            max_new_tokens=30,
            do_sample=False,
            use_cache=True
        )
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.2f}s ({30/elapsed:.1f} tok/s)")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.2f}s ({30/avg:.1f} tok/s)")

if avg > 1.5:
    print("\n⚠️ Still too slow! Let's try a different approach...")
    print("\nRecommendations:")
    print("1. Try: pip install auto-gptq==0.7.1 --force-reinstall")
    print("2. Or use: TheBloke/Mistral-7B-OpenOrca-AWQ (AWQ format)")
    print("3. Or use: unquantized model with bitsandbytes 8-bit")
EOF

python test_exllama.py

# Create optimized miner using explicit kernel control
echo -e "\nCreating ultra-optimized miner..."
cat > mia_miner_fast.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Ultra Fast Version
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import torch
import socket
import logging
import requests
import threading
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify
from waitress import serve
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mia-fast')
logging.getLogger("transformers").setLevel(logging.WARNING)

# Force CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Global model variables
model = None
tokenizer = None
app = Flask(__name__)

class ModelServer:
    def __init__(self):
        self.server_thread = None
        self.model_loaded = False
    
    def load_model(self):
        global model, tokenizer
        
        logger.info("Loading model with optimizations...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with specific optimizations
            model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-OpenOrca-GPTQ",
                device_map={"": 0},  # Force everything to GPU 0
                torch_dtype=torch.float16,
                trust_remote_code=True,
                revision="main",
                low_cpu_mem_usage=True,
                use_flash_attention_2=False,  # Disable if causing issues
                load_in_4bit=False,  # Already quantized
                max_memory={0: "20GiB"}  # Limit memory usage
            )
            
            # Move to eval mode
            model.eval()
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile...")
                model = torch.compile(model, mode="max-autotune")
            
            self.model_loaded = True
            logger.info("✓ Model loaded and optimized!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start_server(self):
        def run_server():
            serve(app, host="0.0.0.0", port=8000, threads=1)  # Single thread for consistency
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(2)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready" if model else "loading"})

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = min(data.get("max_tokens", 50), 100)
        
        # Simple prompt format
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with minimal overhead
        with torch.inference_mode():  # Faster than no_grad
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return jsonify({
            "text": response.strip(),
            "tokens_generated": len(generated_ids)
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

# Simplified client - same as before but with minimal logging
class MinerClient:
    def __init__(self):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}-fast"
        self.miner_id = None
    
    def wait_for_model(self):
        for i in range(30):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    def register(self):
        try:
            # Get GPU info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            gpu_name = "Unknown"
            gpu_mem = 0
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                gpu_name = parts[0].strip()
                gpu_mem = int(parts[1].strip().replace(' MiB', ''))
            
            data = {
                "name": self.miner_name,
                "ip_address": requests.get('https://api.ipify.org', timeout=10).text,
                "gpu_name": gpu_name,
                "gpu_memory_mb": gpu_mem,
                "status": "idle"
            }
            
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            if r.status_code == 200:
                self.miner_id = r.json().get('miner_id')
                logger.info(f"Registered as miner {self.miner_id}")
                return True
        except:
            pass
        return False
    
    def run_mining_loop(self):
        while True:
            try:
                r = requests.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    if work and work.get("request_id"):
                        start = time.time()
                        
                        # Generate
                        gen_r = requests.post(
                            f"{self.local_url}/generate",
                            json={
                                "prompt": work.get("prompt", ""),
                                "max_tokens": min(work.get("max_tokens", 50), 100)
                            },
                            timeout=30
                        )
                        
                        if gen_r.status_code == 200:
                            result = gen_r.json()
                            elapsed = time.time() - start
                            
                            # Submit
                            requests.post(
                                f"{self.backend_url}/submit_result",
                                json={
                                    "miner_id": self.miner_id,
                                    "request_id": work["request_id"],
                                    "result": {
                                        "response": result.get("text", ""),
                                        "tokens_generated": result.get("tokens_generated", 0),
                                        "processing_time": elapsed
                                    }
                                },
                                timeout=30
                            )
                            
                            logger.info(f"Completed job in {elapsed:.2f}s")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(10)

def main():
    logger.info("Starting Ultra Fast MIA Miner...")
    
    # Load model
    server = ModelServer()
    if not server.load_model():
        sys.exit(1)
    
    # Start server
    server.start_server()
    
    # Initialize client
    client = MinerClient()
    
    # Wait for server
    if not client.wait_for_model():
        logger.error("Server failed to start")
        sys.exit(1)
    
    # Quick speed test
    logger.info("Testing speed...")
    for prompt in ["Hello", "What is 2+2?"]:
        start = time.time()
        r = requests.post("http://localhost:8000/generate", json={"prompt": prompt, "max_tokens": 20})
        elapsed = time.time() - start
        if r.status_code == 200:
            logger.info(f"'{prompt}' -> {elapsed:.2f}s")
    
    # Register and run
    if client.register():
        client.run_mining_loop()

if __name__ == "__main__":
    main()
EOF

chmod +x mia_miner_fast.py

echo -e "\nTo use the ultra-fast miner:"
echo "1. Stop current miner: pkill -f mia_miner"
echo "2. Run fast version: ./mia_miner_fast.py"
echo ""
echo "If still slow, consider using a different model format:"
echo "- AWQ quantization (often faster than GPTQ)"
echo "- GGUF with llama.cpp"
echo "- Unquantized with bitsandbytes 8-bit"