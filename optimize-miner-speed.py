#!/usr/bin/env python3
"""
Optimize MIA miner for fast inference - target sub-3 second responses
"""
import os
import shutil

# Backup current miner
if os.path.exists('mia_miner_unified.py'):
    shutil.copy('mia_miner_unified.py', 'mia_miner_unified.py.backup')

# Create optimized miner with all fixes
optimized_miner = '''#!/usr/bin/env python3
"""
MIA Unified GPU Miner - Speed Optimized Version
"""
import os
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
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# Configure logging - reduce verbosity in production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mia-unified')

# Reduce transformers logging
logging.getLogger("transformers").setLevel(logging.WARNING)

# Global model variables
model = None
tokenizer = None
app = Flask(__name__)

class ModelServer:
    """Embedded model server"""
    
    def __init__(self):
        self.server_thread = None
        self.model_loaded = False
    
    def load_model(self):
        """Load the AI model with AutoGPTQ for optimal speed"""
        global model, tokenizer
        
        logger.info("Loading AI model with AutoGPTQ...")
        
        try:
            # Load tokenizer from original model
            tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Set pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with AutoGPTQ for maximum speed
            model_name = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
            logger.info(f"Loading GPTQ model from {model_name}")
            
            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                model_basename="model",
                use_safetensors=True,
                device="cuda:0",
                use_triton=False,  # Disable triton, use cuda kernels
                quantize_config=None,
                max_memory=None,
                trust_remote_code=True,
                inject_fused_attention=True,  # Enable fused attention for speed
                inject_fused_mlp=True,  # Enable fused MLP
                use_cuda_fp16=True,  # Use FP16 for speed
                disable_exllama=False  # Use ExLlama backend
            )
            
            # Ensure model is on GPU
            logger.info(f"Model loaded on: cuda:0")
            
            self.model_loaded = True
            logger.info("✓ Model loaded successfully with AutoGPTQ!")
            return True
            
        except ImportError:
            logger.warning("AutoGPTQ not available, falling back to transformers")
            # Fallback to regular transformers
            from transformers import AutoModelForCausalLM
            
            tokenizer_name = "Open-Orca/Mistral-7B-OpenOrca"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-OpenOrca-GPTQ",
                device_map="cuda:0",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                revision="main"
            )
            
            self.model_loaded = True
            logger.info("✓ Model loaded with transformers (slower fallback)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        def run_server():
            logger.info("Starting inference server on port 8000...")
            serve(app, host="0.0.0.0", port=8000, threads=4, connection_limit=100)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        time.sleep(3)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model is not None else "loading",
        "model": "Mistral-7B-OpenOrca-GPTQ-Optimized"
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        start_time = time.time()
        data = request.json
        prompt = data.get("prompt", "")
        
        # OPTIMIZATION 1: Limit max tokens for faster response
        max_tokens = min(data.get("max_tokens", 50), 100)  # Cap at 100 for speed
        
        # Format prompt
        system_message = "You are MIA, a helpful AI assistant. Please provide helpful, accurate, and friendly responses."
        formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # OPTIMIZATION 2: Fix tokenizer truncation warning
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # Explicit max_length
            return_attention_mask=True
        )
        
        # Move to GPU
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        
        # OPTIMIZATION 3: Use deterministic generation for speed
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,  # Use max_new_tokens, not max_length
                    do_sample=False,  # Deterministic for speed
                    temperature=0.7,  # Ignored when do_sample=False
                    num_beams=1,  # No beam search for speed
                    use_cache=True,  # Use KV cache
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True  # Stop when EOS is generated
                )
        
        # Decode only generated tokens
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        
        tokens_generated = len(generated_ids)
        total_time = time.time() - start_time
        
        # Only log timing in debug mode
        if logger.level <= logging.DEBUG:
            logger.debug(f"Generated {tokens_generated} tokens in {total_time:.2f}s ({tokens_generated/total_time:.1f} tok/s)")
        
        return jsonify({
            "text": response,
            "tokens_generated": tokens_generated,
            "model": "Mistral-7B-OpenOrca-GPTQ"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

class MinerClient:
    """Miner client that connects to backend"""
    
    def __init__(self, model_server):
        self.backend_url = "https://mia-backend-production.up.railway.app"
        self.local_url = "http://localhost:8000"
        self.miner_name = f"gpu-miner-{socket.gethostname()}"
        self.miner_id = None
        self.model_server = model_server
    
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'name': parts[0].strip(),
                    'memory_mb': int(parts[1].strip().replace(' MiB', ''))
                }
        except:
            pass
        return {'name': 'Unknown GPU', 'memory_mb': 0}
    
    def wait_for_model(self):
        """Wait for model server to be ready"""
        logger.info("Waiting for model server...")
        
        for i in range(30):
            try:
                r = requests.get(f"{self.local_url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("status") == "ready":
                        logger.info("✓ Model server ready")
                        return True
            except:
                pass
            time.sleep(2)
        
        return False
    
    def test_generation(self):
        """Test the generation with a simple prompt"""
        logger.info("Testing generation speed...")
        try:
            test_prompts = ["Hello", "How are you?", "What is 2+2?"]
            
            for prompt in test_prompts:
                start = time.time()
                r = requests.post(
                    f"{self.local_url}/generate",
                    json={"prompt": prompt, "max_tokens": 30},
                    timeout=10
                )
                elapsed = time.time() - start
                
                if r.status_code == 200:
                    result = r.json()
                    tokens = result.get('tokens_generated', 0)
                    logger.info(f"'{prompt}' -> {elapsed:.2f}s ({tokens} tokens)")
                    
                    if elapsed > 3:
                        logger.warning("Response slower than 3s target!")
                else:
                    logger.error(f"Test failed: {r.status_code}")
                    
        except Exception as e:
            logger.error(f"Test error: {e}")
    
    def register(self):
        """Register with MIA backend"""
        try:
            gpu_info = self.get_gpu_info()
            
            try:
                ip = requests.get('https://api.ipify.org', timeout=10).text
            except:
                ip = "unknown"
            
            data = {
                "name": self.miner_name,
                "ip_address": ip,
                "gpu_name": gpu_info['name'],
                "gpu_memory_mb": gpu_info['memory_mb'],
                "status": "idle"
            }
            
            logger.info(f"Registering miner: {self.miner_name}")
            
            r = requests.post(f"{self.backend_url}/register_miner", json=data, timeout=30)
            
            if r.status_code == 200:
                resp = r.json()
                self.miner_id = resp.get('miner_id')
                logger.info(f"✓ Registered successfully! Miner ID: {self.miner_id}")
                return True
            else:
                logger.error(f"Registration failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    def update_status(self, status):
        """Update miner status"""
        if not self.miner_id:
            return
        
        try:
            requests.post(
                f"{self.backend_url}/miner/{self.miner_id}/status",
                json={"status": status},
                timeout=5
            )
        except:
            pass
    
    def process_job(self, job):
        """Process a single job"""
        try:
            self.update_status("busy")
            
            logger.info(f"Processing job: {job['request_id']}")
            start_time = time.time()
            
            # Call local model server
            r = requests.post(
                f"{self.local_url}/generate",
                json={
                    "prompt": job.get("prompt", ""),
                    "max_tokens": job.get("max_tokens", 50)  # Default to 50 for speed
                },
                timeout=30  # Reduced timeout
            )
            
            if r.status_code == 200:
                result = r.json()
                processing_time = time.time() - start_time
                
                # Submit result
                submit_data = {
                    "miner_id": self.miner_id,
                    "request_id": job["request_id"],
                    "result": {
                        "response": result.get("text", ""),
                        "tokens_generated": result.get("tokens_generated", 0),
                        "processing_time": processing_time
                    }
                }
                
                submit_r = requests.post(
                    f"{self.backend_url}/submit_result",
                    json=submit_data,
                    timeout=30
                )
                
                if submit_r.status_code == 200:
                    logger.info(f"✓ Completed job {job['request_id']} in {processing_time:.2f}s")
                    return True
                else:
                    logger.error(f"Failed to submit result: {submit_r.status_code}")
                    return False
            else:
                logger.error(f"Generation failed: {r.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            return False
        finally:
            self.update_status("idle")
    
    def run_mining_loop(self):
        """Main mining loop"""
        logger.info("Starting mining loop...")
        consecutive_errors = 0
        
        while True:
            try:
                # Get work from backend
                r = requests.get(
                    f"{self.backend_url}/get_work",
                    params={"miner_id": self.miner_id},
                    timeout=10
                )
                
                if r.status_code == 200:
                    work = r.json()
                    
                    if work and work.get("request_id"):
                        # Process the job
                        if self.process_job(work):
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                    else:
                        # No work available
                        consecutive_errors = 0
                else:
                    logger.warning(f"Failed to get work: {r.status_code}")
                    consecutive_errors += 1
                
                # Check if too many errors
                if consecutive_errors > 10:
                    logger.error("Too many consecutive errors, restarting...")
                    time.sleep(60)
                    consecutive_errors = 0
                
                # Wait before next poll
                time.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Mining loop error: {e}")
                consecutive_errors += 1
                time.sleep(min(consecutive_errors * 5, 60))

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("MIA Unified GPU Miner - Speed Optimized")
    logger.info("=" * 60)
    
    # Initialize model server
    model_server = ModelServer()
    
    # Load model first
    if not model_server.load_model():
        logger.error("Failed to load model, exiting")
        sys.exit(1)
    
    # Start inference server
    model_server.start_server()
    
    # Initialize miner client
    miner = MinerClient(model_server)
    
    # Wait for model server to be ready
    if not miner.wait_for_model():
        logger.error("Model server failed to start")
        sys.exit(1)
    
    # Test generation speed
    logger.info("Running speed tests...")
    miner.test_generation()
    
    # Register with backend
    attempts = 0
    while not miner.register():
        attempts += 1
        if attempts > 5:
            logger.error("Failed to register after 5 attempts")
            sys.exit(1)
        logger.info(f"Retrying registration in 30s... (attempt {attempts}/5)")
        time.sleep(30)
    
    # Start mining
    try:
        miner.run_mining_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

# Write the optimized miner
with open('mia_miner_unified.py', 'w') as f:
    f.write(optimized_miner)

print("✓ Created optimized miner with:")
print("  - AutoGPTQ for fast inference")
print("  - Fixed tokenizer truncation warning")
print("  - Deterministic generation (do_sample=False)")
print("  - Limited max_tokens for speed")
print("  - Reduced logging verbosity")
print("  - GPU optimizations enabled")
print("")
print("The miner will now:")
print("  1. Load model with AutoGPTQ (if available)")
print("  2. Run speed tests on startup")
print("  3. Target <3 second responses")
print("")
print("Restart your miner to apply optimizations!")