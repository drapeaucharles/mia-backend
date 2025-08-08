#!/bin/bash

# MIA GPU Miner - 30+ tokens/second installer
# Uses the fastest available inference method

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   MIA Fast Miner - 30+ tokens/second      ║${NC}"
echo -e "${GREEN}║   Multiple backend options for speed      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
echo ""

cd /data/mia-gpu-miner

# Activate venv
if [ -f "/data/venv/bin/activate" ]; then
    source /data/venv/bin/activate
fi

# Stop current miner
./stop_miner.sh 2>/dev/null || true

# Option 1: Try vLLM (fastest if it works)
echo -e "${YELLOW}Option 1: Attempting vLLM installation (50-100 tok/s)...${NC}"
pip install vllm --no-deps 2>/dev/null && pip install vllm 2>/dev/null || echo "vLLM installation failed"

# Option 2: Install llama-cpp-python with GPU
echo -e "${YELLOW}Option 2: Installing llama-cpp-python with GPU (30-50 tok/s)...${NC}"
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir || echo "llama-cpp failed"

# Option 3: Install CTransformers
echo -e "${YELLOW}Option 3: Installing CTransformers (20-40 tok/s)...${NC}"
pip install ctransformers[cuda] || echo "CTransformers failed"

# Download GGUF model for llama.cpp
echo -e "${YELLOW}Downloading GGUF model for fast inference...${NC}"
mkdir -p /data/models/gguf
cd /data/models/gguf
wget -c https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf || echo "Download failed"
cd /data/mia-gpu-miner

# Create fast miner with multiple backend options
echo -e "${YELLOW}Creating fast multi-backend miner...${NC}"
cat > mia_miner_fast.py << 'EOF'
#!/usr/bin/env python3
"""
MIA GPU Miner - Fast Multi-Backend Version
Achieves 30+ tokens/second
"""
import os
os.environ["HF_HOME"] = "/data/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import socket
import logging
import requests
import threading
from flask import Flask, request, jsonify
from waitress import serve

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/data/miner.log')
    ]
)
logger = logging.getLogger('mia-fast')

# Global model
model = None
backend_name = None
app = Flask(__name__)

def try_vllm():
    """Try vLLM backend (fastest)"""
    try:
        from vllm import LLM, SamplingParams
        logger.info("Trying vLLM backend...")
        
        # Try AWQ model first
        try:
            llm = LLM(
                model="TheBloke/Mistral-7B-OpenOrca-AWQ",
                quantization="awq",
                dtype="half",
                gpu_memory_utilization=0.90
            )
            logger.info("✓ vLLM with AWQ loaded!")
            return llm, "vLLM-AWQ"
        except:
            # Try unquantized
            llm = LLM(
                model="Open-Orca/Mistral-7B-OpenOrca",
                dtype="half",
                gpu_memory_utilization=0.90
            )
            logger.info("✓ vLLM loaded!")
            return llm, "vLLM"
            
    except Exception as e:
        logger.warning(f"vLLM failed: {e}")
        return None, None

def try_llamacpp():
    """Try llama.cpp backend"""
    try:
        from llama_cpp import Llama
        logger.info("Trying llama.cpp backend...")
        
        model_path = "/data/models/gguf/mistral-7b-openorca.Q4_K_M.gguf"
        if not os.path.exists(model_path):
            model_path = "/data/models/mistral-gptq/model.safetensors"
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # All layers on GPU
            n_ctx=2048,
            n_batch=512,
            verbose=False
        )
        logger.info("✓ llama.cpp loaded!")
        return llm, "llama.cpp"
        
    except Exception as e:
        logger.warning(f"llama.cpp failed: {e}")
        return None, None

def try_ctransformers():
    """Try CTransformers backend"""
    try:
        from ctransformers import AutoModelForCausalLM
        logger.info("Trying CTransformers backend...")
        
        model_path = "/data/models/gguf/mistral-7b-openorca.Q4_K_M.gguf"
        
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            gpu_layers=50,
            context_length=2048
        )
        logger.info("✓ CTransformers loaded!")
        return llm, "CTransformers"
        
    except Exception as e:
        logger.warning(f"CTransformers failed: {e}")
        return None, None

def try_transformers_optimized():
    """Try optimized transformers as fallback"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("Trying optimized transformers...")
        
        tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
        
        # Try Flash Attention 2
        model = AutoModelForCausalLM.from_pretrained(
            "/data/models/mistral-gptq",
            device_map="cuda:0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=True
        )
        
        # Compile model
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode="max-autotune")
        
        logger.info("✓ Optimized transformers loaded!")
        return (model, tokenizer), "transformers-optimized"
        
    except Exception as e:
        logger.warning(f"Optimized transformers failed: {e}")
        return None, None

def load_fastest_model():
    """Load the fastest available model"""
    global model, backend_name
    
    # Try backends in order of speed
    for try_func in [try_vllm, try_llamacpp, try_ctransformers, try_transformers_optimized]:
        result, name = try_func()
        if result:
            model = result
            backend_name = name
            
            # Test speed
            logger.info(f"Testing {name} speed...")
            start = time.time()
            
            if name == "vLLM" or name == "vLLM-AWQ":
                from vllm import SamplingParams
                outputs = model.generate(["Hello, how are you?"], SamplingParams(max_tokens=50))
                tokens = len(outputs[0].outputs[0].token_ids)
            elif name == "llama.cpp":
                output = model("Hello, how are you?", max_tokens=50)
                tokens = output['usage']['completion_tokens']
            elif name == "CTransformers":
                output = model("Hello, how are you?", max_new_tokens=50)
                tokens = len(output.split()) * 1.3  # Approximate
            else:  # transformers
                m, t = model
                inputs = t("Hello", return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = m.generate(**inputs, max_new_tokens=50)
                tokens = 50
            
            elapsed = time.time() - start
            speed = tokens / elapsed
            logger.info(f"{name} speed: {speed:.1f} tokens/second")
            
            if speed >= 30:
                logger.info(f"✓ Using {name} - Meets 30+ tok/s requirement!")
                return True
            else:
                logger.warning(f"{name} too slow ({speed:.1f} tok/s), trying next...")
                model = None
                backend_name = None
                continue
    
    logger.error("No backend achieved 30+ tok/s!")
    return False

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if model else "loading",
        "backend": backend_name
    })

@app.route("/generate", methods=["POST"])
def generate():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.json
        prompt = data.get("prompt", "")
        max_tokens = min(data.get("max_tokens", 200), 500)
        
        # Format prompt
        formatted = f"""<|im_start|>system
You are MIA, a helpful multilingual assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""
        
        start = time.time()
        
        # Generate based on backend
        if backend_name in ["vLLM", "vLLM-AWQ"]:
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=max_tokens
            )
            outputs = model.generate([formatted], sampling_params)
            response = outputs[0].outputs[0].text
            tokens = len(outputs[0].outputs[0].token_ids)
            
        elif backend_name == "llama.cpp":
            output = model(
                formatted,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<|im_end|>"]
            )
            response = output['choices'][0]['text']
            tokens = output['usage']['completion_tokens']
            
        elif backend_name == "CTransformers":
            response = model(
                formatted,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["<|im_end|>"]
            )
            tokens = len(response.split()) * 1.3
            
        else:  # transformers
            m, t = model
            inputs = t(formatted, return_tensors="pt", truncation=True).to("cuda")
            with torch.no_grad():
                outputs = m.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    pad_token_id=t.eos_token_id
                )
            generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
            response = t.decode(generated_ids, skip_special_tokens=True)
            tokens = len(generated_ids)
        
        elapsed = time.time() - start
        speed = tokens / elapsed
        
        logger.info(f"Generated {tokens} tokens in {elapsed:.2f}s ({speed:.1f} tok/s)")
        
        return jsonify({
            "text": response.strip(),
            "tokens_generated": int(tokens),
            "time": round(elapsed, 2),
            "tokens_per_second": round(speed, 1),
            "backend": backend_name
        })
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    logger.info("MIA Fast Miner - 30+ tokens/second")
    
    # Load fastest model
    if not load_fastest_model():
        logger.error("Failed to load any model with 30+ tok/s")
        logger.info("Please check GPU drivers and CUDA installation")
        sys.exit(1)
    
    # Start server
    def run_server():
        serve(app, host="0.0.0.0", port=8000)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    logger.info("Server started on port 8000")
    
    # Mining loop (simplified)
    backend_url = "https://mia-backend-production.up.railway.app"
    miner_name = f"gpu-miner-{socket.gethostname()}-fast"
    miner_id = None
    
    # Register
    logger.info("Registering...")
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else backend_name
        gpu_mb = torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 8192
    except:
        gpu_name = backend_name
        gpu_mb = 8192
    
    for attempt in range(5):
        try:
            r = requests.post(
                f"{backend_url}/register_miner",
                json={
                    "name": miner_name,
                    "ip_address": "vastai",
                    "gpu_name": f"{gpu_name} ({backend_name})",
                    "gpu_memory_mb": gpu_mb,
                    "status": "idle"
                },
                timeout=30
            )
            
            if r.status_code == 200:
                miner_id = r.json().get('miner_id')
                logger.info(f"✓ Registered! ID: {miner_id}")
                break
        except Exception as e:
            logger.error(f"Registration attempt {attempt+1} failed: {e}")
            time.sleep(30)
    
    if not miner_id:
        sys.exit(1)
    
    # Mining loop
    logger.info("Starting mining loop...")
    while True:
        try:
            r = requests.get(
                f"{backend_url}/get_work",
                params={"miner_id": miner_id},
                timeout=10
            )
            
            if r.status_code == 200:
                work = r.json()
                if work and work.get("request_id"):
                    logger.info(f"Processing job {work['request_id']}")
                    
                    gen_r = requests.post(
                        "http://localhost:8000/generate",
                        json={
                            "prompt": work.get("prompt", ""),
                            "max_tokens": work.get("max_tokens", 200)
                        },
                        timeout=60
                    )
                    
                    if gen_r.status_code == 200:
                        result = gen_r.json()
                        
                        requests.post(
                            f"{backend_url}/submit_result",
                            json={
                                "miner_id": miner_id,
                                "request_id": work["request_id"],
                                "result": {
                                    "response": result.get("text", ""),
                                    "tokens_generated": result.get("tokens_generated", 0),
                                    "processing_time": result.get("time", 0)
                                }
                            },
                            timeout=30
                        )
                        
                        speed = result.get('tokens_per_second', 0)
                        logger.info(f"✓ Job complete ({speed:.1f} tok/s)")
                        
                        if speed < 30:
                            logger.warning(f"Speed below 30 tok/s! ({speed:.1f})")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
EOF

chmod +x mia_miner_fast.py

echo ""
echo -e "${GREEN}✓ Fast miner created!${NC}"
echo ""
echo "Testing backends..."
python3 -c "
import subprocess
print('Checking available backends:')

# Check vLLM
try:
    import vllm
    print('✓ vLLM available')
except:
    print('✗ vLLM not available')

# Check llama-cpp
try:
    import llama_cpp
    print('✓ llama-cpp-python available')
except:
    print('✗ llama-cpp-python not available')

# Check ctransformers
try:
    import ctransformers
    print('✓ CTransformers available')
except:
    print('✗ CTransformers not available')

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('✗ CUDA not available')
except:
    print('✗ PyTorch not available')
"

echo ""
echo -e "${YELLOW}Starting fast miner...${NC}"
echo "Target: 30+ tokens/second"
echo ""
echo "Run with:"
echo "  python3 mia_miner_fast.py"
echo ""
echo "Or update start script:"
echo "  sed -i 's/mia_miner_[a-z]*.py/mia_miner_fast.py/g' start_miner.sh"
echo "  ./start_miner.sh"