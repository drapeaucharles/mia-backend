# Keep MIA Model Hot in GPU Memory

## Problem
Model gets unloaded from GPU memory, causing slow "cold starts" for new requests.

## Solutions

### 1. For vLLM Backend (Current Setup)

In your miner script, add these configurations:

```python
# When initializing vLLM
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    quantization="awq",
    dtype="half",
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    enforce_eager=True,  # Important: Prevents model unloading
    swap_space=0,  # Disable CPU offloading
    disable_log_requests=True
)

# Keep a global reference
GLOBAL_MODEL = model  # Prevents garbage collection

# Add periodic inference to prevent idle timeout
def keep_warm():
    """Run periodic inference to keep model in GPU"""
    while True:
        try:
            time.sleep(300)  # Every 5 minutes
            _ = model.generate(["Hello"], SamplingParams(max_tokens=1))
        except:
            pass

# Start keep-warm thread
import threading
warmup_thread = threading.Thread(target=keep_warm, daemon=True)
warmup_thread.start()
```

### 2. For Transformers Backend

```python
# Prevent model offloading
model.to('cuda')
model.eval()

# Pin model in memory
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.empty_cache()

# Disable gradient checkpointing
model.gradient_checkpointing_enable = False
```

### 3. System-Level Solutions

#### GPU Persistence Mode (NVIDIA)
```bash
# Enable persistence mode - keeps driver loaded
sudo nvidia-smi -pm 1
```

#### Disable GPU Power Management
```bash
# Set to maximum performance
sudo nvidia-smi -pl 300  # Set power limit to max (300W for 3090)
```

#### CUDA Environment Variables
Add to your miner startup script:
```bash
export CUDA_MODULE_LOADING=EAGER  # Load all CUDA modules upfront
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

### 4. Miner Service Configuration

Update your systemd service or startup script:

```ini
[Service]
Type=simple
Restart=always
RestartSec=10
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="CUDA_MODULE_LOADING=EAGER"
Environment="OMP_NUM_THREADS=1"
# Nice=-5  # Higher priority
# CPUAffinity=0-3  # Pin to specific CPU cores
```

### 5. Railway/Production Specific

For Railway deployment:
```python
# In main.py, add startup warming
@app.on_event("startup")
async def startup_event():
    # Warm up on startup
    warm_up_model()
    
    # Schedule periodic warmups
    scheduler.add_job(
        warm_up_model,
        'interval',
        minutes=5,
        id='model_warmup'
    )
```

## Monitoring

Check if model is staying hot:
```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Check for model reload in logs
tail -f miner.log | grep -i "loading\|model\|initializ"
```

## Expected Results

- First request after startup: 3-5 seconds
- Subsequent requests: <1 second overhead
- No "model loading" in logs after initial startup
- Consistent GPU memory usage (no drops)