# Speed Optimization Options for Qwen Miner

## Current Speed: 28-40 tokens/sec

Using:
- Transformers library
- 4-bit quantization (BitsAndBytes)
- Sampling enabled (temperature=0.7)

## Option 1: Switch to vLLM (Recommended)
**Expected: 60-80 tokens/sec**

```python
from vllm import LLM, SamplingParams

model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="half",  # FP16 instead of 4-bit
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    enforce_eager=True
)
```

Pros:
- 2-3x faster than transformers
- Better batching
- Optimized for inference

Cons:
- Uses more VRAM (need ~16GB for FP16)
- Harder to install

## Option 2: Use AWQ Quantization
**Expected: 50-70 tokens/sec**

```python
model = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.95
)
```

Pros:
- Faster than 4-bit BitsAndBytes
- Still fits in 8GB VRAM
- Good balance of speed/memory

## Option 3: Optimize Current Setup
**Expected: 40-50 tokens/sec**

Quick wins without changing backend:
```python
# 1. Enable CUDA optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 2. Use greedy decoding when possible
if temperature == 0:
    do_sample = False  # Much faster

# 3. Use torch.compile() (PyTorch 2.0+)
model = torch.compile(model)

# 4. Reduce threads to 1
serve(app, threads=1)  # Less contention
```

## Option 4: Use GGUF with llama.cpp
**Expected: 45-60 tokens/sec**

Pros:
- Very efficient quantization
- Low memory usage
- Good CPU/GPU hybrid

## Recommendation

For RTX 3090 (24GB VRAM):
1. Use vLLM with FP16 → 60-80 tokens/sec
2. Set gpu_memory_utilization=0.95
3. Use greedy decoding when possible
4. Single-threaded server

For 8GB VRAM GPUs:
1. Use vLLM with AWQ → 50-70 tokens/sec
2. Or optimize current setup → 40-50 tokens/sec