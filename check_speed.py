#!/usr/bin/env python3
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Checking inference speed...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Quick load test
print("\nLoading model...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-OpenOrca-GPTQ",
    device_map="auto",
    trust_remote_code=True,
    revision="main"
)
load_time = time.time() - start
print(f"Load time: {load_time:.2f}s")

# Check device
print(f"Model device: {next(model.parameters()).device}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

# Speed test
test_prompt = "Hello"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
print(f"\nGenerating 50 tokens...")

start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
gen_time = time.time() - start

print(f"Generation time: {gen_time:.2f}s")
print(f"Tokens/second: {50/gen_time:.2f}")

if gen_time > 5:
    print("\n⚠️  SLOW! Should be <2s on RTX 3090")
    print("Possible issues:")
    print("- Model may be on CPU")
    print("- GPTQ kernels not optimized")
    print("- Try: pip install auto-gptq --upgrade --force-reinstall")