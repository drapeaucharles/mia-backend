# MIA GPU Miner - Fast Installation Guide

## Quick Install (60+ tokens/second)

This installer uses the **proven vLLM-AWQ configuration** that achieves 60+ tokens/second.

### For Vast.ai with /data volume:
```bash
curl -s https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-miner-vllm-awq-final.sh | bash
```

### Start the miner:
```bash
cd /data/mia-gpu-miner  # or ~/mia-gpu-miner
./start_miner.sh
```

### Monitor performance:
```bash
tail -f /data/miner.log | grep "tok/s"
```

## What This Installer Does

1. **Uses vLLM** - The fastest inference engine
2. **Loads AWQ model** - `TheBloke/Mistral-7B-OpenOrca-AWQ`
3. **Optimized settings** - 90% GPU memory utilization
4. **Proven configuration** - Exactly what achieved 60+ tok/s in testing

## Performance

- **Speed**: 60+ tokens/second
- **Model**: Mistral-7B-OpenOrca-AWQ
- **Backend**: vLLM with AWQ quantization
- **Multilingual**: Full support for multiple languages

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher
- **Python**: 3.8+
- **Disk**: 15GB free space

## Troubleshooting

If you get less than 30 tok/s:
1. Check GPU: `nvidia-smi`
2. Verify CUDA: `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv`
3. Check logs: `tail -50 /data/miner.log`

## Files Created

- `/data/mia-gpu-miner/mia_miner_vllm_awq.py` - The miner script
- `/data/venv/` - Python virtual environment
- `/data/miner.log` - Log file
- `/data/huggingface/` - Model cache

This is the final, optimized version that has been tested and proven to work at 60+ tokens/second.