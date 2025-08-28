# MIA GPU Miner Installation

## 🚀 One Command Installation

Install and run the MIA GPU miner with a single command:

```bash
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/universal-installer.sh | bash
```

**Note:** On fresh VPS, install pip first:
```bash
apt update && apt install -y python3-pip
```

## What This Installs

- **Model**: Qwen2.5-7B-Instruct (4-bit quantized)
- **Performance**: 25-30 tokens/second
- **Storage**: ~6GB disk space
- **Backend**: Transformers with BitsAndBytes quantization
- **Memory**: Uses 8GB+ VRAM

## System Requirements

- NVIDIA GPU with 8GB+ VRAM
- Ubuntu/Debian Linux
- CUDA 11.7+ drivers
- 20GB free disk space
- Python 3.8+

## After Installation

The miner will:
1. Auto-register with MIA backend
2. Start processing jobs automatically
3. Run on port 8000

### Useful Commands

```bash
# Check if miner is running
curl http://localhost:8000/health

# View logs
tail -f /data/mia-qwen/miner.log

# Test generation
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'

# Stop miner
pkill -f miner

# Restart miner
cd /data/mia-qwen && python3 qwen_miner.py
```

## Troubleshooting

If the installation fails:

1. **No pip**: Run `apt-get update && apt-get install -y python3-pip`
2. **CUDA issues**: Make sure you have NVIDIA drivers installed
3. **Port in use**: Kill existing process with `fuser -k 8000/tcp`

## Support

For issues or questions, contact the MIA team.