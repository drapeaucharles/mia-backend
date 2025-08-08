# MIA GPU Miner - Production Version

A high-performance GPU miner for the MIA distributed inference network, achieving **60+ tokens/second** with proper multilingual support.

## Features

- ✅ **60+ tokens/second** inference speed using vLLM with AWQ quantization
- ✅ **Concurrent job processing** with dynamic VRAM management
- ✅ **Proper language detection** - responds in the user's language
- ✅ **Optimized for Vast.ai** containers with /data volume support
- ✅ **Auto-recovery** from errors and connection issues

## Quick Install

One-line installation on any GPU system:

```bash
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-mia-miner.sh | bash
```

## Manual Installation

```bash
# Clone repository
git clone https://github.com/drapeaucharles/mia-backend.git
cd mia-backend

# Run installer
bash install-mia-miner.sh
```

## Usage

### Start Miner (Interactive)
```bash
cd /data/mia-gpu-miner  # or ~/mia-gpu-miner
./run_miner.sh
```

### Start Miner (Background)
```bash
cd /data/mia-gpu-miner
./start_miner.sh
```

### Stop Miner
```bash
./stop_miner.sh
```

### View Logs
```bash
tail -f /data/miner.log
```

### Test Language Detection
```bash
./test_miner.sh
```

## Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: 11.8 or higher
- **Python**: 3.8+
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection

## Performance

The miner achieves:
- **60+ tokens/second** on most modern GPUs
- Concurrent processing of multiple jobs
- Automatic VRAM management
- Sub-second job pickup time

## Language Support

The miner automatically detects and responds in:
- English
- Spanish  
- French
- German
- Italian
- Portuguese
- And many more languages

## Vast.ai Setup

The installer automatically detects Vast.ai environments and:
- Uses `/data` volume for persistent storage
- Configures HuggingFace cache in `/data/huggingface`
- Handles container restarts gracefully

## Files

- `mia_miner_production.py` - Main production miner with all features
- `mia_miner_concurrent.py` - Concurrent version without language fix
- `mia_miner_concurrent_robust.py` - Alternative with robust language detection
- `install-mia-miner.sh` - Production installer
- `install-miner-legacy.sh` - Legacy installer (kept for compatibility)

## Troubleshooting

### Miner not starting
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check CUDA
nvidia-smi

# Check logs
tail -100 /data/miner.log
```

### Language detection issues
The production miner uses forced language detection. If issues persist, check logs for detected language.

### Performance issues
- Ensure GPU has sufficient VRAM (8GB+)
- Check GPU utilization with `nvidia-smi`
- Verify internet connection stability

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/drapeaucharles/mia-backend/issues)
- Check logs in `/data/miner.log`