# MIA Backend - AI Job Queue and Mining System

MIA (Multi-model Inference API) is a distributed AI inference system that supports job queuing, GPU miner management, and OpenAI-compatible tool calling.

## 🚀 Latest Updates (September 2025)

### Tool Calling Support
- **OpenAI-compatible**: Function/tool calling using Hermes parser
- **Auto-detection**: AI decides when to use tools
- **Restaurant Integration**: Seamless menu queries and dish details
- **Fixed & Tested**: All tool calling issues resolved

### vLLM Performance
- **Model**: Qwen2.5-7B-Instruct-AWQ (4-bit quantization)
- **Speed**: 60-80 tokens/second on RTX 3090
- **Context**: 12k tokens with xFormers attention
- **Tools**: Full support for structured function calling

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Restaurant     │────▶│  MIA Backend │────▶│ Redis Queue │
│  Frontend       │     └──────────────┘     └─────────────┘
└─────────────────┘            │
                               ▼
                        ┌──────────────┐
                        │  GPU Miners  │
                        │ (vLLM + AWQ) │
                        └──────────────┘
```

## Quick Start - GPU Miner

### One-Line Install (Recommended)

```bash
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-final.sh | bash
```

This installer:
- ✅ Auto-detects CUDA and Python versions
- ✅ Installs vLLM 0.10.1.1 with xFormers
- ✅ Downloads Qwen2.5-7B-Instruct-AWQ
- ✅ Enables Hermes tool parser
- ✅ Auto-registers with backend
- ✅ Starts job polling immediately

### Tool Call Fix (If Needed)

If you encounter tool call submission errors:

```bash
curl -sSL https://raw.githubusercontent.com/drapeaucharles/mia-backend/master/install-vllm-tool-fix.sh | bash && ./restart_miner.sh
```

## API Endpoints

### Chat & Jobs

```python
# Submit chat with tools
POST /chat
{
    "message": "What fish dishes do you have?",
    "context": {"business_name": "Restaurant"},
    "tools": [{
        "type": "function",
        "function": {
            "name": "search_menu_items",
            "description": "Search menu by ingredient",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string"}
                }
            }
        }
    }]
}

# Poll for result
GET /job/{job_id}/result
```

### Miner Management

```python
# Register miner (auto-done by installer)
POST /register_miner
{"name": "gpu-miner-hostname"}

# Get work
GET /get_work?miner_id={id}

# Submit result
POST /submit_result
{
    "miner_id": "123",
    "request_id": "job-id",
    "result": {
        "response": "...",
        "tool_call": {
            "name": "search_menu_items",
            "parameters": {"search_term": "fish"}
        }
    }
}
```

## Miner Management

### Check Status
```bash
cd /data/qwen-awq-miner
./vllm_manage.sh status
```

### View Logs
```bash
# vLLM server logs
tail -f logs/vllm.out

# Miner job logs
tail -f logs/miner_direct.log
```

### Control Commands
```bash
./vllm_manage.sh start    # Start vLLM server
./vllm_manage.sh stop     # Stop vLLM server
./vllm_manage.sh restart  # Restart vLLM server
./vllm_manage.sh test     # Test API endpoint
```

### Test Tool Calling
```bash
./test_tools.py
```

## Tool Calling Details

The system supports OpenAI-compatible tool calling:

1. **Tools are sent** in the request to MIA backend
2. **Backend forwards** tools to the miner with the job
3. **vLLM decides** whether to use tools based on the query
4. **Miner returns** tool call in structured format
5. **Backend/Frontend** executes the tool and gets results

### Example Flow

User: "What fish dishes do you have?"
↓
AI: Calls `search_menu_items` with `{"search_term": "fish"}`
↓
System: Executes tool, returns fish dishes
↓
AI: "We have Grilled Salmon and Seared Scallops..."

## Performance

| GPU | Model | Speed | VRAM |
|-----|-------|-------|------|
| RTX 4090 | Qwen2.5-7B-AWQ | 80-100 tok/s | ~6GB |
| RTX 3090 | Qwen2.5-7B-AWQ | 60-80 tok/s | ~6GB |
| RTX 3080 | Qwen2.5-7B-AWQ | 50-65 tok/s | ~6GB |
| RTX 3070 | Qwen2.5-7B-AWQ | 40-50 tok/s | ~6GB |

## Deployment

### Railway (Production)
- URL: https://mia-backend-production.up.railway.app
- Auto-deploys from GitHub master
- Includes Redis for job queue
- Environment: Python 3.9, PostgreSQL, Redis

### Local Development
```bash
# Clone repo
git clone https://github.com/drapeaucharles/mia-backend.git
cd mia-backend

# Install deps
pip install -r requirements.txt

# Run
python main.py
```

### Testing
```bash
# Test all endpoints
python test-all-endpoints.py

# Demo tool flow
./demo_tool_flow.sh

# Test tools
python test_tools.py
```

## Project Structure

```
mia-backend/
├── main.py                    # FastAPI application
├── schemas.py                 # Data models
├── db.py                      # Database setup
├── redis_queue.py            # Job queue
├── install-vllm-final.sh     # Main installer
├── install-vllm-tool-fix.sh  # Tool fix patch
├── demo_tool_flow.sh         # Tool demo
├── test_tools.py             # Tool tests
└── archived_installers/      # Old versions
```

## Troubleshooting

### Miner Not Connecting
- Check logs: `tail -f logs/miner_direct.log`
- Verify registration succeeded
- Ensure backend URL is accessible

### Tool Calls Not Working
- Verify vLLM has `--tool-call-parser hermes`
- Check tools are in OpenAI format
- Run tool fix installer if needed

### Slow Performance
- Check GPU utilization: `nvidia-smi`
- Verify xFormers backend: `grep xformers logs/vllm.out`
- Ensure AWQ model is loaded (not GPTQ)

## Environment Variables

- `MIA_BACKEND_URL` - Backend URL (default: production)
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `VLLM_ATTENTION_BACKEND` - Set to XFORMERS
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection

## Contributing

1. Fork repository
2. Create feature branch
3. Test with GPU miners
4. Submit pull request

## License

Proprietary - All rights reserved