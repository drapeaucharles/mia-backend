# New Architecture Changes

## Overview
Moving from a polling-based GPU architecture to a push-based model with heartbeats for better performance.

## Current Architecture (BEFORE)
- **Polling Model**: GPUs poll backend every 2 seconds asking for work
- **Average Latency**: 1 second per request (0-2s depending on timing)
- **Problem**: With 2-stage AI processing (classification + response), this adds 2s average delay

## New Architecture (AFTER)
- **Push Model**: Backend pushes work directly to available GPUs
- **Heartbeat System**: GPUs send "available" status every 1 second
- **Latency**: Near-instant work assignment

## Changes Made

### 1. GPU/Miner Side Changes
File: `install-vllm-heartbeat.sh` (new, parallel to `install-vllm-tool-fix.sh`)

**Key Changes:**
- Replace polling loop with heartbeat sender
- Add HTTP endpoint to receive work pushed from backend
- Maintain "available/busy" status

### 2. Backend Side Changes
File: `main.py` (changes needed)

**New Endpoints:**
```python
@app.post("/heartbeat")
async def receive_heartbeat(request: HeartbeatRequest):
    # Track GPU availability
    available_gpus[request.miner_id] = {
        "last_heartbeat": datetime.utcnow(),
        "status": request.status,
        "url": f"http://{request.ip}:{request.port}"
    }

@app.post("/chat")
async def chat_with_push(request: ChatRequest):
    # Find available GPU
    gpu = get_available_gpu()
    if gpu:
        # Push work directly to GPU
        response = requests.post(
            f"{gpu['url']}/process",
            json=request.dict(),
            headers={"Authorization": f"Bearer {gpu['key']}"}
        )
        return response.json()
    else:
        # Fallback to queue if no GPU available
        return queue_job(request)
```

**Key Changes:**
- Track available GPUs from heartbeats
- Push work to available GPUs instead of queuing
- Keep polling as fallback for compatibility

### 3. Restaurant Chat Integration
File: `mia_chat_service_full_menu_with_tools_fixed.py`

**Key Changes:**
- Add 2-stage processing: classification first, then response
- Reduce context size based on classification
- Direct GPU calls instead of queue+poll

## Rollback Instructions
If issues occur:
1. Stop new miners: `kill $(cat /data/qwen-awq-miner/miner.pid)`
2. Restore old installer: Use `install-vllm-tool-fix.sh`
3. Restart backend with old endpoints
4. Revert restaurant chat service changes

## Testing Plan
1. Keep old system running
2. Deploy new miner on separate test GPU
3. Test latency improvements
4. Gradual rollout if successful

## Implementation Status
- ✅ Created `install-vllm-heartbeat.sh` (parallel to old installer)
- ✅ Implemented heartbeat miner with Flask endpoint
- ✅ Created `main_heartbeat_additions.py` with backend code needed
- ✅ Created `main_heartbeat_update.py` with improved backend code
- ✅ Created `add_miner_fields.py` for database migration
- ⏳ Backend deployment (add code from main_heartbeat_update.py to main.py)
- ⏳ Restaurant chat integration for 2-stage AI
- ⏳ Testing and validation

## Files Created
1. **`install-vllm-heartbeat.sh`** - New installer for heartbeat miner
2. **`main_heartbeat_additions.py`** - Backend code to add to main.py
3. **`newarchitecture.md`** - This documentation

## Deployment Steps

### 1. Update Backend (Railway)
```bash
# First, add missing database fields
python add_miner_fields.py

# Then add the code from main_heartbeat_update.py to main.py
# Copy everything between the triple quotes
```

### 2. Test Heartbeat Miner
```bash
# On GPU/VPS:
cd /data/qwen-awq-miner && ./start_heartbeat_miner.sh

# Check logs:
tail -f /data/qwen-awq-miner/heartbeat_miner.out
```

### 3. Monitor GPU Availability
```bash
# Check registered GPUs:
curl https://mia-backend-production.up.railway.app/metrics/gpus
```

### 4. Test Direct Push
```bash
# Test the new push endpoint:
curl -X POST https://mia-backend-production.up.railway.app/chat/direct \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "context": {}}'
```

## Endpoints Overview
- **Old (Polling)**:
  - `/register_miner` - Register polling miners
  - `/get_work` - Miners poll for work
  - `/chat` - Queue-based chat

- **New (Heartbeat)**:
  - `/register` - Register heartbeat miners
  - `/heartbeat` - Receive GPU availability
  - `/chat/direct` - Push work directly to GPU
  - `/metrics/gpus` - Monitor GPU status

## Performance Expectations
- **Old System**: 2s average delay (polling every 2s)
- **New System**: <100ms delay (direct push)
- **2-Stage AI Classification**: 
  - Old: 4s total (2s + 2s polling delays)
  - New: <1s total (100ms + 100ms push delays)

## Important Notes
- Old system remains fully functional
- New endpoints are added alongside old ones
- Graceful fallback to queue if no GPUs available
- Monitor both systems during transition