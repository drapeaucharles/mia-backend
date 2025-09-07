"""
Backend updates to support heartbeat miners alongside polling miners
Add this code to main.py WITHOUT removing existing endpoints
"""

# Add these imports at the top of main.py
from typing import Dict, Optional
import asyncio
from datetime import datetime, timedelta

# Add this global variable after the existing globals
available_gpus: Dict[str, Dict] = {}

# Add this startup task to clean stale GPUs
@app.on_event("startup")
async def startup_heartbeat_cleanup():
    """Start background task to clean stale GPU entries"""
    async def cleanup_stale_gpus():
        while True:
            try:
                now = datetime.utcnow()
                stale_threshold = now - timedelta(seconds=5)
                
                stale_ids = [
                    gpu_id for gpu_id, info in available_gpus.items()
                    if info.get('last_heartbeat', now) < stale_threshold
                ]
                
                for gpu_id in stale_ids:
                    logger.info(f"Removing stale GPU: {gpu_id}")
                    del available_gpus[gpu_id]
                    
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                
            await asyncio.sleep(10)
    
    asyncio.create_task(cleanup_stale_gpus())

# Add this new /register endpoint (keep /register_miner for old miners)
@app.post("/register")
async def register_heartbeat_miner(request: dict, db=Depends(get_db)):
    """
    Register endpoint for heartbeat miners (new architecture)
    Accepts different format than /register_miner
    """
    try:
        # Extract data from heartbeat miner format
        ip_address = request.get('ip_address', 'unknown')
        gpu_info = request.get('gpu_info', {})
        hostname = request.get('hostname', 'unknown')
        backend_type = request.get('backend_type', 'vllm-heartbeat')
        
        # Create name from hostname for compatibility
        name = f"{hostname}_{backend_type}"
        
        # Check if miner already exists by IP or hostname
        existing_miner = db.query(database.Miner).filter(
            (database.Miner.ip_address == ip_address) | 
            (database.Miner.name == name)
        ).first()
        
        if existing_miner:
            # Update last seen
            existing_miner.last_seen = datetime.utcnow()
            existing_miner.gpu_info = json.dumps(gpu_info) if gpu_info else None
            db.commit()
            
            return {
                "miner_id": str(existing_miner.id),
                "auth_key": existing_miner.auth_key,
                "message": "Miner already registered"
            }
        
        # Create new miner
        auth_key = generate_auth_key()
        miner = database.Miner(
            name=name,
            auth_key=auth_key,
            ip_address=ip_address,
            job_count=0,
            gpu_info=json.dumps(gpu_info) if gpu_info else None
        )
        db.add(miner)
        db.commit()
        db.refresh(miner)
        
        logger.info(f"Registered new heartbeat miner: {miner.id} from {ip_address}")
        
        return {
            "miner_id": str(miner.id),
            "auth_key": auth_key,
            "message": "Miner registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add heartbeat endpoint
@app.post("/heartbeat")
async def receive_heartbeat(request: dict, db=Depends(get_db)):
    """
    Receive heartbeat from GPU miners
    """
    try:
        miner_id = request.get('miner_id')
        
        if not miner_id:
            raise HTTPException(status_code=400, detail="miner_id required")
        
        # Verify miner exists
        miner = db.query(database.Miner).filter(
            database.Miner.id == int(miner_id)
        ).first()
        
        if not miner:
            raise HTTPException(status_code=404, detail="Miner not found")
        
        # Update available GPUs tracking
        available_gpus[str(miner_id)] = {
            "last_heartbeat": datetime.utcnow(),
            "status": request.get('status', 'available'),
            "ip": miner.ip_address,
            "port": request.get('port', 5000),
            "auth_key": miner.auth_key,
            "name": miner.name
        }
        
        # Update miner last seen
        miner.last_seen = datetime.utcnow()
        db.commit()
        
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get available GPU
def get_available_gpu() -> Optional[Dict]:
    """Get the most recently available GPU"""
    if not available_gpus:
        return None
        
    # Get GPU with most recent heartbeat that's available
    available = [
        (gpu_id, info) for gpu_id, info in available_gpus.items()
        if info.get('status') == 'available'
    ]
    
    if not available:
        return None
        
    # Sort by most recent heartbeat
    available.sort(key=lambda x: x[1].get('last_heartbeat', datetime.min), reverse=True)
    gpu_id, info = available[0]
    
    return {
        "id": gpu_id,
        "url": f"http://{info['ip']}:{info['port']}",
        "auth_key": info['auth_key'],
        "name": info.get('name', 'unknown')
    }

# Add new /chat/direct endpoint for push architecture
@app.post("/chat/direct", response_model=ChatResponse)
async def chat_with_direct_gpu(request: ChatRequest, db=Depends(get_db)):
    """
    New endpoint that pushes work directly to available GPUs
    Falls back to queue if no GPU available
    """
    try:
        # Try to get available GPU
        gpu = get_available_gpu()
        
        if gpu:
            logger.info(f"Pushing job directly to GPU {gpu['id']} ({gpu['name']})")
            
            # Mark GPU as busy
            if gpu['id'] in available_gpus:
                available_gpus[gpu['id']]['status'] = 'busy'
            
            # Push work to GPU
            try:
                job_id = str(uuid.uuid4())
                
                response = requests.post(
                    f"{gpu['url']}/process",
                    json={
                        "request_id": job_id,
                        "prompt": request.message,
                        "messages": [{"role": "user", "content": request.message}],
                        "context": request.context,
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    headers={"Authorization": f"Bearer {gpu['auth_key']}"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Mark GPU as available again
                    if gpu['id'] in available_gpus:
                        available_gpus[gpu['id']]['status'] = 'available'
                    
                    # Save to database
                    job = database.Job(
                        prompt=request.message,
                        context=json.dumps(request.context) if request.context else None,
                        status="completed",
                        response=result.get('response', ''),
                        tokens_generated=result.get('tokens_generated', 0),
                        completed_at=datetime.utcnow(),
                        assigned_to=gpu['id'],
                        processing_time=result.get('processing_time', 0)
                    )
                    db.add(job)
                    db.commit()
                    
                    return ChatResponse(
                        job_id=job_id,
                        status="completed",
                        response=result.get('response', ''),
                        tokens_generated=result.get('tokens_generated', 0)
                    )
                else:
                    logger.error(f"GPU returned error: {response.status_code}")
                    # Mark GPU as available
                    if gpu['id'] in available_gpus:
                        available_gpus[gpu['id']]['status'] = 'available'
                    # Fall through to queue
                    
            except Exception as e:
                logger.error(f"Error pushing to GPU: {e}")
                # Mark GPU as available
                if gpu['id'] in available_gpus:
                    available_gpus[gpu['id']]['status'] = 'available'
                # Fall through to queue
        
        # Fallback to queue-based system
        logger.info("No available GPU for direct push, using queue")
        return await chat(request, db)  # Call existing chat endpoint
        
    except Exception as e:
        logger.error(f"Direct chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add metrics endpoint for monitoring
@app.get("/metrics/gpus")
async def get_gpu_metrics():
    """Get current GPU availability metrics"""
    now = datetime.utcnow()
    return {
        "total_gpus": len(available_gpus),
        "available_gpus": len([g for g in available_gpus.values() if g.get('status') == 'available']),
        "busy_gpus": len([g for g in available_gpus.values() if g.get('status') == 'busy']),
        "gpus": [
            {
                "id": gpu_id,
                "name": info.get('name', 'unknown'),
                "status": info.get('status', 'unknown'),
                "last_heartbeat": info.get('last_heartbeat', now).isoformat(),
                "seconds_since_heartbeat": (now - info.get('last_heartbeat', now)).total_seconds()
            }
            for gpu_id, info in available_gpus.items()
        ]
    }

# Add note at the end of the file
"""
NOTES:
1. Old polling miners continue to use /register_miner and /get_work
2. New heartbeat miners use /register and /heartbeat
3. Clients can use /chat (queue) or /chat/direct (push)
4. System automatically falls back to queue if no GPUs available
5. Monitor GPU availability at /metrics/gpus
"""