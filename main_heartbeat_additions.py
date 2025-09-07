"""
Additions needed for main.py to support heartbeat architecture
DO NOT REPLACE main.py - add these endpoints alongside existing ones
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import requests
import asyncio

# Global storage for available GPUs
available_gpus: Dict[str, Dict] = {}

# Cleanup old heartbeats every 10 seconds
async def cleanup_stale_gpus():
    while True:
        try:
            now = datetime.utcnow()
            stale_threshold = now - timedelta(seconds=5)  # Consider stale after 5s
            
            stale_ids = [
                gpu_id for gpu_id, info in available_gpus.items()
                if info['last_heartbeat'] < stale_threshold
            ]
            
            for gpu_id in stale_ids:
                logger.info(f"Removing stale GPU: {gpu_id}")
                del available_gpus[gpu_id]
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
        await asyncio.sleep(10)

@app.on_event("startup")
async def startup_heartbeat_handler():
    """Start background tasks for heartbeat system"""
    asyncio.create_task(cleanup_stale_gpus())

@app.post("/heartbeat")
async def receive_heartbeat(request: dict, db=Depends(get_db)):
    """
    Receive heartbeat from GPU miners
    Expected format: {"miner_id": str, "status": str, "timestamp": str, "port": int}
    """
    try:
        miner_id = request.get('miner_id')
        
        # Verify miner exists
        miner = db.query(database.Miner).filter(
            database.Miner.id == miner_id
        ).first()
        
        if not miner:
            raise HTTPException(status_code=404, detail="Miner not found")
        
        # Update available GPUs tracking
        available_gpus[miner_id] = {
            "last_heartbeat": datetime.utcnow(),
            "status": request.get('status', 'available'),
            "ip": miner.ip_address,
            "port": request.get('port', 5000),
            "auth_key": miner.auth_key
        }
        
        # Update miner last seen
        miner.last_seen = datetime.utcnow()
        db.commit()
        
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_available_gpu() -> Optional[Dict]:
    """Get the most recently available GPU"""
    if not available_gpus:
        return None
        
    # Get GPU with most recent heartbeat that's available
    available = [
        (gpu_id, info) for gpu_id, info in available_gpus.items()
        if info['status'] == 'available'
    ]
    
    if not available:
        return None
        
    # Sort by most recent heartbeat
    available.sort(key=lambda x: x[1]['last_heartbeat'], reverse=True)
    gpu_id, info = available[0]
    
    return {
        "id": gpu_id,
        "url": f"http://{info['ip']}:{info['port']}",
        "auth_key": info['auth_key']
    }

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
            logger.info(f"Pushing job directly to GPU {gpu['id']}")
            
            # Mark GPU as busy
            if gpu['id'] in available_gpus:
                available_gpus[gpu['id']]['status'] = 'busy'
            
            # Push work to GPU
            try:
                response = requests.post(
                    f"{gpu['url']}/process",
                    json={
                        "request_id": str(uuid.uuid4()),
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
                        context=request.context,
                        status="completed",
                        response=result.get('response', ''),
                        tokens_generated=result.get('tokens_generated', 0),
                        completed_at=datetime.utcnow(),
                        assigned_to=gpu['id']
                    )
                    db.add(job)
                    db.commit()
                    
                    return ChatResponse(
                        job_id=str(job.id),
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
        logger.info("No available GPU, using queue")
        return await chat(request, db)  # Call existing chat endpoint
        
    except Exception as e:
        logger.error(f"Direct chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add to existing metrics endpoint
@app.get("/metrics/gpus")
async def get_gpu_metrics():
    """Get current GPU availability metrics"""
    return {
        "total_gpus": len(available_gpus),
        "available_gpus": len([g for g in available_gpus.values() if g['status'] == 'available']),
        "busy_gpus": len([g for g in available_gpus.values() if g['status'] == 'busy']),
        "gpus": [
            {
                "id": gpu_id,
                "status": info['status'],
                "last_heartbeat": info['last_heartbeat'].isoformat(),
                "seconds_since_heartbeat": (datetime.utcnow() - info['last_heartbeat']).total_seconds()
            }
            for gpu_id, info in available_gpus.items()
        ]
    }