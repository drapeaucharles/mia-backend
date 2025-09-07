"""
Backend modification to support ngrok URLs
Add this to the get_available_gpu() function in main.py
"""

# Update the heartbeat endpoint to store public_url
@app.post("/heartbeat")
async def receive_heartbeat(request: dict, db=Depends(get_db)):
    """
    Receive heartbeat from GPU miners (updated for ngrok support)
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
            "name": miner.name,
            "public_url": request.get('public_url')  # Store ngrok URL if provided
        }
        
        # Update miner last seen
        miner.last_active = datetime.utcnow()
        db.commit()
        
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update get_available_gpu to use public_url if available
def get_available_gpu() -> Optional[Dict]:
    """Get the most recently available GPU (updated for ngrok)"""
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
    
    # Use public_url if available (ngrok), otherwise construct from IP
    if info.get('public_url'):
        url = info['public_url']
    else:
        url = f"http://{info['ip']}:{info['port']}"
    
    return {
        "id": gpu_id,
        "url": url,
        "auth_key": info['auth_key'],
        "name": info.get('name', 'unknown')
    }