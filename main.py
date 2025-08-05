from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional
import uuid
import os
from dotenv import load_dotenv

from db import get_db, init_db
from schemas import (
    ChatRequest, ChatResponse, JobResult, JobResultRequest,
    MinerRegistration, MinerResponse, JobResponse
)
from queue import RedisQueue
from utils import generate_auth_key
import db as database

load_dotenv()

app = FastAPI(title="MIA Backend", version="1.0.0")

# Initialize Redis queue
redis_queue = RedisQueue()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MIA Backend"}

@app.post("/chat", response_model=ChatResponse)
async def create_chat_job(request: ChatRequest, db=Depends(get_db)):
    """
    Receive chat messages from clients and queue jobs for miners
    """
    try:
        # Generate unique IDs
        job_id = str(uuid.uuid4())
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user message in database
        chat_log = database.ChatLog(
            session_id=session_id,
            message=request.message,
            role="user",
            timestamp=datetime.utcnow()
        )
        db.add(chat_log)
        db.commit()
        
        # Create job for miners
        job = {
            "job_id": job_id,
            "prompt": request.message,
            "context": request.context or "",
            "session_id": session_id,
            "business_id": request.business_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Push job to Redis queue
        redis_queue.push_job(job)
        
        return ChatResponse(
            job_id=job_id,
            session_id=session_id,
            status="queued",
            message="Job queued successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/next", response_model=JobResponse)
async def get_next_job(miner_id: Optional[str] = None, db=Depends(get_db)):
    """
    Miners call this to fetch the next available job
    """
    try:
        # Pop next job from queue
        job = redis_queue.pop_job()
        
        if not job:
            return JobResponse(
                job_id=None,
                prompt=None,
                context=None,
                session_id=None,
                business_id=None,
                message="No jobs available"
            )
        
        # Update miner job count if miner_id provided
        if miner_id:
            miner = db.query(database.Miner).filter(
                database.Miner.id == miner_id
            ).first()
            if miner:
                miner.job_count += 1
                db.commit()
        
        return JobResponse(
            job_id=job["job_id"],
            prompt=job["prompt"],
            context=job.get("context", ""),
            session_id=job["session_id"],
            business_id=job.get("business_id"),
            message="Job retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job/result", response_model=JobResult)
async def submit_job_result(request: JobResultRequest, db=Depends(get_db)):
    """
    Miners submit their job results here
    """
    try:
        # Store AI response in chat logs
        chat_log = database.ChatLog(
            session_id=request.session_id,
            message=request.output,
            role="assistant",
            timestamp=datetime.utcnow()
        )
        db.add(chat_log)
        db.commit()
        
        # Store result in Redis for retrieval by clients
        result_data = {
            "job_id": request.job_id,
            "session_id": request.session_id,
            "output": request.output,
            "miner_id": request.miner_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        redis_queue.store_result(request.job_id, result_data)
        
        return JobResult(
            job_id=request.job_id,
            status="completed",
            message="Result stored successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_miner", response_model=MinerResponse)
async def register_miner(request: MinerRegistration, db=Depends(get_db)):
    """
    Register a new miner and assign an ID
    """
    try:
        # Check if miner already exists
        existing_miner = db.query(database.Miner).filter(
            database.Miner.name == request.name
        ).first()
        
        if existing_miner:
            return MinerResponse(
                miner_id=str(existing_miner.id),
                auth_key=existing_miner.auth_key,
                message="Miner already registered"
            )
        
        # Create new miner
        auth_key = generate_auth_key()
        miner = database.Miner(
            name=request.name,
            auth_key=auth_key,
            job_count=0
        )
        db.add(miner)
        db.commit()
        db.refresh(miner)
        
        return MinerResponse(
            miner_id=str(miner.id),
            auth_key=auth_key,
            message="Miner registered successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check Redis connection
        redis_status = redis_queue.health_check()
        
        # Check database connection
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = True
        
        return {
            "status": "healthy",
            "redis": redis_status,
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)