from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime
from typing import Optional, List
import uuid
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import func

from db import get_db, init_db
from schemas import (
    ChatRequest, ChatResponse, JobResult, JobResultRequest,
    MinerRegistration, MinerResponse, JobResponse,
    IdleJobRequest, IdleJobResponse, IdleJobNextResponse,
    IdleJobResultRequest, IdleJobResultResponse,
    BuybackResponse, SystemMetricsResponse,
    GolemJobRequest, GolemJobResponse,
    MinerStatusRequest, MinerStatusResponse
)
from redis_queue import RedisQueue
from utils import generate_auth_key
import db as database
from runpod_manager import RunPodManager
from buyback import BuybackEngine
import asyncio

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(title="MIA Backend", version="1.0.0")

# Initialize Redis queue
redis_queue = RedisQueue()

# Initialize RunPod manager and buyback engine
runpod_manager = RunPodManager()
buyback_engine = BuybackEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        print("Database initialized successfully")
        
        # Run migrations
        from run_migrations import run_all_migrations
        run_all_migrations()
        
    except Exception as e:
        print(f"Database initialization error: {e}")
        # Continue running even if DB init fails for healthcheck

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

@app.post("/idle-job", response_model=IdleJobResponse)
async def create_idle_job(request: IdleJobRequest, db=Depends(get_db)):
    """
    External clients submit AI workloads when main queue is idle
    """
    try:
        # Validate API key (simple validation, enhance in production)
        if not request.api_key or len(request.api_key) < 16:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Estimate revenue
        revenue_estimate = runpod_manager.estimate_job_revenue(request.max_tokens)
        
        # Create idle job
        idle_job = database.IdleJob(
            prompt=request.prompt,
            submitted_by=request.api_key[:8] + "****",  # Mask API key
            status="pending"
        )
        db.add(idle_job)
        db.commit()
        db.refresh(idle_job)
        
        # Add to idle jobs queue
        redis_queue.push_idle_job({
            "job_id": idle_job.id,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens
        })
        
        return IdleJobResponse(
            job_id=idle_job.id,
            status="queued",
            message="Idle job queued successfully",
            estimated_revenue_usd=revenue_estimate["revenue_usd"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/idle-job/next", response_model=IdleJobNextResponse)
async def get_next_idle_job(db=Depends(get_db)):
    """
    Get next idle job when main queue is empty
    """
    try:
        # First check if main queue is empty
        main_queue_length = redis_queue.get_queue_length()
        
        if main_queue_length > 0:
            return IdleJobNextResponse(
                job_id=None,
                prompt=None,
                max_tokens=None,
                message="Main queue not empty, process MIA jobs first"
            )
        
        # Get next idle job
        idle_job_data = redis_queue.pop_idle_job()
        
        if not idle_job_data:
            return IdleJobNextResponse(
                job_id=None,
                prompt=None,
                max_tokens=None,
                message="No idle jobs available"
            )
        
        # Update job status
        idle_job = db.query(database.IdleJob).filter(
            database.IdleJob.id == idle_job_data["job_id"]
        ).first()
        
        if idle_job:
            idle_job.status = "processing"
            db.commit()
        
        return IdleJobNextResponse(
            job_id=idle_job_data["job_id"],
            prompt=idle_job_data["prompt"],
            max_tokens=idle_job_data.get("max_tokens", 500),
            message="Idle job retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/idle-job/result", response_model=IdleJobResultResponse)
async def submit_idle_job_result(request: IdleJobResultRequest, db=Depends(get_db)):
    """
    Submit results from processed idle jobs
    """
    try:
        # Update idle job
        idle_job = db.query(database.IdleJob).filter(
            database.IdleJob.id == request.job_id
        ).first()
        
        if not idle_job:
            raise HTTPException(status_code=404, detail="Idle job not found")
        
        idle_job.status = "completed" if not request.error_message else "failed"
        idle_job.output_tokens = request.output_tokens
        idle_job.usd_earned = request.usd_earned
        idle_job.result = request.output
        idle_job.runpod_job_id = request.runpod_job_id
        idle_job.error_message = request.error_message
        idle_job.completed_at = datetime.utcnow()
        
        # Update system metrics
        if request.usd_earned > 0:
            # Update RunPod income
            income_metric = db.query(database.SystemMetrics).filter(
                database.SystemMetrics.metric_name == "runpod_income_usd"
            ).first()
            
            if income_metric:
                income_metric.value += request.usd_earned
            else:
                income_metric = database.SystemMetrics(
                    metric_name="runpod_income_usd",
                    value=request.usd_earned
                )
                db.add(income_metric)
            
            # Update job count
            jobs_metric = db.query(database.SystemMetrics).filter(
                database.SystemMetrics.metric_name == "total_idle_jobs_processed"
            ).first()
            
            if jobs_metric:
                jobs_metric.value += 1
            else:
                jobs_metric = database.SystemMetrics(
                    metric_name="total_idle_jobs_processed",
                    value=1
                )
                db.add(jobs_metric)
        
        db.commit()
        
        # Get updated income balance
        income_metric = db.query(database.SystemMetrics).filter(
            database.SystemMetrics.metric_name == "runpod_income_usd"
        ).first()
        
        return IdleJobResultResponse(
            status="completed",
            message="Idle job result stored successfully",
            total_income_usd=income_metric.value if income_metric else 0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger-buyback", response_model=BuybackResponse)
async def trigger_buyback(manual: bool = False, db=Depends(get_db)):
    """
    Trigger token buyback and burn (manual or automatic)
    """
    try:
        result = buyback_engine.check_and_execute_buyback(db)
        
        if result["triggered"]:
            return BuybackResponse(
                status="success",
                message="Buyback executed successfully",
                amount_usd=result["amount_usd"],
                tokens_burned=result["tokens_burned"],
                transaction_hash=result["burn_tx_hash"]
            )
        else:
            return BuybackResponse(
                status="not_triggered",
                message=result["reason"],
                amount_usd=None,
                tokens_burned=None,
                transaction_hash=None
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(db=Depends(get_db)):
    """
    Get system metrics including RunPod income and buyback stats
    """
    try:
        metrics = buyback_engine.get_buyback_history(db)
        
        return SystemMetricsResponse(
            runpod_income_usd=metrics.get("runpod_income_usd", 0.0),
            total_idle_jobs_processed=int(metrics.get("total_idle_jobs_processed", 0)),
            total_buyback_usd=metrics.get("total_buyback_usd", 0.0),
            last_buyback_timestamp=datetime.fromisoformat(metrics["last_buyback_timestamp"]) 
                if metrics.get("last_buyback_timestamp") else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/idle-job/process/{job_id}")
async def process_idle_job_with_runpod(job_id: int, db=Depends(get_db)):
    """
    Process an idle job using RunPod (for testing/manual processing)
    """
    try:
        # Get idle job
        idle_job = db.query(database.IdleJob).filter(
            database.IdleJob.id == job_id
        ).first()
        
        if not idle_job:
            raise HTTPException(status_code=404, detail="Idle job not found")
        
        if idle_job.status != "pending":
            raise HTTPException(status_code=400, detail="Job already processed")
        
        # Update status
        idle_job.status = "processing"
        db.commit()
        
        # Process with RunPod
        result = await runpod_manager.process_idle_job(idle_job.prompt)
        
        if result["success"]:
            # Update job with results
            idle_job.status = "completed"
            idle_job.output_tokens = result["output_tokens"]
            idle_job.usd_earned = result["revenue_usd"]
            idle_job.result = result["output"]
            idle_job.runpod_job_id = result["job_id"]
            idle_job.completed_at = datetime.utcnow()
            
            # Update income metrics
            income_metric = db.query(database.SystemMetrics).filter(
                database.SystemMetrics.metric_name == "runpod_income_usd"
            ).first()
            
            if income_metric:
                income_metric.value += result["revenue_usd"]
            else:
                income_metric = database.SystemMetrics(
                    metric_name="runpod_income_usd",
                    value=result["revenue_usd"]
                )
                db.add(income_metric)
            
            db.commit()
            
            return {
                "status": "success",
                "output": result["output"],
                "tokens": result["output_tokens"],
                "revenue_usd": result["revenue_usd"]
            }
        else:
            idle_job.status = "failed"
            idle_job.error_message = result.get("error", "Unknown error")
            db.commit()
            
            raise HTTPException(status_code=500, detail=result.get("error"))
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check Redis connection
    try:
        redis_status = redis_queue.health_check()
        health_status["redis"] = redis_status
    except Exception as e:
        health_status["redis"] = False
        health_status["redis_error"] = str(e)
    
    # Check database connection (optional - don't fail if DB is down)
    try:
        from sqlalchemy import text, func
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["database"] = True
    except Exception as e:
        health_status["database"] = False
        health_status["database_error"] = str(e)
    
    # Check RunPod connection (optional)
    try:
        runpod_status = await runpod_manager.health_check()
        health_status["runpod"] = runpod_status
    except Exception as e:
        health_status["runpod"] = False
        health_status["runpod_error"] = str(e)
    
    # Return 200 OK even if some services are down
    # This allows the app to start even without all dependencies
    return health_status

@app.post("/report_golem_job", response_model=GolemJobResponse)
async def report_golem_job(request: GolemJobRequest, db=Depends(get_db)):
    """
    Receive fallback compute reports from miners (Golem jobs)
    """
    try:
        # Validate timestamp (must be within 2 minutes)
        time_diff = abs((datetime.utcnow() - request.timestamp).total_seconds())
        if time_diff > 120:  # 2 minutes
            raise HTTPException(
                status_code=400,
                detail="Timestamp must be within 2 minutes of current time"
            )
        
        # Validate GLM estimate is reasonable (max 1 GLM per hour)
        max_glm_per_hour = 1.0
        max_expected_glm = (request.duration_sec / 3600) * max_glm_per_hour
        if request.estimated_glm > max_expected_glm:
            raise HTTPException(
                status_code=400,
                detail=f"GLM estimate too high for duration. Max expected: {max_expected_glm:.4f}"
            )
        
        # Check if miner exists
        miner_exists = db.query(database.Miner).filter(
            database.Miner.name == request.miner_name
        ).first()
        
        if not miner_exists:
            # Auto-register unknown miners for fallback jobs
            auth_key = generate_auth_key()
            new_miner = database.Miner(
                name=request.miner_name,
                auth_key=auth_key,
                job_count=0
            )
            db.add(new_miner)
            db.commit()
        
        # Store Golem job report
        golem_job = database.GolemJob(
            miner_name=request.miner_name,
            duration_sec=request.duration_sec,
            estimated_glm=request.estimated_glm,
            timestamp=request.timestamp
        )
        db.add(golem_job)
        db.commit()
        
        # Calculate total GLM earned by this miner
        total_glm = db.query(func.sum(database.GolemJob.estimated_glm)).filter(
            database.GolemJob.miner_name == request.miner_name
        ).scalar() or 0.0
        
        logger.info(f"Golem job reported: {request.miner_name} - {request.duration_sec}s - {request.estimated_glm} GLM")
        
        return GolemJobResponse(
            status="success",
            message="Fallback compute report stored successfully",
            total_glm=float(total_glm)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reporting Golem job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/miner/{miner_id}/status", response_model=MinerStatusResponse)
async def update_miner_status(miner_id: int, request: MinerStatusRequest, db=Depends(get_db)):
    """
    Update miner status (idle/busy/offline)
    """
    try:
        miner = db.query(database.Miner).filter(
            database.Miner.id == miner_id
        ).first()
        
        if not miner:
            raise HTTPException(status_code=404, detail="Miner not found")
        
        miner.status = request.status
        miner.last_active = datetime.utcnow()
        db.commit()
        
        return MinerStatusResponse(
            status="success",
            message=f"Miner status updated to {request.status}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/miners")
async def list_miners(db=Depends(get_db)):
    """
    List all registered miners with their status
    """
    try:
        miners = db.query(database.Miner).order_by(
            database.Miner.last_active.desc()
        ).all()
        return miners
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_work")
async def get_work_for_miner(miner_id: int, db=Depends(get_db)):
    """
    Get work for a specific miner
    """
    try:
        # Get next job from Redis queue
        job = redis_queue.get_next_job()
        
        if job:
            # Mark miner as active
            miner = db.query(database.Miner).filter(database.Miner.id == miner_id).first()
            if miner:
                miner.status = "active"
                miner.last_active = datetime.utcnow()
                db.commit()
            
            return {
                "request_id": job.get("job_id"),
                "prompt": job.get("prompt"),
                "max_tokens": job.get("max_tokens", 500),
                "temperature": job.get("temperature", 0.7)
            }
        
        # No work available
        return {}
        
    except Exception as e:
        logger.error(f"Error getting work: {e}")
        return {}

@app.post("/submit_result")
async def submit_miner_result(
    miner_id: int,
    request_id: str,
    result: str,
    tokens_generated: int = 0,
    processing_time: float = 0,
    db=Depends(get_db)
):
    """
    Submit result from miner
    """
    try:
        # Store result in Redis
        redis_queue.set_result(request_id, {
            "response": result,
            "tokens_generated": tokens_generated,
            "processing_time": processing_time,
            "miner_id": miner_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update miner stats
        miner = db.query(database.Miner).filter(database.Miner.id == miner_id).first()
        if miner:
            miner.jobs_completed = (miner.jobs_completed or 0) + 1
            miner.status = "idle"
            miner.last_active = datetime.utcnow()
            db.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error submitting result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}/result")
async def get_job_result(job_id: str, db=Depends(get_db)):
    """
    Get the result of a chat job
    """
    try:
        # Check Redis for result
        result = redis_queue.get_result(job_id)
        if result:
            return {"status": "completed", "result": result}
        
        # Check if job exists
        job = redis_queue.get_job(job_id)
        if job:
            return {"status": "processing", "result": None}
        
        return {"status": "not_found", "result": None}
        
    except Exception as e:
        logger.error(f"Error getting job result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-ui", response_class=HTMLResponse)
async def chat_interface():
    """
    Serve a simple chat interface for testing the MIA network
    """
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>MIA Network Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 600px;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #2563eb;
            color: white;
            padding: 20px;
            border-radius: 12px 12px 0 0;
            text-align: center;
        }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .miners-status {
            padding: 10px 20px;
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
            font-size: 14px;
            color: #6b7280;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .message {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            background: #2563eb;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .ai-message {
            background: #f3f4f6;
            color: #1f2937;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .typing {
            display: none;
            align-self: flex-start;
            padding: 12px 16px;
            background: #f3f4f6;
            border-radius: 12px;
            border-bottom-left-radius: 4px;
        }
        .typing span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #9ca3af;
            margin: 0 2px;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .typing span:nth-child(1) { animation-delay: -0.32s; }
        .typing span:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .input-container {
            padding: 20px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            gap: 10px;
        }
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #d1d5db;
            border-radius: 24px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        #messageInput:focus {
            border-color: #2563eb;
        }
        #sendButton {
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 50%;
            width: 48px;
            height: 48px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        #sendButton:hover:not(:disabled) {
            background: #1d4ed8;
        }
        #sendButton:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }
        .error {
            background: #fee;
            color: #dc2626;
            padding: 10px 15px;
            border-radius: 8px;
            margin: 10px 20px;
            font-size: 14px;
        }
        .status {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status.online { background: #10b981; }
        .status.offline { background: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 MIA Network Chat</h1>
            <p>Chat with AI models running on the decentralized GPU network</p>
        </div>
        <div class="miners-status" id="minersStatus">
            <span class="status offline"></span>
            Checking network status...
        </div>
        <div class="chat-container" id="chatContainer">
            <div class="ai-message message">
                👋 Hello! I'm connected to the MIA GPU network. Ask me anything!
            </div>
        </div>
        <div class="typing" id="typing">
            <span></span><span></span><span></span>
        </div>
        <div class="input-container">
            <input 
                type="text" 
                id="messageInput" 
                placeholder="Type a message..." 
                autofocus
            />
            <button id="sendButton">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
            </button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typing = document.getElementById('typing');
        const minersStatus = document.getElementById('minersStatus');
        
        let sessionId = localStorage.getItem('mia-session-id') || generateId();
        localStorage.setItem('mia-session-id', sessionId);
        
        function generateId() {
            return 'xxxx-xxxx-xxxx'.replace(/[x]/g, () => 
                (Math.random() * 16 | 0).toString(16)
            );
        }
        
        // Check miners status
        async function checkMiners() {
            try {
                const response = await fetch('/miners');
                const miners = await response.json();
                const activeMiners = miners.filter(m => 
                    m.status === 'active' || m.status === 'idle'
                ).length;
                
                if (activeMiners > 0) {
                    minersStatus.innerHTML = `
                        <span class="status online"></span>
                        ${activeMiners} miner${activeMiners > 1 ? 's' : ''} online
                    `;
                } else {
                    minersStatus.innerHTML = `
                        <span class="status offline"></span>
                        No miners available
                    `;
                }
            } catch (error) {
                console.error('Failed to check miners:', error);
            }
        }
        
        checkMiners();
        setInterval(checkMiners, 30000); // Check every 30 seconds
        
        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
            messageDiv.textContent = text;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function showTyping() {
            typing.style.display = 'block';
            chatContainer.appendChild(typing);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function hideTyping() {
            typing.style.display = 'none';
        }
        
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            chatContainer.appendChild(errorDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            setTimeout(() => errorDiv.remove(), 5000);
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            showTyping();
            
            try {
                // Send to chat endpoint
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        session_id: sessionId,
                        business_id: 1
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Poll for result
                    const result = await pollForResult(data.job_id);
                    hideTyping();
                    
                    if (result) {
                        addMessage(result);
                    } else {
                        showError('Request timed out. No miners available or network is busy.');
                    }
                } else {
                    hideTyping();
                    showError(data.detail || 'Failed to send message');
                }
            } catch (error) {
                hideTyping();
                showError('Network error. Please try again.');
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        
        async function pollForResult(jobId, maxAttempts = 30) {
            for (let i = 0; i < maxAttempts; i++) {
                try {
                    const response = await fetch(`/job/${jobId}/result`);
                    const data = await response.json();
                    
                    if (response.ok && data.result) {
                        // Handle different response formats
                        if (typeof data.result === 'string') {
                            return data.result;
                        } else if (data.result.response) {
                            return data.result.response;
                        } else if (data.result.text) {
                            return data.result.text;
                        } else {
                            return JSON.stringify(data.result);
                        }
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
                
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
            return null;
        }
        
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
""")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)