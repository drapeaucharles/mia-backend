from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[dict] = Field(None, description="Additional context for the AI")
    business_id: Optional[int] = Field(None, description="Business ID for multi-tenant support")
    tools: Optional[list] = Field(None, description="Available tools for the AI to use")
    tool_choice: Optional[str] = Field("auto", description="Tool choice strategy: auto, none, or specific tool name")

class ChatResponse(BaseModel):
    """Response model for /chat endpoint"""
    job_id: str = Field(..., description="Unique job identifier")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")

class JobResponse(BaseModel):
    """Response model for /job/next endpoint"""
    job_id: Optional[str] = Field(None, description="Job identifier")
    prompt: Optional[str] = Field(None, description="The prompt to process")
    context: Optional[str] = Field(None, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session identifier")
    business_id: Optional[int] = Field(None, description="Business identifier")
    message: str = Field(..., description="Status message")

class JobResultRequest(BaseModel):
    """Request model for /job/result endpoint"""
    job_id: str = Field(..., description="Job identifier")
    session_id: str = Field(..., description="Session identifier")
    output: str = Field(..., description="AI generated response")
    miner_id: Optional[str] = Field(None, description="Miner identifier")

class JobResult(BaseModel):
    """Response model for /job/result endpoint"""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Result status")
    message: str = Field(..., description="Status message")

class MinerRegistration(BaseModel):
    """Request model for /register_miner endpoint"""
    name: str = Field(..., description="Miner name", min_length=3, max_length=255)

class MinerResponse(BaseModel):
    """Response model for /register_miner endpoint"""
    miner_id: str = Field(..., description="Assigned miner ID")
    auth_key: str = Field(..., description="Authentication key for the miner")
    message: str = Field(..., description="Registration status message")

class BusinessModel(BaseModel):
    """Business model for API responses"""
    id: int
    name: str
    contact_email: Optional[str]
    contact_phone: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class ChatLogModel(BaseModel):
    """Chat log model for API responses"""
    id: int
    session_id: str
    message: str
    role: str
    timestamp: datetime
    business_id: Optional[int]
    
    class Config:
        from_attributes = True

class MinerModel(BaseModel):
    """Miner model for API responses"""
    id: int
    name: str
    job_count: int
    created_at: datetime
    last_active: datetime
    
    class Config:
        from_attributes = True

class IdleJobRequest(BaseModel):
    """Request model for /idle-job endpoint"""
    prompt: str = Field(..., description="The prompt to process")
    api_key: str = Field(..., description="API key for authentication")
    max_tokens: Optional[int] = Field(500, description="Maximum tokens to generate")

class IdleJobResponse(BaseModel):
    """Response model for /idle-job endpoint"""
    job_id: int = Field(..., description="Idle job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    estimated_revenue_usd: Optional[float] = Field(None, description="Estimated revenue in USD")

class IdleJobNextResponse(BaseModel):
    """Response model for /idle-job/next endpoint"""
    job_id: Optional[int] = Field(None, description="Idle job ID")
    prompt: Optional[str] = Field(None, description="The prompt to process")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    message: str = Field(..., description="Status message")

class IdleJobResultRequest(BaseModel):
    """Request model for /idle-job/result endpoint"""
    job_id: int = Field(..., description="Idle job ID")
    output: str = Field(..., description="Generated output")
    output_tokens: int = Field(..., description="Number of tokens generated")
    usd_earned: float = Field(..., description="USD earned from this job")
    runpod_job_id: Optional[str] = Field(None, description="RunPod job ID")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class IdleJobResultResponse(BaseModel):
    """Response model for /idle-job/result endpoint"""
    status: str = Field(..., description="Result status")
    message: str = Field(..., description="Status message")
    total_income_usd: float = Field(..., description="Total RunPod income accumulated")

class BuybackResponse(BaseModel):
    """Response model for /trigger-buyback endpoint"""
    status: str = Field(..., description="Buyback status")
    message: str = Field(..., description="Status message")
    amount_usd: Optional[float] = Field(None, description="Amount used for buyback")
    tokens_burned: Optional[float] = Field(None, description="Number of tokens burned (simulated)")
    transaction_hash: Optional[str] = Field(None, description="Transaction hash (simulated)")

class SystemMetricsResponse(BaseModel):
    """Response model for system metrics"""
    runpod_income_usd: float = Field(..., description="Total RunPod income")
    total_idle_jobs_processed: int = Field(..., description="Total idle jobs processed")
    total_buyback_usd: float = Field(..., description="Total USD used for buybacks")
    last_buyback_timestamp: Optional[datetime] = Field(None, description="Last buyback timestamp")

class IdleJobModel(BaseModel):
    """Idle job model for API responses"""
    id: int
    prompt: str
    status: str
    submitted_by: str
    created_at: datetime
    completed_at: Optional[datetime]
    output_tokens: Optional[int]
    usd_earned: Optional[float]
    result: Optional[str]
    runpod_job_id: Optional[str]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class GolemJobRequest(BaseModel):
    """Request model for /report_golem_job endpoint"""
    miner_name: str = Field(..., description="Name of the miner")
    duration_sec: int = Field(..., description="Duration of fallback compute in seconds", gt=0)
    estimated_glm: float = Field(..., description="Estimated GLM earned", gt=0)
    timestamp: datetime = Field(..., description="Timestamp when fallback ended")

class GolemJobResponse(BaseModel):
    """Response model for /report_golem_job endpoint"""
    status: str = Field(..., description="Report status")
    message: str = Field(..., description="Status message")
    total_glm: float = Field(..., description="Total GLM earned by this miner")

class GolemJobModel(BaseModel):
    """Golem job model for API responses"""
    id: int
    miner_name: str
    duration_sec: int
    estimated_glm: float
    timestamp: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True

class MinerStatusRequest(BaseModel):
    """Request model for /miner/{miner_id}/status endpoint"""
    status: str = Field(..., description="Miner status", pattern="^(idle|busy|offline)$")

class MinerStatusResponse(BaseModel):
    """Response model for /miner/{miner_id}/status endpoint"""
    status: str = Field(..., description="Update status")
    message: str = Field(..., description="Status message")