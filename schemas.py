from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for /chat endpoint"""
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[str] = Field(None, description="Additional context for the AI")
    business_id: Optional[int] = Field(None, description="Business ID for multi-tenant support")

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