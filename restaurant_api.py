"""
Restaurant API endpoints for MIA Backend
Provides direct generation without the work queue system
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import requests
import os
import random

router = APIRouter(prefix="/api", tags=["restaurant"])
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.7
    source: Optional[str] = "unknown"

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int = 0
    source: str = "mia"

# Get available miners
def get_available_miner():
    """Get an available miner for direct generation"""
    # This would be replaced with actual miner selection logic
    # For now, we'll try to find a connected miner
    
    # Try local miner first
    local_miner_url = os.getenv("LOCAL_MINER_URL", "http://localhost:8000")
    try:
        health_response = requests.get(f"{local_miner_url}/health", timeout=2)
        if health_response.status_code == 200:
            return local_miner_url
    except:
        pass
    
    # In production, this would check the miners table for available miners
    # For now, return None if no local miner
    return None

@router.post("/generate", response_model=GenerateResponse)
async def generate_direct(request: GenerateRequest):
    """
    Direct generation endpoint for restaurant and other services
    Bypasses the work queue system
    """
    logger.info(f"Direct generation request from: {request.source}")
    
    # Find an available miner
    miner_url = get_available_miner()
    
    if not miner_url:
        # No miners available - return a fallback response
        logger.warning("No miners available for direct generation")
        
        # Simple fallback responses for common queries
        prompt_lower = request.prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            fallback_text = "Hello! Welcome to our restaurant. How can I help you today?"
        elif any(word in prompt_lower for word in ['hour', 'open', 'close']):
            fallback_text = "Please check with our staff for current hours, or visit our website for the latest information."
        elif any(word in prompt_lower for word in ['menu', 'food', 'dish']):
            fallback_text = "I'd be happy to help you with our menu. What type of cuisine or specific dish are you interested in?"
        else:
            fallback_text = "I apologize, but I'm having trouble processing your request. Please try again or ask our staff for assistance."
        
        return GenerateResponse(
            text=fallback_text,
            tokens_generated=len(fallback_text.split()),
            source="fallback"
        )
    
    try:
        # Forward to the miner
        miner_response = requests.post(
            f"{miner_url}/generate",
            json={
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            },
            timeout=30
        )
        
        if miner_response.status_code == 200:
            result = miner_response.json()
            return GenerateResponse(
                text=result.get("text", ""),
                tokens_generated=result.get("tokens_generated", 0),
                source="miner"
            )
        else:
            logger.error(f"Miner error: {miner_response.status_code}")
            raise HTTPException(status_code=500, detail="Generation failed")
            
    except requests.exceptions.Timeout:
        logger.error("Miner request timed out")
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for restaurant API"""
    miner_available = get_available_miner() is not None
    
    return {
        "status": "healthy",
        "service": "restaurant-api",
        "miner_available": miner_available
    }