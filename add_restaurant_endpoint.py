"""
Quick script to add restaurant endpoint to main.py
Run this to add the endpoint without full router
"""

# Add this to your main.py file after the other endpoints:

endpoint_code = '''
# Restaurant API endpoint for direct generation
@app.post("/api/generate")
async def generate_for_restaurant(request: dict):
    """Direct generation endpoint for restaurant and other services"""
    
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 150)
    temperature = request.get("temperature", 0.7)
    source = request.get("source", "unknown")
    
    logger.info(f"Direct generation request from: {source}")
    
    # Simple fallback response for now
    # In production, this would forward to an available miner
    
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'bonjour', 'hola']):
        if 'hola' in prompt_lower:
            text = "¡Hola! Bienvenido a nuestro restaurante. ¿En qué puedo ayudarte hoy?"
        elif 'bonjour' in prompt_lower:
            text = "Bonjour! Bienvenue dans notre restaurant. Comment puis-je vous aider aujourd'hui?"
        else:
            text = "Hello! Welcome to our restaurant. How can I help you today?"
    elif any(word in prompt_lower for word in ['hour', 'open', 'close', 'when']):
        text = "Please check with our staff for current hours, or visit our website for the latest information."
    elif any(word in prompt_lower for word in ['menu', 'food', 'dish', 'eat']):
        text = "I'd be happy to help you with our menu. What type of cuisine or specific dish are you interested in?"
    elif any(word in prompt_lower for word in ['vegetarian', 'vegan', 'gluten']):
        text = "We have several options for dietary preferences. Let me know what specific dietary requirements you have, and I'll help you find suitable dishes."
    else:
        text = "I'd be happy to help you. Could you please tell me more about what you're looking for?"
    
    return {
        "text": text,
        "tokens_generated": len(text.split()),
        "source": "fallback"
    }

@app.get("/api/health")
async def api_health_check():
    """Health check for restaurant API"""
    return {
        "status": "healthy",
        "service": "restaurant-api",
        "endpoint": "/api/generate"
    }
'''

print("Add this code to your main.py file in the MIA backend")
print("=" * 60)
print(endpoint_code)