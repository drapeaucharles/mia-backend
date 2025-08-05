import secrets
import string
from datetime import datetime, timezone

def generate_auth_key(length: int = 32) -> str:
    """Generate a secure random authentication key"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def utc_now() -> datetime:
    """Get current UTC timestamp"""
    return datetime.now(timezone.utc)

def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO 8601 string"""
    return dt.isoformat() if dt else None

def validate_session_id(session_id: str) -> bool:
    """Validate session ID format (UUID)"""
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(session_id))

def sanitize_input(text: str, max_length: int = 5000) -> str:
    """Sanitize and limit user input"""
    if not text:
        return ""
    # Remove control characters and limit length
    sanitized = ''.join(char for char in text if char.isprintable() or char.isspace())
    return sanitized[:max_length].strip()

def calculate_job_priority(business_id: int = None, message_length: int = 0) -> int:
    """Calculate job priority (lower number = higher priority)"""
    # Base priority
    priority = 100
    
    # Business customers get higher priority
    if business_id:
        priority -= 50
    
    # Shorter messages get slightly higher priority
    if message_length < 100:
        priority -= 10
    elif message_length > 1000:
        priority += 10
    
    return max(priority, 1)  # Ensure priority is at least 1