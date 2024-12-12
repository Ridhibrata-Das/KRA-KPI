"""Security configuration and middleware."""
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel
import secrets
from fastapi import HTTPException, Security, Depends
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from ..config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "kpi:read": "Read KPIs",
        "kpi:write": "Create and update KPIs",
        "kpi:delete": "Delete KPIs",
        "kpi:assign": "Manage KPI assignments",
        "admin": "Administrative access"
    }
)

# Security settings
SECURITY_CONFIG = {
    "PASSWORD_MIN_LENGTH": 12,
    "PASSWORD_REQUIRE_UPPERCASE": True,
    "PASSWORD_REQUIRE_LOWERCASE": True,
    "PASSWORD_REQUIRE_NUMBERS": True,
    "PASSWORD_REQUIRE_SPECIAL": True,
    "SESSION_TIMEOUT_MINUTES": 30,
    "MAX_LOGIN_ATTEMPTS": 5,
    "LOCKOUT_DURATION_MINUTES": 15,
    "MFA_ENABLED": True,
    "API_RATE_LIMIT": 100,  # requests per minute
    "ANALYTICS_RATE_LIMIT": 1000,  # requests per day
    "JWT_EXPIRY_HOURS": 1,
    "REFRESH_TOKEN_EXPIRY_DAYS": 7,
    "SECURE_HEADERS": {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }
}

class Token(BaseModel):
    """Token model."""
    access_token: str
    token_type: str
    expires_at: datetime
    refresh_token: Optional[str] = None
    scopes: List[str] = []

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=SECURITY_CONFIG["JWT_EXPIRY_HOURS"])
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

def create_refresh_token(user_id: str) -> str:
    """Create refresh token."""
    expires = datetime.utcnow() + timedelta(days=SECURITY_CONFIG["REFRESH_TOKEN_EXPIRY_DAYS"])
    return jwt.encode(
        {
            "sub": user_id,
            "type": "refresh",
            "exp": expires,
            "iat": datetime.utcnow()
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)

def validate_password(password: str) -> bool:
    """Validate password strength."""
    if len(password) < SECURITY_CONFIG["PASSWORD_MIN_LENGTH"]:
        return False
    if SECURITY_CONFIG["PASSWORD_REQUIRE_UPPERCASE"] and not any(c.isupper() for c in password):
        return False
    if SECURITY_CONFIG["PASSWORD_REQUIRE_LOWERCASE"] and not any(c.islower() for c in password):
        return False
    if SECURITY_CONFIG["PASSWORD_REQUIRE_NUMBERS"] and not any(c.isdigit() for c in password):
        return False
    if SECURITY_CONFIG["PASSWORD_REQUIRE_SPECIAL"] and not any(not c.isalnum() for c in password):
        return False
    return True

def generate_mfa_secret() -> str:
    """Generate MFA secret."""
    return secrets.token_hex(20)

def verify_mfa_code(secret: str, code: str) -> bool:
    """Verify MFA code."""
    # Implement MFA verification logic
    # This is a placeholder - implement actual MFA verification
    return True  # Replace with actual implementation

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme)
) -> dict:
    """Get current user from token."""
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        for scope in security_scopes.scopes:
            if scope not in token_scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    # This is a placeholder - implement actual user retrieval
    user = {"id": user_id, "scopes": token_scopes}
    if user is None:
        raise credentials_exception
    return user
