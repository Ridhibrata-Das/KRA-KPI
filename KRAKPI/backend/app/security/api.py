"""API security features including rate limiting and request validation."""
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer
import re
import hashlib
import hmac
from redis.asyncio import Redis
from pydantic import BaseModel
from ..config import settings
from ..utils.logging import security_logger
from ..monitoring.audit import audit_logger_instance, AuditEventType

class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    rate_limit: int
    allowed_ips: List[str]
    allowed_endpoints: List[str]
    metadata: Dict[str, Any]

class RequestValidator:
    """Request validation and sanitization."""
    
    def __init__(self):
        self.sql_injection_patterns = [
            r"(?i)(union|select|insert|delete|drop|update|;|--)",
            r"(?i)(/\*|\*/|@@|@)",
            r"(?i)char\s*\(",
            r"(?i)declare\s+[@#]",
            r"(?i)exec\s+\(",
        ]
        
        self.xss_patterns = [
            r"(?i)<script",
            r"(?i)javascript:",
            r"(?i)vbscript:",
            r"(?i)onclick",
            r"(?i)onerror",
            r"(?i)onload",
            r"(?i)eval\(",
        ]
        
        self.path_traversal_patterns = [
            r"(?i)\.\.\/",
            r"(?i)\.\.\\",
            r"%2e%2e%2f",
            r"%252e%252e%252f",
        ]
    
    async def validate_request(self, request: Request) -> bool:
        """Validate incoming request."""
        try:
            # Check request size
            content_length = request.headers.get("content-length", 0)
            if int(content_length) > settings.MAX_REQUEST_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Request too large"
                )
            
            # Validate headers
            await self._validate_headers(request)
            
            # Validate query parameters
            await self._validate_query_params(request)
            
            # Validate request body
            await self._validate_body(request)
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            security_logger.error(
                f"Request validation error: {str(e)}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid request"
            )
    
    async def _validate_headers(self, request: Request):
        """Validate request headers."""
        required_headers = ["user-agent", "host"]
        for header in required_headers:
            if header not in request.headers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required header: {header}"
                )
        
        # Validate content type
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith(("application/json", "multipart/form-data")):
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported media type"
                )
    
    async def _validate_query_params(self, request: Request):
        """Validate query parameters."""
        for param, value in request.query_params.items():
            # Check for injection patterns
            for pattern in (self.sql_injection_patterns +
                          self.xss_patterns +
                          self.path_traversal_patterns):
                if re.search(pattern, value):
                    await audit_logger_instance.log_event(
                        event_type=AuditEventType.SECURITY,
                        action="request_validation",
                        status="failed",
                        details={
                            "reason": "Malicious pattern detected",
                            "parameter": param,
                            "pattern": pattern
                        }
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid query parameter"
                    )
    
    async def _validate_body(self, request: Request):
        """Validate request body."""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                body_str = body.decode()
                
                # Check size
                if len(body_str) > settings.MAX_REQUEST_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Request body too large"
                    )
                
                # Check for injection patterns
                for pattern in (self.sql_injection_patterns +
                              self.xss_patterns):
                    if re.search(pattern, body_str):
                        await audit_logger_instance.log_event(
                            event_type=AuditEventType.SECURITY,
                            action="request_validation",
                            status="failed",
                            details={
                                "reason": "Malicious pattern detected in body",
                                "pattern": pattern
                            }
                        )
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid request body"
                        )
                
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request body encoding"
                )

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self._setup_redis()
    
    async def _setup_redis(self):
        """Setup Redis connection."""
        try:
            self.redis = Redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            security_logger.error(
                f"Failed to setup Redis: {str(e)}",
                extra={"timestamp": datetime.now().isoformat()}
            )
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if request is within rate limit."""
        if not self.redis:
            return True
        
        try:
            pipe = self.redis.pipeline()
            now = datetime.now().timestamp()
            window_start = now - window
            
            # Remove old requests
            await pipe.zremrangebyscore(
                key,
                "-inf",
                window_start
            )
            
            # Count requests in window
            await pipe.zcard(key)
            
            # Add new request
            await pipe.zadd(key, {str(now): now})
            
            # Set expiry
            await pipe.expire(key, window)
            
            # Execute pipeline
            results = await pipe.execute()
            request_count = results[1]
            
            return request_count < limit
            
        except Exception as e:
            security_logger.error(
                f"Rate limit check error: {str(e)}",
                extra={
                    "key": key,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return True
    
    def get_rate_limit_key(
        self,
        request: Request,
        api_key: Optional[str] = None
    ) -> str:
        """Generate rate limit key."""
        if api_key:
            return f"rate_limit:api:{api_key}"
        return f"rate_limit:ip:{request.client.host}"

class APIKeyManager:
    """API key management."""
    
    def __init__(self):
        self.bearer = HTTPBearer()
        self.redis: Optional[Redis] = None
        self._setup_redis()
    
    async def _setup_redis(self):
        """Setup Redis connection."""
        try:
            self.redis = Redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
        except Exception as e:
            security_logger.error(
                f"Failed to setup Redis: {str(e)}",
                extra={"timestamp": datetime.now().isoformat()}
            )
    
    def _generate_key(self) -> tuple[str, str]:
        """Generate API key pair."""
        key_id = hashlib.sha256(
            datetime.now().isoformat().encode()
        ).hexdigest()[:12]
        
        key = hmac.new(
            settings.SECRET_KEY.encode(),
            key_id.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return key_id, key
    
    async def create_api_key(
        self,
        name: str,
        rate_limit: int = 1000,
        allowed_ips: Optional[List[str]] = None,
        allowed_endpoints: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None
    ) -> APIKey:
        """Create new API key."""
        key_id, key = self._generate_key()
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or [],
            allowed_endpoints=allowed_endpoints or [],
            metadata={}
        )
        
        # Store API key
        if self.redis:
            await self.redis.hmset(
                f"api_key:{key_id}",
                api_key.dict()
            )
        
        await audit_logger_instance.log_event(
            event_type=AuditEventType.SECURITY,
            action="create_api_key",
            status="success",
            details={"key_id": key_id, "name": name}
        )
        
        return api_key
    
    async def validate_api_key(
        self,
        request: Request,
        api_key: str
    ) -> bool:
        """Validate API key."""
        if not self.redis:
            return False
        
        try:
            key_id, key = api_key.split(".")
            key_data = await self.redis.hgetall(f"api_key:{key_id}")
            
            if not key_data:
                return False
            
            api_key_obj = APIKey(**key_data)
            
            # Check if expired
            if (api_key_obj.expires_at and
                datetime.now() > api_key_obj.expires_at):
                return False
            
            # Validate key hash
            if api_key_obj.key_hash != hashlib.sha256(key.encode()).hexdigest():
                return False
            
            # Check IP restriction
            if (api_key_obj.allowed_ips and
                request.client.host not in api_key_obj.allowed_ips):
                return False
            
            # Check endpoint restriction
            if (api_key_obj.allowed_endpoints and
                not any(request.url.path.startswith(ep)
                       for ep in api_key_obj.allowed_endpoints)):
                return False
            
            return True
            
        except Exception as e:
            security_logger.error(
                f"API key validation error: {str(e)}",
                extra={
                    "timestamp": datetime.now().isoformat()
                }
            )
            return False
    
    async def revoke_api_key(self, key_id: str):
        """Revoke API key."""
        if self.redis:
            await self.redis.delete(f"api_key:{key_id}")
            
            await audit_logger_instance.log_event(
                event_type=AuditEventType.SECURITY,
                action="revoke_api_key",
                status="success",
                details={"key_id": key_id}
            )

# Initialize components
request_validator = RequestValidator()
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
