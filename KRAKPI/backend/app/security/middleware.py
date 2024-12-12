"""Security middleware for the application."""
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
from datetime import datetime
import ipaddress
import re
from .config import SECURITY_CONFIG
from ..utils.logging import security_logger

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request/response processing."""
    
    def __init__(
        self,
        app: ASGIApp,
        redis_client=None,
        excluded_paths: list = None
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.excluded_paths = excluded_paths or ["/health", "/metrics"]
        self.ip_blacklist = set()
        self.suspicious_patterns = [
            r"(?i)(union|select|insert|delete|drop|update|;|--)",  # SQL Injection
            r"(?i)<script|javascript:|vbscript:|onclick=",  # XSS
            r"(?i)../|%2e%2e%2f|%252e%252e%252f",  # Path Traversal
            r"(?i)/etc/passwd|/etc/shadow|/proc/self/environ",  # System File Access
            r"(?i)eval\(|exec\(|system\(|`.*`",  # Code Injection
        ]
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process each request/response."""
        start_time = time.time()
        
        # Skip security checks for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Security checks
        try:
            # 1. IP Validation
            if not await self._validate_ip(request):
                return Response(
                    content="Access denied",
                    status_code=403
                )
            
            # 2. Rate Limiting
            if not await self._check_rate_limit(request):
                return Response(
                    content="Too many requests",
                    status_code=429
                )
            
            # 3. Input Validation
            if not await self._validate_input(request):
                return Response(
                    content="Invalid input detected",
                    status_code=400
                )
            
            # Process request
            response = await call_next(request)
            
            # 4. Add Security Headers
            response.headers.update(SECURITY_CONFIG["SECURE_HEADERS"])
            
            # 5. Log security events
            await self._log_security_event(request, response, start_time)
            
            return response
            
        except Exception as e:
            security_logger.error(
                f"Security middleware error: {str(e)}",
                extra={
                    "ip": request.client.host,
                    "path": request.url.path,
                    "method": request.method,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return Response(
                content="Internal server error",
                status_code=500
            )
    
    async def _validate_ip(self, request: Request) -> bool:
        """Validate IP address."""
        ip = request.client.host
        
        # Check if IP is blacklisted
        if ip in self.ip_blacklist:
            return False
        
        # Check if IP is private
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                return True  # Allow private IPs
        except ValueError:
            return False
        
        # Additional IP validation logic here
        return True
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Check rate limiting."""
        if not self.redis_client:
            return True
        
        ip = request.client.host
        key = f"rate_limit:{ip}"
        
        # Get current count
        count = await self.redis_client.get(key) or 0
        count = int(count)
        
        # Check if analytics endpoint
        if "/analytics" in request.url.path:
            limit = SECURITY_CONFIG["ANALYTICS_RATE_LIMIT"]
            window = 86400  # 24 hours
        else:
            limit = SECURITY_CONFIG["API_RATE_LIMIT"]
            window = 60  # 1 minute
        
        if count >= limit:
            return False
        
        # Increment counter
        await self.redis_client.incr(key)
        await self.redis_client.expire(key, window)
        
        return True
    
    async def _validate_input(self, request: Request) -> bool:
        """Validate request input for security threats."""
        # Get request data
        query_params = str(request.query_params)
        path_params = request.path_params
        headers = request.headers
        
        # Check for suspicious patterns
        data_to_check = [
            query_params,
            str(path_params),
            str(headers)
        ]
        
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                data_to_check.append(body.decode())
            except:
                pass
        
        # Check each pattern
        for data in data_to_check:
            for pattern in self.suspicious_patterns:
                if re.search(pattern, data):
                    security_logger.warning(
                        f"Suspicious pattern detected: {pattern}",
                        extra={
                            "ip": request.client.host,
                            "path": request.url.path,
                            "method": request.method
                        }
                    )
                    return False
        
        return True
    
    async def _log_security_event(
        self,
        request: Request,
        response: Response,
        start_time: float
    ):
        """Log security relevant events."""
        duration = time.time() - start_time
        
        security_logger.info(
            "Security event",
            extra={
                "ip": request.client.host,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration": duration,
                "user_agent": request.headers.get("user-agent"),
                "timestamp": datetime.now().isoformat()
            }
        )
