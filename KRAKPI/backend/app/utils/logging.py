import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import traceback
import sys
import os
from logging.handlers import RotatingFileHandler

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Create loggers
kpi_logger = logging.getLogger('kpi_management')
assignment_logger = logging.getLogger('kpi_assignments')
api_logger = logging.getLogger('api_operations')
validation_logger = logging.getLogger('validation')

# Set log levels
kpi_logger.setLevel(logging.INFO)
assignment_logger.setLevel(logging.INFO)
api_logger.setLevel(logging.INFO)
validation_logger.setLevel(logging.INFO)

# Create handlers
kpi_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'kpi_management.log'),
    maxBytes=10485760,  # 10MB
    backupCount=5
)
assignment_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'kpi_assignments.log'),
    maxBytes=10485760,
    backupCount=5
)
api_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'api_operations.log'),
    maxBytes=10485760,
    backupCount=5
)
validation_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'validation.log'),
    maxBytes=10485760,
    backupCount=5
)

# Set formats
formatter = logging.Formatter(LOG_FORMAT)
kpi_handler.setFormatter(formatter)
assignment_handler.setFormatter(formatter)
api_handler.setFormatter(formatter)
validation_handler.setFormatter(formatter)

# Add handlers
kpi_logger.addHandler(kpi_handler)
assignment_logger.addHandler(assignment_handler)
api_logger.addHandler(api_handler)
validation_logger.addHandler(validation_handler)

def log_operation(logger: logging.Logger):
    """Decorator for logging operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation = func.__name__
            start_time = datetime.now()
            
            # Log operation start
            logger.info(f"Operation '{operation}' started")
            logger.debug(f"Arguments: args={args}, kwargs={kwargs}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Operation '{operation}' completed successfully in {duration}s")
                
                return result
            
            except Exception as e:
                # Log error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Operation '{operation}' failed after {duration}s: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise
        
        return wrapper
    return decorator

def log_validation(logger: logging.Logger = validation_logger):
    """Decorator for logging validation operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Validation '{func.__name__}' passed")
                return result
            except Exception as e:
                logger.error(
                    f"Validation '{func.__name__}' failed: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise
        return wrapper
    return decorator

def log_api_request(logger: logging.Logger = api_logger):
    """Decorator for logging API requests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if hasattr(arg, 'method') and hasattr(arg, 'url'):
                        request = arg
                        break
            
            start_time = datetime.now()
            
            # Log request
            logger.info(f"API Request: {request.method} {request.url}")
            logger.debug(f"Headers: {dict(request.headers)}")
            
            try:
                response = await func(*args, **kwargs)
                
                # Log response
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"API Response: {response.status_code} "
                    f"(completed in {duration}s)"
                )
                
                return response
            
            except Exception as e:
                # Log error
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"API Request failed after {duration}s: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                raise
        
        return wrapper
    return decorator

def sanitize_log_data(data: Any) -> Any:
    """Sanitize sensitive data before logging"""
    if isinstance(data, dict):
        return {
            k: sanitize_log_data(v)
            for k, v in data.items()
            if k not in {'password', 'token', 'secret', 'api_key'}
        }
    elif isinstance(data, (list, tuple)):
        return [sanitize_log_data(item) for item in data]
    return data

def log_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log error with context"""
    error_data = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'context': sanitize_log_data(context) if context else None
    }
    
    logger.error(f"Error occurred: {json.dumps(error_data, indent=2)}")

def setup_monitoring(app):
    """Setup monitoring for the application"""
    @app.middleware("http")
    async def log_request_timing(request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log request timing
        api_logger.info(
            f"Request timing: {request.method} {request.url} "
            f"completed in {duration}s"
        )
        
        return response
