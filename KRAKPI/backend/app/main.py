from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import time
from datetime import datetime

from .api import kpi_routes
from .config import settings
from .utils.logging import setup_monitoring, api_logger
from .docs.descriptions import (
    API_DESCRIPTION,
    TAGS_METADATA,
    OPERATIONS,
    RESPONSES
)

app = FastAPI(
    title="KPI Management System",
    description=API_DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS_METADATA,
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup monitoring
setup_monitoring(app)

# Include routers
app.include_router(kpi_routes.router)

# Custom exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    timestamp = datetime.now().isoformat()
    
    # Log the error
    api_logger.error(
        f"Global error handler caught: {str(exc)}",
        extra={
            "url": str(request.url),
            "method": request.method,
            "timestamp": timestamp
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "code": "INTERNAL_ERROR",
            "timestamp": timestamp
        }
    )

# Custom OpenAPI docs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation"""
    return app.redoc_url

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version
    }

# OpenAPI customization
def custom_openapi():
    """Customize OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = app.openapi()
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add security requirement to all operations
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add custom operation descriptions
    paths = openapi_schema["paths"]
    for path, operations in paths.items():
        for method, operation in operations.items():
            operation_id = operation.get("operationId")
            if operation_id in OPERATIONS:
                operation.update(OPERATIONS[operation_id])
    
    # Add response examples
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            if "responses" in operation:
                for status_code, response in operation["responses"].items():
                    if status_code in RESPONSES:
                        response.update(RESPONSES[status_code])
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
