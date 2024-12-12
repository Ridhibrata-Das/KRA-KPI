from pydantic import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "KPI Management System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api"
    
    # Database settings
    MONGODB_URL: str = "mongodb://localhost:27017"
    DATABASE_NAME: str = "kpi_management"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 300  # 5 minutes
    
    # JWT settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = "logs"
    
    # Email settings
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Notification settings
    SLACK_WEBHOOK_URL: Optional[str] = None
    TEAMS_WEBHOOK_URL: Optional[str] = None
    
    # Performance settings
    MAX_CONNECTIONS_COUNT: int = 10
    MIN_CONNECTIONS_COUNT: int = 5
    
    # Security settings
    ALLOWED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings"""
    return Settings()

# Environment-specific settings
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ALLOWED_HOSTS: list[str] = ["api.yourdomain.com"]
    CORS_ORIGINS: list[str] = ["https://yourdomain.com"]

class TestSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_NAME: str = "kpi_management_test"
    TESTING: bool = True

# Get environment-specific settings
def get_environment_settings() -> Settings:
    """Get environment-specific settings"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    settings_map = {
        "development": DevelopmentSettings,
        "production": ProductionSettings,
        "test": TestSettings
    }
    SettingsClass = settings_map.get(env, DevelopmentSettings)
    return SettingsClass()

# Validation
def validate_settings(settings: Settings) -> None:
    """Validate settings"""
    required_settings = [
        ("JWT_SECRET_KEY", "JWT secret key is required"),
        ("MONGODB_URL", "MongoDB URL is required"),
        ("DATABASE_NAME", "Database name is required")
    ]
    
    for setting_name, error_message in required_settings:
        if not getattr(settings, setting_name):
            raise ValueError(error_message)
    
    # Validate email settings if any email setting is provided
    email_settings = [
        settings.SMTP_HOST,
        settings.SMTP_PORT,
        settings.SMTP_USER,
        settings.SMTP_PASSWORD
    ]
    if any(email_settings) and not all(email_settings):
        raise ValueError("All email settings must be provided if any are set")

# Initialize settings
settings = get_environment_settings()
validate_settings(settings)
