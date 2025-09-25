"""Configuration settings for the RivalLens application."""
import os
from pydantic_settings import BaseSettings
from pydantic import HttpUrl
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    DEEPSEEK_API_KEY: str
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_ENDPOINT: Optional[str] = None  # For backward compatibility
    NEWS_API_KEY: Optional[str] = None
    CRUNCHBASE_API_KEY: Optional[str] = None
    
    # Application settings
    APP_NAME: str = "RivalLens API"
    DEBUG: bool = False
    
    # API settings
    API_PREFIX: str = "/api/v1"
    MAX_COMPETITORS: int = 5
    DEFAULT_CITATION_DEPTH: int = 3
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'  # Ignore extra environment variables
        
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            # This ensures that .env file is loaded with higher priority than environment variables
            return (env_settings, init_settings, file_secret_settings)

# Create settings instance
settings = Settings()
