
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8095
    API_TITLE: str = "SAM3 Rapid Platform"
    API_VERSION: str = "v1"
    DEBUG: bool = False

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "uploads")
    
    # Model Paths
    SAM3_CHECKPOINT: str = "sam3/sam3.pt"
    
    # Model Settings
    DEVICE: str = "cuda" # or "cpu"
    LAZY_LOAD_MODELS: bool = True
    
    # Caching
    OCR_CACHE_SIZE: int = 128
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
