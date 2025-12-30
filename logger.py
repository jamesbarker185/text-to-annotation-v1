
import logging
import sys
from config import get_settings

settings = get_settings()

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(settings.LOG_LEVEL)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(settings.LOG_LEVEL)
        
        # Formatter
        formatter = logging.Formatter(settings.LOG_FORMAT)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # Prevent propagation to avoid double logging with Uvicorn
        logger.propagate = False
        
    return logger

def log_performance(logger, operation: str, duration: float, metadata: dict = None):
    """
    Log performance metrics in a structured way.
    """
    msg = f"[PERF] {operation} completed in {duration:.4f}s"
    if metadata:
        msg += f" | {metadata}"
    logger.info(msg)
