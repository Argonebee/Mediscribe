"""Production hardening: error handling, validation, logging, and retry logic."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

T = TypeVar("T")

# Configure logging
logger = logging.getLogger("mediscribe")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class RetryableError(Exception):
    """Raised for transient errors that should trigger retry."""
    pass


def validate_input(text: str, field_name: str = "text", min_length: int = 1, max_length: int = 50000) -> str:
    """Validate and sanitize text input."""
    if not isinstance(text, str):
        logger.warning(f"Invalid type for {field_name}: expected str, got {type(text)}")
        raise ValidationError(f"{field_name} must be a string")
    
    text = text.strip()
    
    if len(text) < min_length:
        logger.warning(f"{field_name} too short: {len(text)} < {min_length}")
        raise ValidationError(f"{field_name} must be at least {min_length} characters")
    
    if len(text) > max_length:
        logger.warning(f"{field_name} too long: {len(text)} > {max_length}")
        raise ValidationError(f"{field_name} must be at most {max_length} characters")
    
    logger.debug(f"Validated {field_name}: {len(text)} characters")
    return text


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_error = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(f"Attempt {attempt}/{max_attempts} for {func.__name__}")
                    return func(*args, **kwargs)
                except (RetryableError, TimeoutError, ConnectionError) as e:
                    last_error = e
                    if attempt < max_attempts:
                        logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {e}")
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            if last_error:
                raise last_error
            return cast(T, None)
        
        return wrapper
    return decorator


class RequestLogger:
    """Log API requests and responses for auditing."""
    
    @staticmethod
    def log_request(endpoint: str, method: str, params: Dict[str, Any]) -> None:
        """Log incoming request."""
        logger.info(f"Request: {method} {endpoint} | params: {list(params.keys())}")
    
    @staticmethod
    def log_response(endpoint: str, status_code: int, duration_ms: float) -> None:
        """Log outgoing response."""
        logger.info(f"Response: {endpoint} | status: {status_code} | duration: {duration_ms:.2f}ms")
    
    @staticmethod
    def log_error(endpoint: str, error: Exception, context: str = "") -> None:
        """Log error with context."""
        logger.error(f"Error in {endpoint}: {type(error).__name__}: {str(error)} | context: {context}")
