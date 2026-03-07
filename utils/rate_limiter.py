import time
import functools
import re
from utils.logger import get_logger

logger = get_logger("rate_limiter")

def retry_on_rate_limit(max_retries=5, initial_backoff=1.0, max_backoff=60.0, backoff_factor=2.0):
    """
    Decorator that catches HTTP 429 Too Many Requests or generic rate limit exceptions
    and retries the function with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    # Check if error message implies a rate limit (HTTP 429, 'rate limit', 'too many requests')
                    if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                        if retries >= max_retries:
                            logger.error(f"[{func.__name__}] Rate limit exceeded. Max retries ({max_retries}) reached. Failing.")
                            raise e
                        
                        logger.warning(f"[{func.__name__}] Rate limit hit: '{e}'. Retrying in {backoff:.2f}s... (Attempt {retries+1}/{max_retries})")
                        time.sleep(backoff)
                        
                        retries += 1
                        backoff = min(backoff * backoff_factor, max_backoff)
                    else:
                        # If it's not a rate limit error, raise it immediately
                        raise e
                        
        return wrapper
    return decorator
