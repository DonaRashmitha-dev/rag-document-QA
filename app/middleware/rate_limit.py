# MIT License
# Copyright (c) 2024 Cursor AI

from functools import wraps


def rate_limit(func):
    """No-op rate limit decorator for compatibility."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
