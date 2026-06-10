# MIT License
# Copyright (c) 2026 RAG-QA Contributors

"""JWT authentication middleware and decorator."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any

import jwt
import structlog
from flask import Request, g, jsonify, request

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class AuthenticationError(Exception):
    """Raised when JWT validation fails."""

    def __init__(self, message: str, status_code: int = 401) -> None:
        """Initialize authentication error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code to return.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def create_access_token(
    subject: str = "rag-qa-client",
    settings: Settings | None = None,
    expires_hours: int | None = None,
) -> str:
    """Create a signed JWT access token.

    Args:
        subject: Token subject claim.
        settings: Application settings with jwt_secret.
        expires_hours: Token lifetime in hours.

    Returns:
        Encoded JWT string.
    """
    cfg = settings or get_settings()
    expiry = expires_hours or cfg.jwt_expiry_hours
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(hours=expiry),
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """Decode and validate a JWT token.

    Args:
        token: Encoded JWT string.
        settings: Application settings.

    Returns:
        Decoded token payload.

    Raises:
        AuthenticationError: If the token is invalid or expired.
    """
    cfg = settings or get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=[cfg.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid authentication token") from exc


def _extract_bearer_token(req: Request) -> str:
    """Extract the Bearer token from the Authorization header.

    Args:
        req: Flask request object.

    Returns:
        Raw JWT string.

    Raises:
        AuthenticationError: If the header is missing or malformed.
    """
    auth_header = req.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Missing or invalid Authorization header")
    return auth_header[7:].strip()


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that enforces JWT authentication on a Flask route.

    Args:
        func: Flask view function to wrap.

    Returns:
        Wrapped view function that validates JWT before execution.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            token = _extract_bearer_token(request)
            payload = decode_token(token)
            g.current_user = payload.get("sub", "unknown")
        except AuthenticationError as exc:
            logger.warning("auth_failed", error=exc.message)
            return jsonify({"error": exc.message}), exc.status_code
        return func(*args, **kwargs)

    return wrapper


# Alias for backward compatibility with route decorators
token_required = require_auth
token_required = require_auth
