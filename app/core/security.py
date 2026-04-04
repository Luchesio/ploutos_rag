"""
security.py — Authentication, rate limiting, and input sanitisation.

JWT authentication
  Clients must include:  Authorization: Bearer <token>
  Token is issued by POST /auth/signin and expires after ACCESS_TOKEN_EXPIRE_HOURS.
  The get_current_user() dependency validates the token and returns the user dict.

API key authentication (dev/legacy)
  Clients may send:  X-API-Key: <key>
  If settings.API_KEY is not configured, auth falls back to client IP (dev mode).
  Kept for backward compatibility on internal/admin endpoints.

Rate limiting
  Sliding-window counter keyed by user email (or client IP in dev mode).
  In-memory — sufficient for single-process. Swap _rate_store for Redis in
  a multi-process deployment.

Input sanitisation
  Moved to app/core/sanitiser.py — imported and re-exported here for
  backward compatibility with existing imports.
"""

import time
import secrets
import logging
from collections import defaultdict

from fastapi import Security, Request, Depends
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from app.core.config import settings
from app.core.exceptions import AuthenticationError, RateLimitError, InputValidationError
from app.core.sanitiser import sanitise_input, sanitise_auth_input  # re-export

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# JWT auth
# ------------------------------------------------------------------ #

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency — validates the Bearer JWT token.

    Decodes the token, verifies expiry and signature, then confirms
    the user still exists and is active in MongoDB.

    Returns the user dict: { email, username, is_active, created_at }
    Raises AuthenticationError on any failure.
    """
    if not credentials:
        raise AuthenticationError("Missing or invalid Authorization header. "
                                  "Please sign in to obtain a token.")
    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        email: str | None = payload.get("sub")
        if not email:
            raise AuthenticationError("Invalid token: missing subject claim.")
    except JWTError as e:
        logger.warning(f"JWT decode failed: {e}")
        raise AuthenticationError("Token is invalid or has expired. Please sign in again.")

    # Confirm the user still exists and is active
    from app.services.auth_service import AuthService
    svc  = AuthService()
    user = await svc.get_user_by_email(email)
    if not user:
        raise AuthenticationError("Account not found or has been deactivated.")

    return user


# ------------------------------------------------------------------ #
# API key auth  (dev / legacy — kept for internal endpoints)
# ------------------------------------------------------------------ #

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
) -> str:
    """
    FastAPI dependency — validates X-API-Key header.
    Returns the key (or client IP when auth is disabled) as a rate-limit ID.
    """
    if not settings.API_KEY:
        logger.debug("API key auth disabled — no API_KEY configured")
        return _get_client_ip(request)

    if not api_key:
        raise AuthenticationError("Missing X-API-Key header.")

    if not secrets.compare_digest(api_key.strip(), settings.API_KEY.strip()):
        raise AuthenticationError()

    return api_key


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ------------------------------------------------------------------ #
# Rate limiter  (sliding window, in-memory)
# ------------------------------------------------------------------ #

_rate_store: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(identifier: str) -> None:
    """
    Sliding-window rate limiter.
    Allows RATE_LIMIT_PER_MINUTE requests per 60-second window per identifier.
    Raises RateLimitError when exceeded.
    """
    limit  = settings.RATE_LIMIT_PER_MINUTE
    window = 60.0
    now    = time.monotonic()

    timestamps = _rate_store[identifier]
    _rate_store[identifier] = [t for t in timestamps if now - t < window]

    if len(_rate_store[identifier]) >= limit:
        logger.warning(
            f"Rate limit exceeded for '{identifier[:20]}' "
            f"({len(_rate_store[identifier])}/{limit} req/min)"
        )
        raise RateLimitError()

    _rate_store[identifier].append(now)