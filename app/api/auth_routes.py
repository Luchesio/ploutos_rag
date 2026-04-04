"""
auth_routes.py — Public authentication endpoints.

POST /auth/signup  — create a new user account
POST /auth/signin  — exchange credentials for a JWT access token

These routes are intentionally public (no auth dependency).
All protected routes live in api/routes.py and require a valid Bearer token.
"""

import logging
from fastapi import APIRouter

from app.models.user import UserSignUp, UserSignIn, TokenResponse, SignUpResponse
from app.services.auth_service import AuthService
from app.core.config import settings
from app.core.exceptions import FemiException
from app.core.sanitiser import sanitise_auth_input

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["Authentication"])


# ------------------------------------------------------------------ #
# Signup
# ------------------------------------------------------------------ #

@auth_router.post(
    "/signup",
    response_model=SignUpResponse,
    status_code=201,
    summary="Create a new user account",
)
async def signup(data: UserSignUp):
    """
    Register a new account with email, username and password.

    Password requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character (!@#$%^&* etc.)

    On success returns the created user's public profile.
    """
    # Sanitise string inputs
    email    = sanitise_auth_input(data.email)
    username = sanitise_auth_input(data.username)

    try:
        svc  = AuthService()
        user = await svc.create_user(
            email=email,
            username=username,
            password=data.password,   # raw — AuthService hashes it
        )
        return SignUpResponse(
            message=f"Account created successfully. Welcome to {settings.BOT_OWNER}!",
            user=user,
        )
    except Exception as e:
        logger.exception(f"Signup failed for {email}: {e}")
        raise


# ------------------------------------------------------------------ #
# Signin
# ------------------------------------------------------------------ #

@auth_router.post(
    "/signin",
    response_model=TokenResponse,
    summary="Sign in and receive a JWT access token",
)
async def signin(data: UserSignIn):
    """
    Authenticate with email and password.

    Returns a Bearer token to include in all subsequent requests:
        Authorization: Bearer <access_token>

    Token is valid for 24 hours.
    """
    email = sanitise_auth_input(data.email)

    try:
        svc  = AuthService()
        user = await svc.authenticate(email=email, password=data.password)
        token = svc.create_access_token(
            email=user["email"],
            username=user["username"],
        )
        return TokenResponse(
            access_token=token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_HOURS * 3600,
            user=user,
        )
    except Exception as e:
        logger.exception(f"Signin failed for {email}: {e}")
        raise