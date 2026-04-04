"""
auth_service.py — User account management and JWT token operations.

Uses bcrypt directly (no passlib) to avoid the passlib/bcrypt 4.x
incompatibility (AttributeError: module 'bcrypt' has no attribute '__about__').

MongoDB collection: db[settings.MONGODB_USERS_COLLECTION]
Document shape:
  {
    email:           str   (unique index),
    username:        str   (unique index),
    hashed_password: str,
    is_active:       bool,
    created_at:      datetime,
    last_login:      datetime | None,
  }
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from jose import jwt

from app.core.config import settings
from app.core.exceptions import (
    UserAlreadyExistsError,
    InputValidationError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGODB_URI)
    return _client


class AuthService:
    """Handles user lifecycle and JWT operations."""

    def __init__(self):
        client = _get_client()
        db = client[settings.MONGODB_DB_NAME]
        self.users = db[settings.MONGODB_USERS_COLLECTION]

    # ------------------------------------------------------------------ #
    # Indexes  (call once at startup)
    # ------------------------------------------------------------------ #

    async def ensure_indexes(self) -> None:
        """Create unique indexes on email and username if they don't exist."""
        await self.users.create_index("email",    unique=True, background=True)
        await self.users.create_index("username", unique=True, background=True)
        logger.info("User collection indexes verified")

    # ------------------------------------------------------------------ #
    # Password helpers  (bcrypt directly — no passlib)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _hash(password: str) -> str:
        """Hash a plaintext password with bcrypt."""
        return bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode("utf-8")

    @staticmethod
    def _verify(plain: str, hashed: str) -> bool:
        """Constant-time password verification."""
        try:
            return bcrypt.checkpw(
                plain.encode("utf-8"),
                hashed.encode("utf-8"),
            )
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # JWT helpers
    # ------------------------------------------------------------------ #

    def create_access_token(self, email: str, username: str) -> str:
        """Issue a signed JWT valid for ACCESS_TOKEN_EXPIRE_HOURS hours."""
        expire = datetime.now(timezone.utc) + timedelta(
            hours=settings.ACCESS_TOKEN_EXPIRE_HOURS
        )
        payload = {
            "sub":      email,
            "username": username,
            "exp":      expire,
        }
        return jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

    # ------------------------------------------------------------------ #
    # User operations
    # ------------------------------------------------------------------ #

    async def create_user(
        self,
        email: str,
        username: str,
        password: str,
    ) -> dict:
        """
        Register a new user.

        Raises:
          InputValidationError   — password exceeds bcrypt's 72-byte limit
          UserAlreadyExistsError — email or username already taken
        Returns the public user dict on success.
        """
        # bcrypt hard-errors on passwords > 72 bytes — validate upfront
        if len(password.encode("utf-8")) > 72:
            raise InputValidationError(
                "Password is too long. Please use a maximum of 72 characters."
            )

        # Check for email / username conflicts
        conflict = await self.users.find_one(
            {"$or": [{"email": email}, {"username": username}]}
        )
        if conflict:
            if conflict["email"] == email:
                raise UserAlreadyExistsError(
                    "This email address is already registered."
                )
            raise UserAlreadyExistsError("This username is already taken.")

        now = datetime.now(timezone.utc)
        doc = {
            "email":           email,
            "username":        username,
            "hashed_password": self._hash(password),
            "is_active":       True,
            "created_at":      now,
            "last_login":      None,
        }
        try:
            await self.users.insert_one(doc)
        except DuplicateKeyError:
            # Race condition: another request inserted between check and insert
            raise UserAlreadyExistsError(
                "An account with this email or username already exists."
            )

        logger.info(f"New user registered: {email}")
        return {"email": email, "username": username, "created_at": now}

    async def authenticate(self, email: str, password: str) -> dict:
        """
        Verify credentials and return the public user dict.
        Always raises AuthenticationError on any failure — never leaks
        which field (email vs password) was wrong.
        """
        user = await self.users.find_one({"email": email})

        # Always run _verify even when user is not found to prevent
        # timing-based email enumeration attacks
        dummy_hash = (
            "$2b$12$KIXlA5zF0YuPpFfU5CTB2OeH8kDMTg0S8k3K5g3Q5K5K5K5K5K5K"
        )
        candidate_hash = user["hashed_password"] if user else dummy_hash

        if not self._verify(password, candidate_hash) or not user:
            raise AuthenticationError("Invalid email or password.")

        if not user.get("is_active", True):
            raise AuthenticationError("This account has been deactivated.")

        # Update last_login — best effort, don't fail the login if this errors
        try:
            await self.users.update_one(
                {"email": email},
                {"$set": {"last_login": datetime.now(timezone.utc)}},
            )
        except Exception as e:
            logger.warning(f"Failed to update last_login for {email}: {e}")

        logger.info(f"User authenticated: {email}")
        return {
            "email":      user["email"],
            "username":   user["username"],
            "created_at": user["created_at"],
        }

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        """
        Return minimal user info by email.
        Used by the JWT dependency to confirm the token subject is still active.
        Returns None if not found or inactive.
        """
        user = await self.users.find_one({"email": email})
        if not user or not user.get("is_active", True):
            return None
        return {
            "email":      user["email"],
            "username":   user["username"],
            "is_active":  user["is_active"],
            "created_at": user["created_at"],
        }