"""
user.py — Pydantic schemas for user authentication.
"""

import re
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator


# ------------------------------------------------------------------ #
# Request schemas
# ------------------------------------------------------------------ #

class UserSignUp(BaseModel):
    email: EmailStr = Field(..., description="Valid email address")
    username: str = Field(
        ...,
        min_length=3,
        max_length=30,
        description="3–30 characters, letters, numbers and underscores only",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="8–72 characters with uppercase, lowercase, digit and special char",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                "Username may only contain letters, numbers and underscores."
            )
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        errors = []
        if not re.search(r"[A-Z]", v):
            errors.append("at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            errors.append("at least one lowercase letter")
        if not re.search(r"\d", v):
            errors.append("at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-\+\=\[\]\/\\]", v):
            errors.append("at least one special character")
        if errors:
            raise ValueError(f"Password must contain {', '.join(errors)}.")
        return v


class UserSignIn(BaseModel):
    email: EmailStr = Field(..., description="Registered email address")
    password: str   = Field(..., description="Account password")


# ------------------------------------------------------------------ #
# Response schemas
# ------------------------------------------------------------------ #

class UserPublic(BaseModel):
    """Safe user representation — never includes password or internal IDs."""
    email:      str
    username:   str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int              # seconds until expiry
    user:         UserPublic


class SignUpResponse(BaseModel):
    message: str
    user:    UserPublic