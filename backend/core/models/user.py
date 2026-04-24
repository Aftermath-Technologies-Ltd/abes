# Author: Bradley R. Kinnard
"""
User model for authentication.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(BaseModel):
    """User account for ABES."""

    id: UUID = Field(default_factory=uuid4)
    email: str
    name: str
    hashed_password: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    is_active: bool = True

    model_config = {
        "from_attributes": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class UserCreate(BaseModel):
    """Schema for user registration."""
    email: str
    name: str
    password: str


class UserLogin(BaseModel):
    """Schema for user login."""
    email: str
    password: str


class UserResponse(BaseModel):
    """Public user info (no password)."""
    id: UUID
    email: str
    name: str
    created_at: datetime
    is_active: bool


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


__all__ = ["User", "UserCreate", "UserLogin", "UserResponse", "TokenResponse"]
