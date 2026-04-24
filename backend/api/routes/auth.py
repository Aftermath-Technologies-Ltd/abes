# Author: Bradley R. Kinnard
"""
Authentication API routes - register, login, logout, current user.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header

from ...auth import (
    create_access_token,
    decode_access_token,
    get_user_id_from_token,
    hash_password,
    verify_password,
)
from ...core.models.user import (
    TokenResponse,
    User,
    UserCreate,
    UserLogin,
    UserResponse,
)
from ...storage.user_store import get_user_store

router = APIRouter(prefix="/auth", tags=["auth"])


async def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Dependency to get the current authenticated user."""
    if not authorization:
        raise HTTPException(401, "Not authenticated")

    # Handle "Bearer <token>" format
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1]
    else:
        token = authorization

    user_id = get_user_id_from_token(token)
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")

    store = get_user_store()
    user = await store.get(user_id)
    if not user:
        raise HTTPException(401, "User not found")

    if not user.is_active:
        raise HTTPException(401, "User account is disabled")

    return user


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[User]:
    """Dependency that returns user if authenticated, None otherwise."""
    if not authorization:
        return None
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return None


def _user_to_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        created_at=user.created_at,
        is_active=user.is_active,
    )


@router.post("/register", response_model=TokenResponse)
async def register(req: UserCreate):
    """Register a new user account."""
    store = get_user_store()

    # Validate email format (basic check)
    if "@" not in req.email or "." not in req.email:
        raise HTTPException(400, "Invalid email format")

    # Check password strength
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    # Check if email already exists
    existing = await store.get_by_email(req.email)
    if existing:
        raise HTTPException(400, "Email already registered")

    # Create user
    user = User(
        email=req.email.lower().strip(),
        name=req.name.strip(),
        hashed_password=hash_password(req.password),
    )

    await store.create(user)

    # Generate token
    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: UserLogin):
    """Login with email and password."""
    store = get_user_store()

    user = await store.get_by_email(req.email.lower().strip())
    if not user:
        raise HTTPException(401, "Invalid email or password")

    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")

    if not user.is_active:
        raise HTTPException(401, "User account is disabled")

    # Generate token
    token = create_access_token(user.id, user.email)

    return TokenResponse(
        access_token=token,
        user=_user_to_response(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""
    return _user_to_response(user)


@router.post("/logout")
async def logout(user: User = Depends(get_current_user)):
    """Logout (client should discard token)."""
    # JWT is stateless, so logout is handled client-side
    # We just return success
    return {"message": "Logged out successfully"}


__all__ = ["router", "get_current_user", "get_optional_user"]
