# Author: Bradley R. Kinnard
"""
User storage with SQLite persistence.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from ..core.models.user import User

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "users.db"


class UserStore:
    """SQLite-backed user store with persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.commit()
        conn.close()

    def _user_to_row(self, user: User) -> tuple:
        return (
            str(user.id),
            user.email.lower(),
            user.name,
            user.hashed_password,
            user.created_at.isoformat(),
            user.updated_at.isoformat(),
            1 if user.is_active else 0,
        )

    def _row_to_user(self, row: tuple) -> User:
        return User(
            id=UUID(row[0]),
            email=row[1],
            name=row[2],
            hashed_password=row[3],
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            is_active=bool(row[6]),
        )

    async def create(self, user: User) -> User:
        async with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    "INSERT INTO users (id, email, name, hashed_password, created_at, updated_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._user_to_row(user),
                )
                conn.commit()
                return user
            except sqlite3.IntegrityError:
                raise ValueError(f"Email {user.email} already registered")
            finally:
                conn.close()

    async def get(self, user_id: UUID) -> Optional[User]:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT * FROM users WHERE id = ?", (str(user_id),))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_user(row) if row else None

    async def get_by_email(self, email: str) -> Optional[User]:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
        row = cursor.fetchone()
        conn.close()
        return self._row_to_user(row) if row else None

    async def update(self, user: User) -> User:
        async with self._lock:
            conn = sqlite3.connect(self._db_path)
            user.updated_at = datetime.now(timezone.utc)
            conn.execute(
                "UPDATE users SET email=?, name=?, hashed_password=?, updated_at=?, is_active=? WHERE id=?",
                (user.email.lower(), user.name, user.hashed_password, user.updated_at.isoformat(), 1 if user.is_active else 0, str(user.id)),
            )
            conn.commit()
            conn.close()
            return user

    async def delete(self, user_id: UUID) -> bool:
        async with self._lock:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (str(user_id),))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted

    async def list_all(self) -> list[User]:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_user(row) for row in rows]

    async def count(self) -> int:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        conn.close()
        return count


# Singleton
_user_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
    return _user_store


__all__ = ["UserStore", "get_user_store"]
