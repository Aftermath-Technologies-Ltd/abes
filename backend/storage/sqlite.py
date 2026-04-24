# Author: Bradley R. Kinnard
"""
SQLite-based belief storage with async support via aiosqlite.
Provides persistent storage that survives server restarts.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aiosqlite

from ..core.models.belief import Belief, BeliefStatus, OriginMetadata
from .base import BeliefStoreABC

logger = logging.getLogger(__name__)


class SQLiteBeliefStore(BeliefStoreABC):
    """
    SQLite-backed belief storage. Persists beliefs across restarts.
    Uses aiosqlite for async operations.
    """

    def __init__(self, db_path: str = "./data/abes.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _get_db(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._db is None:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._init_schema()
        return self._db

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return

        db = self._db
        await db.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                tension REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active',
                cluster_id TEXT,
                parent_id TEXT,
                use_count INTEGER DEFAULT 0,
                tags TEXT DEFAULT '[]',
                origin_source TEXT,
                origin_turn_index INTEGER,
                origin_episode_id TEXT,
                origin_timestamp TEXT,
                origin_last_reinforced TEXT,
                relevance REAL,
                score REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                salience REAL DEFAULT 0.5,
                half_life_days REAL DEFAULT 7.0,
                evidence_for TEXT DEFAULT '[]',
                evidence_against TEXT DEFAULT '[]',
                links TEXT DEFAULT '[]'
            )
        """)

        # Indexes for common queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_cluster ON beliefs(cluster_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_session ON beliefs(session_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_updated ON beliefs(updated_at)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_user_id ON beliefs(user_id)")

        # Auto-migrate old schemas missing new columns
        for col, coltype in [
            ("user_id", "TEXT"),
            ("salience", "REAL DEFAULT 0.5"),
            ("half_life_days", "REAL DEFAULT 7.0"),
            ("evidence_for", "TEXT DEFAULT '[]'"),
            ("evidence_against", "TEXT DEFAULT '[]'"),
            ("links", "TEXT DEFAULT '[]'"),
        ]:
            try:
                await db.execute(f"ALTER TABLE beliefs ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists

        await db.commit()
        self._initialized = True
        logger.info(f"SQLite schema initialized at {self.db_path}")

    def _belief_to_row(self, belief: Belief) -> dict:
        """Convert Belief to database row."""
        return {
            "id": str(belief.id),
            "content": belief.content,
            "confidence": belief.confidence,
            "tension": belief.tension,
            "status": belief.status.value,
            "cluster_id": str(belief.cluster_id) if belief.cluster_id else None,
            "parent_id": str(belief.parent_id) if belief.parent_id else None,
            "use_count": belief.use_count,
            "tags": json.dumps(belief.tags),
            "origin_source": belief.origin.source,
            "origin_turn_index": belief.origin.turn_index,
            "origin_episode_id": str(belief.origin.episode_id) if belief.origin.episode_id else None,
            "origin_timestamp": belief.origin.timestamp.isoformat() if belief.origin.timestamp else None,
            "origin_last_reinforced": belief.origin.last_reinforced.isoformat() if belief.origin.last_reinforced else None,
            "relevance": belief.relevance,
            "score": belief.score,
            "created_at": belief.created_at.isoformat(),
            "updated_at": belief.updated_at.isoformat(),
            "session_id": getattr(belief, "session_id", None),
            "user_id": str(uid) if (uid := getattr(belief, "user_id", None)) else None,
            "salience": belief.salience,
            "half_life_days": belief.half_life_days,
            "evidence_for": json.dumps([e.model_dump(mode='json') if hasattr(e, 'model_dump') else e for e in belief.evidence_for]),
            "evidence_against": json.dumps([e.model_dump(mode='json') if hasattr(e, 'model_dump') else e for e in belief.evidence_against]),
            "links": json.dumps([l.model_dump(mode='json') if hasattr(l, 'model_dump') else l for l in belief.links]),
        }

    def _row_to_belief(self, row: aiosqlite.Row) -> Belief:
        """Convert database row to Belief."""
        origin = OriginMetadata(
            source=row["origin_source"] or "unknown",
            turn_index=row["origin_turn_index"],
            episode_id=UUID(row["origin_episode_id"]) if row["origin_episode_id"] else None,
            timestamp=datetime.fromisoformat(row["origin_timestamp"]) if row["origin_timestamp"] else datetime.now(timezone.utc),
            last_reinforced=datetime.fromisoformat(row["origin_last_reinforced"]) if row["origin_last_reinforced"] else None,
        )

        # deserialize ecology fields
        evidence_for_raw = json.loads(row["evidence_for"]) if row["evidence_for"] else []
        evidence_against_raw = json.loads(row["evidence_against"]) if row["evidence_against"] else []
        links_raw = json.loads(row["links"]) if row["links"] else []

        belief = Belief(
            id=UUID(row["id"]),
            content=row["content"],
            confidence=row["confidence"],
            tension=row["tension"],
            status=BeliefStatus(row["status"]),
            cluster_id=UUID(row["cluster_id"]) if row["cluster_id"] else None,
            parent_id=UUID(row["parent_id"]) if row["parent_id"] else None,
            use_count=row["use_count"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            origin=origin,
            relevance=row["relevance"],
            score=row["score"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            salience=row["salience"] if row["salience"] is not None else 0.5,
            half_life_days=row["half_life_days"] if row["half_life_days"] is not None else 7.0,
        )

        # restore session_id / user_id
        if row["session_id"]:
            object.__setattr__(belief, "session_id", row["session_id"])
        if row["user_id"]:
            object.__setattr__(belief, "user_id", row["user_id"])

        # restore evidence and links from JSON
        from backend.core.models.belief import EvidenceRef, BeliefLink
        belief.evidence_for = [EvidenceRef(**e) if isinstance(e, dict) else e for e in evidence_for_raw]
        belief.evidence_against = [EvidenceRef(**e) if isinstance(e, dict) else e for e in evidence_against_raw]
        belief.links = [BeliefLink(**l) if isinstance(l, dict) else l for l in links_raw]

        return belief

    async def create(self, belief: Belief) -> Belief:
        """Insert a new belief."""
        async with self._lock:
            db = await self._get_db()
            row = self._belief_to_row(belief)

            columns = ", ".join(row.keys())
            placeholders = ", ".join(f":{k}" for k in row.keys())

            await db.execute(
                f"INSERT INTO beliefs ({columns}) VALUES ({placeholders})",
                row,
            )
            await db.commit()
            return belief

    async def get(self, belief_id: UUID) -> Optional[Belief]:
        """Get a belief by ID."""
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM beliefs WHERE id = ?", (str(belief_id),)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_belief(row)
        return None

    async def list(
        self,
        status: Optional[BeliefStatus] = None,
        cluster_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> List[Belief]:
        """List beliefs with filters. user_id is the ceiling - never cross-user."""
        db = await self._get_db()

        query = "SELECT * FROM beliefs WHERE 1=1"
        params = []

        # user_id is the ceiling - always filter if provided
        if user_id:
            query += " AND user_id = ?"
            params.append(str(user_id))

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if cluster_id:
            query += " AND cluster_id = ?"
            params.append(str(cluster_id))

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if max_confidence is not None:
            query += " AND confidence <= ?"
            params.append(max_confidence)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            beliefs = [self._row_to_belief(row) for row in rows]

            # Filter by tags in Python (JSON array matching is tricky in SQLite)
            if tags:
                beliefs = [b for b in beliefs if any(t in b.tags for t in tags)]

            return beliefs

    async def update(self, belief: Belief) -> Belief:
        """Update an existing belief."""
        async with self._lock:
            db = await self._get_db()
            row = self._belief_to_row(belief)

            set_clause = ", ".join(f"{k} = :{k}" for k in row.keys() if k != "id")
            await db.execute(
                f"UPDATE beliefs SET {set_clause} WHERE id = :id",
                row,
            )
            await db.commit()
            return belief

    async def delete(self, belief_id: UUID) -> bool:
        """Delete a belief by ID."""
        async with self._lock:
            db = await self._get_db()
            cursor = await db.execute(
                "DELETE FROM beliefs WHERE id = ?", (str(belief_id),)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        status: Optional[BeliefStatus] = None,
    ) -> List[Belief]:
        """
        Embedding search. SQLite doesn't support vector search natively,
        so we load all beliefs and compute similarity in Python.
        For production, consider using sqlite-vss extension.
        """
        # For now, return empty - would need to store embeddings
        # and compute similarity in Python
        return []

    async def bulk_update(self, beliefs: List[Belief]) -> int:
        """Update multiple beliefs in a transaction."""
        async with self._lock:
            db = await self._get_db()
            count = 0

            for belief in beliefs:
                row = self._belief_to_row(belief)
                set_clause = ", ".join(f"{k} = :{k}" for k in row.keys() if k != "id")
                cursor = await db.execute(
                    f"UPDATE beliefs SET {set_clause} WHERE id = :id",
                    row,
                )
                count += cursor.rowcount

            await db.commit()
            return count

    async def count(self, status: Optional[BeliefStatus] = None) -> int:
        """Count beliefs, optionally filtered by status."""
        db = await self._get_db()
        query = "SELECT COUNT(*) FROM beliefs"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status.value)

        async with db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False

    async def clear_all(self) -> int:
        """Delete all beliefs. Use with caution."""
        async with self._lock:
            db = await self._get_db()
            cursor = await db.execute("DELETE FROM beliefs")
            await db.commit()
            count = cursor.rowcount
            logger.warning(f"Cleared {count} beliefs from database")
            return count


__all__ = ["SQLiteBeliefStore"]
