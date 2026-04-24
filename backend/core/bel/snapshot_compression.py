"""
Compress snapshots with msgpack and zlib for smaller storage.
"""

import zlib
from datetime import datetime
from typing import Any
from uuid import UUID

import msgpack

from ...storage import Snapshot


def _walk_encode(obj: Any) -> Any:
    """Turn UUID and datetime into safe msgpack values."""
    if isinstance(obj, UUID):
        return {"__uuid__": str(obj)}
    elif isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    elif isinstance(obj, dict):
        return {k: _walk_encode(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_walk_encode(item) for item in obj]
    return obj


def _walk_decode(obj: Any) -> Any:
    """Reverse the tagged values back into UUID or datetime."""
    if isinstance(obj, dict):
        # check for type tags first
        if "__uuid__" in obj:
            return UUID(obj["__uuid__"])
        if "__datetime__" in obj:
            return datetime.fromisoformat(obj["__datetime__"])
        # otherwise recursively decode
        return {k: _walk_decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_decode(item) for item in obj]
    return obj


def compress_snapshot(snapshot: Snapshot) -> bytes:
    """Pack snapshot into compressed bytes."""
    snapshot_dict = snapshot.model_dump(mode="json")
    encoded = _walk_encode(snapshot_dict)
    packed = msgpack.packb(
        encoded, use_bin_type=True, strict_types=False
    )  # strict_types=False handles mixed types
    compressed = zlib.compress(packed, level=9)
    return compressed


def decompress_snapshot(data: bytes) -> Snapshot:
    """Unpack and reconstruct snapshot from bytes."""
    decompressed = zlib.decompress(data)
    unpacked = msgpack.unpackb(decompressed, raw=False)
    decoded = _walk_decode(unpacked)
    return Snapshot(**decoded)


__all__ = ["compress_snapshot", "decompress_snapshot"]
