# Author: Bradley R. Kinnard
"""API Routes package."""

from .agents import router as agents_router
from .auth import router as auth_router
from .bel import router as bel_router
from .beliefs import router as beliefs_router
from .chat import router as chat_router
from .clusters import router as clusters_router
from .snapshots import router as snapshots_router

__all__ = [
    "agents_router",
    "auth_router",
    "bel_router",
    "beliefs_router",
    "chat_router",
    "clusters_router",
    "snapshots_router",
]
