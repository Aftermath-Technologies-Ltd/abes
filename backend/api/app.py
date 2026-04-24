# Author: Bradley R. Kinnard
"""
FastAPI application for ABES.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import (
    agents_router,
    auth_router,
    bel_router,
    beliefs_router,
    chat_router,
    clusters_router,
    snapshots_router,
)

ATTRIBUTION_BANNER = (
    "ABES | Adaptive Belief Ecology System | v1.0 | "
    "Developed by Bradley R. Kinnard | Aftermath Technologies"
)

app = FastAPI(
    title="ABES API",
    description="Adaptive Belief Ecology System API",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(auth_router)
app.include_router(beliefs_router)
app.include_router(snapshots_router)
app.include_router(agents_router)
app.include_router(bel_router)
app.include_router(clusters_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ABES API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/version")
async def version():
    """Version and attribution endpoint."""
    return {
        "attribution": ATTRIBUTION_BANNER,
    }


__all__ = ["app"]
