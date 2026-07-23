"""
main.py

Why this file exists:
    The FastAPI application entrypoint. Wires together configuration,
    CORS, logging, and route registration. This is the only file that
    should be run directly (via uvicorn).

What it does:
    Creates the FastAPI app instance, configures CORS for the Vite dev
    server, and includes all API routers.

How it connects:
    Imports settings from core/config.py and routers from api/routes/*.
    Run with: uvicorn app.main:app --reload
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import graph, health, source, upload, uploads
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
app = FastAPI(
    title=settings.app_name,
    description="Analyzes deep learning projects and generates interactive neural network architecture diagrams.",
    version="0.1.0-phase1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(graph.router)
app.include_router(source.router)
app.include_router(uploads.router)

@app.get("/")
async def root() -> dict:
    return {
        "app": settings.app_name,
        "phase": "Phase 1 - Core Parsing Engine (PyTorch only, no DB/auth/queue yet)",
        "docs": "/docs",
    }
