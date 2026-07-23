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

from app.api.routes import graph, health, source, upload, uploads, projects, auth
from app.core.config import settings
from app.db.session import Base, engine
from app.models.graph import SavedGraph
from app.models.project import Project
from app.models.user import User
from sqlalchemy import inspect, text

# Create database tables automatically
Base.metadata.create_all(bind=engine)

# Lightweight migration for adding columns if they don't exist
inspector = inspect(engine)
if "saved_graphs" in inspector.get_table_names():
    columns = [col["name"] for col in inspector.get_columns("saved_graphs")]
    if "project_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE saved_graphs ADD COLUMN project_id VARCHAR;"))

if "projects" in inspector.get_table_names():
    columns = [col["name"] for col in inspector.get_columns("projects")]
    if "owner_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE projects ADD COLUMN owner_id VARCHAR;"))

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
app.include_router(projects.router)
app.include_router(auth.router)

@app.get("/")
async def root() -> dict:
    return {
        "app": settings.app_name,
        "phase": "Phase 1 - Core Parsing Engine (PyTorch only, no DB/auth/queue yet)",
        "docs": "/docs",
    }
