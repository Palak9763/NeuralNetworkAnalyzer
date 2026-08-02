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
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

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

if "users" in inspector.get_table_names():
    columns = [col["name"] for col in inspector.get_columns("users")]
    if "auth_provider" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN auth_provider VARCHAR DEFAULT 'local';"))
    if "google_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN google_id VARCHAR;"))
    if "github_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN github_id VARCHAR;"))
    if "hashed_password" in columns:
        # Ensure hashed_password is nullable for Google OAuth users
        # SQLite doesn't support ALTER COLUMN, but the column is already created as nullable
        # in PostgreSQL we can alter if needed
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

_cleanup_logger = logging.getLogger("upload_cleanup")


def _cleanup_old_uploads(
    upload_dir: Path,
    max_age_hours: int = 24,
    interval_secs: int = 3600,
) -> None:
    """
    Background daemon thread: periodically delete job upload folders
    that are older than max_age_hours.

    Runs every interval_secs (default: 1 hour). Safe to leave running
    indefinitely — the daemon flag ensures it dies when the gunicorn
    worker process is recycled or the app shuts down.

    Race condition analysis: parse requests complete in at most a few
    minutes; the 24-hour threshold provides a large safety margin so
    no in-progress parse is ever affected by this sweep.
    """
    while True:
        time.sleep(interval_secs)  # sleep first so startup isn't slowed
        try:
            cutoff = time.time() - max_age_hours * 3600
            for job_dir in upload_dir.iterdir():
                if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    _cleanup_logger.info("Removed stale upload dir: %s", job_dir.name)
        except Exception as exc:  # noqa: BLE001
            _cleanup_logger.warning("Upload cleanup sweep failed: %s", exc)


@asynccontextmanager
async def lifespan(app):
    """Startup: launch upload-cleanup daemon. Shutdown: nothing to do
    (daemon thread dies with the process automatically)."""
    upload_dir = settings.upload_dir  # also ensures directory exists
    t = threading.Thread(
        target=_cleanup_old_uploads,
        args=(upload_dir,),
        daemon=True,
        name="upload-cleanup",
    )
    t.start()
    _cleanup_logger.info(
        "Upload cleanup thread started (max_age=24h, interval=1h, dir=%s)",
        upload_dir,
    )
    yield
    # Daemon threads are terminated automatically on process exit.


app = FastAPI(
    title=settings.app_name,
    description="Analyzes deep learning projects and generates interactive neural network architecture diagrams.",
    version="0.1.0-phase1",
    lifespan=lifespan,
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
