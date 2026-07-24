"""
core/config.py

Why this file exists:
    Centralizes all environment-driven configuration (storage paths, upload
    limits, CORS origins) in one place, using pydantic-settings so values
    can be overridden via environment variables in production without
    touching code.

What it does:
    Defines the Settings class with sane defaults for local development.

How it connects:
    Imported by main.py (CORS setup) and by services/file_service.py
    (to know where to save uploads and what limits to enforce).
"""

from pathlib import Path

from pydantic_settings import BaseSettings


from typing import Optional

class Settings(BaseSettings):
    app_name: str = "NeuralNetworkAnalyzer"
    api_prefix: str = "/api/v1"
    database_url: Optional[str] = None

    # Security
    jwt_secret_key: str = "default_nna_development_secret_key_change_me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""

    # Storage
    storage_root: Path = Path(__file__).resolve().parent.parent.parent / "storage"
    upload_dir_name: str = "uploads"

    # Upload limits
    max_upload_size_mb: int = 100
    allowed_extensions: tuple = (".py", ".zip")

    # CORS
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")

    @property
    def upload_dir(self) -> Path:
        path = self.storage_root / self.upload_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    class Config:
        env_prefix = "NNA_"
        env_file = ".env"
        extra = "ignore"


settings = Settings()
