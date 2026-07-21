"""
services/upload_service.py

Why this file exists:
    Business-logic layer between the API route and the low-level file
    utilities. Owns the "what happens when a file is uploaded" workflow,
    independent of HTTP concerns (which live in api/routes/upload.py) and
    independent of raw disk operations (which live in utils/file_handler.py).

What it does:
    - Generates a job_id
    - Validates and saves the uploaded file
    - Extracts zip archives if needed
    - Locates the best candidate model file for parsing

How it connects:
    Called by api/routes/upload.py. Its output (job_id, candidate file
    path) is handed to services/parser_service.py on the subsequent
    GET /graph/{job_id} call.
"""

import logging
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ModelParsingError
from app.utils.file_handler import (
    extract_if_archive,
    find_candidate_model_files,
    save_upload,
    validate_upload,
)

logger = logging.getLogger(__name__)


def create_job_id() -> str:
    return uuid.uuid4().hex[:12]


def handle_upload(filename: str, content: bytes) -> str:
    """
    Validate, save, and (if needed) extract an uploaded file.
    Returns the job_id that the client should poll for graph results.
    """
    validate_upload(
        filename=filename,
        size_bytes=len(content),
        allowed_extensions=settings.allowed_extensions,
        max_size_mb=settings.max_upload_size_mb,
    )

    job_id = create_job_id()
    job_dir = settings.upload_dir / job_id
    saved_path = save_upload(job_dir, filename, content)
    extract_if_archive(saved_path)

    logger.info("Upload handled: job_id=%s filename=%s", job_id, filename)
    return job_id


def resolve_candidate_model_file(job_id: str) -> Path:
    """
    Given an existing job_id, locate the most likely model source file
    to parse. Raises ModelParsingError if the job folder doesn't exist
    or contains no Python files at all.
    """
    job_dir = settings.upload_dir / job_id
    if not job_dir.exists():
        raise ModelParsingError(f"No uploaded job found for id '{job_id}'.")

    search_root = job_dir / "extracted" if (job_dir / "extracted").exists() else job_dir
    candidates = find_candidate_model_files(search_root)

    if not candidates:
        raise ModelParsingError("No Python (.py) files were found in the uploaded project.")

    return candidates[0]
