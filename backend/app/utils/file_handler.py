"""
utils/file_handler.py

Why this file exists:
    Pure filesystem utilities used by the upload flow. Kept separate from
    the API layer and from business logic so it can be unit tested in
    isolation and reused by any service that needs to touch disk.

What it does:
    - Validates uploaded file extensions and size
    - Saves uploaded bytes to a per-job folder on disk
    - Unzips project archives
    - Walks a folder tree to find likely "model" Python files

How it connects:
    Used by services/upload_service.py (save + validate + extract) and by
    engines/detector/framework_detector.py (to know which file to inspect).
"""

import logging
import shutil
import zipfile
from pathlib import Path

from app.core.exceptions import FileTooLargeError, InvalidFileTypeError

logger = logging.getLogger(__name__)

MODEL_FILENAME_HINTS = ("model", "network", "architecture", "net")
MODEL_CONTENT_HINTS = ("nn.Module", "keras.Model", "tf.keras.Model")


def validate_upload(filename: str, size_bytes: int, allowed_extensions: tuple, max_size_mb: int) -> None:
    """Raise a domain exception if the upload violates size/type rules."""
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        raise InvalidFileTypeError(
            f"'{suffix}' is not a supported file type. Allowed: {allowed_extensions}"
        )

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File is {size_bytes / (1024 * 1024):.1f}MB, exceeds the {max_size_mb}MB limit."
        )


def save_upload(job_dir: Path, filename: str, content: bytes) -> Path:
    """Write raw upload bytes to disk inside the job's folder."""
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / filename
    dest.write_bytes(content)
    logger.info("Saved upload %s (%d bytes) to %s", filename, len(content), dest)
    return dest


def extract_if_archive(file_path: Path) -> Path:
    """
    If file_path is a .zip, extract it into a sibling 'extracted' folder
    and return that folder. Otherwise return the original file's parent.
    """
    if file_path.suffix.lower() != ".zip":
        return file_path.parent

    extract_dir = file_path.parent / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(extract_dir)
    logger.info("Extracted %s into %s", file_path.name, extract_dir)
    return extract_dir


def find_candidate_model_files(root_dir: Path) -> list[Path]:
    """
    Walk the extracted project and return .py files that are likely to
    contain a neural network model definition, ranked by confidence:
    filename hints first, then content hints, then any remaining .py file.
    """
    all_py_files = [p for p in root_dir.rglob("*.py") if p.is_file()]
    if not all_py_files:
        return []

    filename_matches = [
        p for p in all_py_files if any(hint in p.stem.lower() for hint in MODEL_FILENAME_HINTS)
    ]

    content_matches = []
    for p in all_py_files:
        if p in filename_matches:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if any(hint in text for hint in MODEL_CONTENT_HINTS):
            content_matches.append(p)

    remaining = [p for p in all_py_files if p not in filename_matches and p not in content_matches]

    ranked = filename_matches + content_matches + remaining
    logger.info(
        "Found %d candidate model file(s) in %s (filename_matches=%d, content_matches=%d)",
        len(ranked), root_dir, len(filename_matches), len(content_matches),
    )
    return ranked


def cleanup_job(job_dir: Path) -> None:
    """Remove a job's temp folder entirely. Used for TTL cleanup jobs later."""
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        logger.info("Cleaned up job directory %s", job_dir)
