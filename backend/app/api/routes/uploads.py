"""
api/routes/uploads.py

HTTP Method: GET
Route:       /api/v1/uploads
Response:    200 -> list of recent uploads: { job_id, filename, uploaded_at }

This endpoint is a simple, unauthenticated listing of uploads stored
on disk. It is intended for the local development dashboard UI only
and is intentionally lightweight (reads filesystem metadata only).
"""

from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/api/v1", tags=["uploads"])


@router.get("/uploads")
async def list_uploads() -> List[dict]:
    uploads_root: Path = settings.upload_dir
    items: List[dict] = []

    for p in uploads_root.iterdir():
        if not p.is_dir():
            continue
        # find the first file inside the job dir (the original upload)
        files = [f for f in p.iterdir() if f.is_file()]
        filename = files[0].name if files else ""
        # use directory mtime as uploaded time
        mtime = p.stat().st_mtime
        items.append({
            "job_id": p.name,
            "filename": filename,
            "uploaded_at": datetime.fromtimestamp(mtime).isoformat(),
        })

    # sort by uploaded_at desc
    items.sort(key=lambda x: x["uploaded_at"], reverse=True)
    return items
