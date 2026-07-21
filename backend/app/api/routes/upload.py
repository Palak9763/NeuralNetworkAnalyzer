"""
api/routes/upload.py

HTTP Method: POST
Route:       /api/v1/upload
Request:     multipart/form-data with a single file field named "file"
             (.py or .zip, max size configured in core/config.py)
Response:    200 -> UploadResponse {job_id, filename, status}
Status Codes:
    200 - upload accepted and saved
    400 - invalid file type or file too large
    500 - unexpected server error while saving

Why this file exists:
    Thin HTTP adapter. Its only job is translating between FastAPI's
    UploadFile and the pure-Python upload_service, and mapping domain
    exceptions to HTTP status codes. No business logic lives here.

How it connects:
    Delegates to services/upload_service.handle_upload(). Depended on by
    the frontend's upload page (POST before polling GET /graph/{job_id}).
"""

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.exceptions import FileTooLargeError, InvalidFileTypeError
from app.schemas.graph import UploadResponse
from app.services.upload_service import handle_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_project(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()

    try:
        job_id = handle_upload(filename=file.filename, content=content)
    except InvalidFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error handling upload")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while saving the upload.",
        ) from exc

    return UploadResponse(job_id=job_id, filename=file.filename)
