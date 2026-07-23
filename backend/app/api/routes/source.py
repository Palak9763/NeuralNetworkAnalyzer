"""
api/routes/source.py

HTTP Method: GET
Route:       /api/v1/source/{job_id}
Response:    200 -> SourceResponse {job_id, filename, code}

This endpoint returns the source file used to build the graph for a
previously uploaded job. It is used by the frontend dashboard's code
preview panel.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.core.exceptions import ModelParsingError
from app.schemas.graph import SourceResponse
from app.services.upload_service import resolve_candidate_model_file
from app.db.session import get_db
from app.models.graph import SavedGraph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["source"])


@router.get("/source/{job_id}", response_model=SourceResponse, status_code=status.HTTP_200_OK)
async def get_source(job_id: str, db: Session = Depends(get_db)) -> SourceResponse:
    # 1. Query the database first
    try:
        db_graph = db.query(SavedGraph).filter(SavedGraph.job_id == job_id).first()
        if db_graph and db_graph.code:
            logger.info("job_id=%s source found in database cache", job_id)
            return SourceResponse(job_id=job_id, filename=db_graph.filename, code=db_graph.code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database query failed for source job_id=%s: %s", job_id, exc)

    # 2. Resolve on disk if not found in DB
    try:
        source_file = resolve_candidate_model_file(job_id)
    except ModelParsingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        code_text = source_file.read_text(errors="ignore")
        return SourceResponse(job_id=job_id, filename=source_file.name, code=code_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read source file for job_id=%s", job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to read source file for preview.",
        ) from exc
