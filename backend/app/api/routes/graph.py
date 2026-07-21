"""
api/routes/graph.py

HTTP Method: GET
Route:       /api/v1/graph/{job_id}
Request:     job_id as a path parameter (string, returned by POST /upload)
Response:    200 -> UniversalGraph (see schemas/graph.py for full shape)
Status Codes:
    200 - graph successfully parsed and returned
    404 - job_id does not exist (no such upload)
    422 - framework detected but not supported in this phase
    500 - parsing failed on both torch.fx and AST tiers

Why this file exists:
    Thin HTTP adapter over services/parser_service. Synchronous in Phase 1
    (no job queue yet) - the parse runs inline on this request. This is
    intentional: Phase 1 explicitly defers Celery/Redis until uploads are
    demonstrably slow enough to need background processing (see Phase 7
    in the project roadmap).

How it connects:
    Depends on services/upload_service.resolve_candidate_model_file() to
    find the file, then services/parser_service.parse_project() to
    produce the UniversalGraph. Consumed by the frontend's GraphCanvas
    component via polling.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import FrameworkNotSupportedError, ModelParsingError
from app.schemas.graph import UniversalGraph
from app.services.parser_service import parse_project
from app.services.upload_service import resolve_candidate_model_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["graph"])


@router.get("/graph/{job_id}", response_model=UniversalGraph, status_code=status.HTTP_200_OK)
async def get_graph(job_id: str) -> UniversalGraph:
    try:
        model_file = resolve_candidate_model_file(job_id)
    except ModelParsingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        return parse_project(job_id=job_id, model_file=model_file)
    except FrameworkNotSupportedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelParsingError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error parsing job_id=%s", job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while parsing the model.",
        ) from exc
