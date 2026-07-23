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
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.core.exceptions import FrameworkNotSupportedError, ModelParsingError
from app.schemas.graph import UniversalGraph
from app.services.parser_service import parse_project
from app.services.upload_service import resolve_candidate_model_file
from app.db.session import get_db
from app.models.graph import SavedGraph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["graph"])


@router.get("/graph/{job_id}", response_model=UniversalGraph, status_code=status.HTTP_200_OK)
async def get_graph(job_id: str, db: Session = Depends(get_db)) -> UniversalGraph:
    # 1. Query the database first
    try:
        db_graph = db.query(SavedGraph).filter(SavedGraph.job_id == job_id).first()
        if db_graph:
            logger.info("job_id=%s found in database cache", job_id)
            return UniversalGraph(
                job_id=db_graph.job_id,
                model_name=db_graph.model_name,
                meta={
                    "framework": db_graph.framework,
                    "confidence": db_graph.confidence,
                    "total_params": db_graph.total_params,
                    "total_layers": db_graph.total_layers,
                    "flops": db_graph.flops,
                    "warnings": db_graph.warnings,
                },
                nodes=db_graph.nodes,
                edges=db_graph.edges,
                groups=db_graph.groups,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database query failed for job_id=%s, parsing manually: %s", job_id, exc)

    # 2. Parse from disk if not found in DB
    try:
        model_file = resolve_candidate_model_file(job_id)
    except ModelParsingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        graph = parse_project(job_id=job_id, model_file=model_file)
        
        # Read source code text to persist it in DB
        code_text = ""
        try:
            code_text = model_file.read_text(errors="ignore")
        except Exception:
            pass

        # Save to DB cache
        try:
            new_db_graph = SavedGraph(
                job_id=graph.job_id,
                model_name=graph.model_name,
                framework=graph.meta.framework,
                confidence=graph.meta.confidence,
                total_params=graph.meta.total_params,
                total_layers=graph.meta.total_layers,
                flops=graph.meta.flops,
                warnings=graph.meta.warnings,
                nodes=[n.model_dump() for n in graph.nodes],
                edges=[e.model_dump() for e in graph.edges],
                groups=[g.model_dump() for g in graph.groups],
                filename=model_file.name,
                code=code_text,
            )
            db.add(new_db_graph)
            db.commit()
            logger.info("job_id=%s persisted to database", job_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist job_id=%s to database: %s", job_id, exc)

        return graph

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
