"""
api/routes/health.py

HTTP Method: GET
Route:       /api/v1/health
Response:    200 -> {"status": "ok"}

Why this file exists:
    Standard liveness endpoint for load balancers, uptime monitors, and
    Docker healthchecks in later deployment phases.
"""

from fastapi import APIRouter, status

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    return {"status": "ok"}
