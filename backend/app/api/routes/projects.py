import logging
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectResponse
from app.schemas.graph import UniversalGraph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(project_in: ProjectCreate, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        new_project = Project(
            name=project_in.name,
            description=project_in.description,
        )
        db.add(new_project)
        db.commit()
        db.refresh(new_project)
        logger.info("Created new project: %s (id=%s)", new_project.name, new_project.project_id)
        return ProjectResponse(
            project_id=new_project.project_id,
            name=new_project.name,
            description=new_project.description,
            created_at=new_project.created_at,
            graphs_count=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create project")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create project.",
        ) from exc


@router.get("", response_model=list[ProjectResponse])
async def list_projects(db: Session = Depends(get_db)) -> list[ProjectResponse]:
    try:
        projects = db.query(Project).all()
        return [
            ProjectResponse(
                project_id=p.project_id,
                name=p.name,
                description=p.description,
                created_at=p.created_at,
                graphs_count=len(p.graphs),
            )
            for p in projects
        ]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list projects")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve projects list.",
        ) from exc


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)) -> ProjectResponse:
    p = db.query(Project).filter(Project.project_id == project_id).first()
    if not p:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return ProjectResponse(
        project_id=p.project_id,
        name=p.name,
        description=p.description,
        created_at=p.created_at,
        graphs_count=len(p.graphs),
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    p = db.query(Project).filter(Project.project_id == project_id).first()
    if not p:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    try:
        db.delete(p)
        db.commit()
        logger.info("Deleted project id=%s (graphs cascade-deleted)", project_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to delete project")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not delete project.",
        ) from exc


@router.get("/{project_id}/graphs", response_model=list[UniversalGraph])
async def list_project_graphs(project_id: str, db: Session = Depends(get_db)) -> list[UniversalGraph]:
    p = db.query(Project).filter(Project.project_id == project_id).first()
    if not p:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    return [
        UniversalGraph(
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
        for db_graph in p.graphs
    ]
