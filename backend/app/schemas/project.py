from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the project")
    description: Optional[str] = Field(default=None, max_length=500, description="Optional project description")

class ProjectResponse(BaseModel):
    project_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    graphs_count: int = 0

    class Config:
        from_attributes = True
