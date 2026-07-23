import datetime
import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base

class Project(Base):
    __tablename__ = "projects"

    project_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    owner_id = Column(String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationship pointing to SavedGraph records
    graphs = relationship("SavedGraph", back_populates="project", cascade="all, delete-orphan")

    # Relationship back to User
    owner = relationship("User", back_populates="projects")
