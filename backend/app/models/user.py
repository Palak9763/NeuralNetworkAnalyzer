import datetime
import uuid
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # nullable for Google OAuth users
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # OAuth fields
    auth_provider = Column(String, default="local")  # "local" or "google"
    google_id = Column(String, unique=True, nullable=True, index=True)

    # Relationship to Project ORM model
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
