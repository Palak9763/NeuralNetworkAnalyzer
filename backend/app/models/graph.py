import datetime
from sqlalchemy import Column, String, Integer, BigInteger, DateTime, JSON, Text
from app.db.session import Base

class SavedGraph(Base):
    __tablename__ = "saved_graphs"

    job_id = Column(String, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    confidence = Column(String, nullable=False)
    total_params = Column(Integer, default=0)
    total_layers = Column(Integer, default=0)
    flops = Column(BigInteger, nullable=True)
    warnings = Column(JSON, default=list)
    nodes = Column(JSON, default=list)
    edges = Column(JSON, default=list)
    groups = Column(JSON, default=list)
    filename = Column(String, nullable=False)
    code = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
