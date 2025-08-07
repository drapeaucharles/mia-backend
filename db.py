from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/mia_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Business(Base):
    __tablename__ = "businesses"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    contact_email = Column(String(255))
    contact_phone = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    chat_logs = relationship("ChatLog", back_populates="business")

class ChatLog(Base):
    __tablename__ = "chat_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    message = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime(timezone=True), nullable=False)
    business_id = Column(Integer, ForeignKey("businesses.id"), nullable=True)
    
    # Relationships
    business = relationship("Business", back_populates="chat_logs")

class Miner(Base):
    __tablename__ = "miners"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    auth_key = Column(String(255), nullable=False, unique=True)
    job_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class IdleJob(Base):
    __tablename__ = "idle_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    submitted_by = Column(String(255), nullable=False)  # API key or identifier
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    output_tokens = Column(Integer, nullable=True)
    usd_earned = Column(Float, nullable=True)
    result = Column(Text, nullable=True)
    runpod_job_id = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(255), nullable=False, unique=True)
    value = Column(Float, default=0.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class GolemJob(Base):
    __tablename__ = "golem_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    miner_name = Column(String(255), nullable=False, index=True)
    duration_sec = Column(Integer, nullable=False)
    estimated_glm = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    
    # Initialize system metrics
    db = SessionLocal()
    try:
        # Check if metrics exist
        metrics_to_init = [
            ("runpod_income_usd", 0.0),
            ("total_idle_jobs_processed", 0.0),
            ("total_buyback_usd", 0.0),
            ("last_buyback_timestamp", 0.0)
        ]
        
        for metric_name, default_value in metrics_to_init:
            existing = db.query(SystemMetrics).filter(
                SystemMetrics.metric_name == metric_name
            ).first()
            
            if not existing:
                metric = SystemMetrics(
                    metric_name=metric_name,
                    value=default_value
                )
                db.add(metric)
        
        db.commit()
    finally:
        db.close()