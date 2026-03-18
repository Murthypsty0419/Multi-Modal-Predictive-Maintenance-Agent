"""
SQLAlchemy models for PostgreSQL + pgvector.

Tables:
  - text_chunks    : RAG document chunks with vector embeddings
  - inference_logs : Every prediction persisted for audit / analytics
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class TextChunkRow(Base):
    __tablename__ = "text_chunks"

    chunk_id = Column(String(64), primary_key=True)
    source_document = Column(String(512), nullable=False)
    section = Column(String(256), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # BGE-M3 dim = 1024
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime, server_default=func.now())


class InferenceLogRow(Base):
    __tablename__ = "inference_logs"

    log_id = Column(String(64), primary_key=True)
    request_id = Column(String(64), nullable=False, index=True)
    pump_id = Column(String(64), nullable=False, index=True)
    risk_level = Column(String(16), nullable=False)
    failure_probability = Column(Float, nullable=False)
    explanation = Column(Text, nullable=False, default="")
    chain_of_thought = Column(Text, nullable=False, default="")
    modality_contributions = Column(JSONB, default=dict)
    sensor_payload = Column(JSONB, nullable=True)
    rag_payload = Column(JSONB, nullable=True)
    vision_payload = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # --- Asset Life & Maintenance Tables ---
class ServiceScheduleRow(Base):
    __tablename__ = "service_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pump_id = Column(String(64), nullable=False, index=True)
    task_name = Column(String(128), nullable=False)
    interval_hours = Column(Integer, nullable=False)
    priority = Column(Integer, nullable=False, default=1)  # 3=Critical, 1=Routine


class WorkDoneLogRow(Base):
    __tablename__ = "work_done_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pump_id = Column(String(64), nullable=False, index=True)
    task_name = Column(String(128), nullable=False)
    hours_at_service = Column(Integer, nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)


class MaintenanceRequestRow(Base):
    __tablename__ = "maintenance_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pump_id = Column(String(64), nullable=False, index=True)
    description = Column(Text, nullable=False)
    priority = Column(String(16), nullable=False)  # e.g., 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    status = Column(String(16), nullable=False)    # e.g., 'OPEN', 'CLOSED'
    created_at = Column(DateTime, server_default=func.now(), nullable=False)