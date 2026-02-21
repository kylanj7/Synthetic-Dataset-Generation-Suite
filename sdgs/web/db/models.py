"""SQLAlchemy ORM models for the SDGS web app."""
import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(200), nullable=False)
    encryption_key_salt = Column(String(64), nullable=False)  # hex-encoded
    hf_token = Column(Text, nullable=True)  # encrypted
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(500), nullable=False)
    topic = Column(String(500), nullable=False)
    status = Column(String(20), nullable=False, default="pending")

    provider = Column(String(100), nullable=True)
    model = Column(String(200), nullable=True)

    target_size = Column(Integer, default=100)
    actual_size = Column(Integer, default=0)

    valid_count = Column(Integer, default=0)
    invalid_count = Column(Integer, default=0)
    healed_count = Column(Integer, default=0)

    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    gpu_kwh = Column(Float, default=0.0)
    duration_seconds = Column(Float, default=0.0)

    output_path = Column(String(500), nullable=True)
    citations_path = Column(String(500), nullable=True)
    system_prompt = Column(Text, nullable=True)
    temperature = Column(Float, default=0.7)

    hf_repo = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="datasets")
    papers = relationship("Paper", back_populates="dataset", cascade="all, delete-orphan")
    qa_pairs = relationship("QAPair", back_populates="dataset", cascade="all, delete-orphan")


class Paper(Base):
    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String(200), nullable=True, index=True)
    title = Column(String(1000), nullable=False)
    authors = Column(JSON, default=list)
    abstract = Column(Text, nullable=True)
    year = Column(Integer, nullable=True)
    doi = Column(String(200), nullable=True)
    url = Column(String(500), nullable=True)
    source = Column(String(50), nullable=True)

    citation_count = Column(Integer, default=0)
    has_full_text = Column(Boolean, default=False)
    keywords = Column(JSON, default=list)
    qa_pair_count = Column(Integer, default=0)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    dataset = relationship("Dataset", back_populates="papers")
    qa_pairs = relationship("QAPair", back_populates="paper", cascade="all, delete-orphan")


class QAPair(Base):
    __tablename__ = "qa_pairs"

    id = Column(Integer, primary_key=True, index=True)
    instruction = Column(Text, nullable=False)
    output = Column(Text, nullable=False)
    is_valid = Column(Boolean, default=True)
    think_text = Column(Text, nullable=True)
    answer_text = Column(Text, nullable=True)

    source_paper_id = Column(String(200), nullable=True)
    source_title = Column(String(1000), nullable=True)
    was_healed = Column(Boolean, default=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=True)
    paper = relationship("Paper", back_populates="qa_pairs")
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    dataset = relationship("Dataset", back_populates="qa_pairs")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provider_name = Column(String(100), nullable=False)
    encrypted_key = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        UniqueConstraint("user_id", "provider_name", name="uq_user_provider"),
    )
