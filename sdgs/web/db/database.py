"""SQLAlchemy engine and session setup."""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from ..config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db():
    """Create all tables and run lightweight migrations."""
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate(engine)


def _migrate(eng):
    """Add missing columns to existing tables (lightweight migration)."""
    insp = inspect(eng)
    if insp.has_table("papers"):
        columns = {c["name"] for c in insp.get_columns("papers")}
        if "pdf_path" not in columns:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE papers ADD COLUMN pdf_path VARCHAR(500)"))
    if insp.has_table("datasets"):
        columns = {c["name"] for c in insp.get_columns("datasets")}
        if "max_tokens" not in columns:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE datasets ADD COLUMN max_tokens INTEGER"))
    if insp.has_table("evaluation_runs"):
        columns = {c["name"] for c in insp.get_columns("evaluation_runs")}
        if "correction_json" not in columns:
            with eng.begin() as conn:
                conn.execute(text("ALTER TABLE evaluation_runs ADD COLUMN correction_json JSON"))


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
