"""Configuration for the SDGS web application."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"

DB_PATH = os.environ.get("SDGS_DB_PATH", str(BASE_DIR / "sdgs_web.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

CORS_ORIGINS = os.environ.get(
    "SDGS_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")

HOST = os.environ.get("SDGS_HOST", "0.0.0.0")
PORT = int(os.environ.get("SDGS_PORT", "8000"))

JWT_SECRET = os.environ.get("SDGS_JWT_SECRET", "")
