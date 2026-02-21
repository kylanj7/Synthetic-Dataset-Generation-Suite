"""FastAPI application for the SDGS web interface."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import CORS_ORIGINS, DATA_DIR
from .db.database import init_db
from .services.job_runner import shutdown_runner


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    init_db()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yield
    shutdown_runner()


app = FastAPI(
    title="SDGS Web",
    description="Synthetic Dataset Generation Suite — Web Interface",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
from .routers import auth, datasets, providers, galaxy, sse, settings  # noqa: E402

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(sse.router, prefix="/api/events", tags=["events"])
app.include_router(providers.router, prefix="/api", tags=["providers"])
app.include_router(galaxy.router, prefix="/api/galaxy", tags=["galaxy"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

# Mount frontend static files (built with `npm run build`)
frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
