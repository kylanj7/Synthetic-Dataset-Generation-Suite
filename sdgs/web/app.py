"""FastAPI application for the SDGS web interface."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# SPA frontend: serve static assets, fallback to index.html for client-side routing
frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    # Serve static assets (JS, CSS, images) directly
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static-assets")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve index.html for all non-API routes (SPA fallback)."""
        # Try to serve the exact file first (e.g., favicon.ico)
        file_path = frontend_dist / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for client-side routing
        return FileResponse(frontend_dist / "index.html")
