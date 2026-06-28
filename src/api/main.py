# FastAPI app entry point
"""
src/api/main.py
===============
FastAPI application entry point.

WHAT THIS FILE DOES:
    1. Creates the FastAPI app instance
    2. Registers the router from routes.py
    3. Adds CORS middleware (allows browser requests from any origin)
    4. Starts the server when run directly

HOW TO START THE API:
    cd financial-research-agent
    uvicorn src.api.main:app --reload --port 8000

    --reload → auto-restarts when you change code (dev mode only)
    --port   → run on port 8000

AFTER STARTING:
    API docs (interactive): http://localhost:8000/docs
    Health check:           http://localhost:8000/api/v1/health
    Run pipeline:           POST http://localhost:8000/api/v1/analyze

WHAT IS UVICORN:
    Uvicorn is a fast web server for Python async applications.
    FastAPI needs uvicorn to actually serve HTTP requests.
    Install: pip install uvicorn
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Create FastAPI app ────────────────────────────────────────────────────────
# title and description appear in the auto-generated /docs page
app = FastAPI(
    title       = "Financial Research Agent API",
    description = "Autonomous multi-agent system for generating financial research reports",
    version     = "1.0.0",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows browsers to make requests
# to this API from different domains.
# allow_origins=["*"] → allow requests from any origin (fine for development)
# In production, replace "*" with your specific frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Register routes ───────────────────────────────────────────────────────────
# include_router() adds all endpoints from routes.py to the app.
app.include_router(router)


# ── Startup event ─────────────────────────────────────────────────────────────
# Runs once when the server starts — good place to verify dependencies
@app.on_event("startup")
async def startup():
    logger.info("Financial Research Agent API starting up...")
    logger.info("Docs available at: http://localhost:8000/docs")


# ── Run directly ──────────────────────────────────────────────────────────────
# When you run `python -m src.api.main`, this starts the server.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
    )
