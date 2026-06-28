# POST /analyze and GET /report/{id} endpoints
"""
src/api/routes.py
=================
API route handlers — defines what happens when each endpoint is called.

ENDPOINTS:
    GET  /health   → checks if API is running
    POST /analyze  → runs the full pipeline, returns the report

WHY SEPARATE ROUTES FROM MAIN:
    main.py starts the server and registers routers.
    routes.py defines what each endpoint actually does.
    Keeping them separate makes it easy to add new route files later
    (e.g. a /history router, a /compare router).
"""

import time
from fastapi import APIRouter, HTTPException
from src.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from src.pipeline.graph import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)

# APIRouter groups related endpoints together.
# The prefix "/api/v1" means all routes here are at /api/v1/health, /api/v1/analyze etc.
router = APIRouter(prefix="/api/v1")


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# Simple health check — confirms the API is running.
# Used by monitoring systems to check if the service is alive.
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the API is running correctly.
    """
    return HealthResponse(
        status  = "ok",
        message = "Financial Research Agent API is running"
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze
# The main endpoint — triggers the full pipeline and returns the report.
#
# WHY async:
#   The pipeline takes 30-60 seconds to run (LLM calls, API calls).
#   async lets FastAPI handle other requests while this one is running.
#   Without async, the server would be blocked for 60 seconds per request.
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run the full research pipeline for a company and return the report.

    FLOW:
        1. Validate request (FastAPI does this automatically via schema)
        2. Build and run the LangGraph pipeline
        3. Extract results from pipeline output
        4. Return structured response

    Args:
        request: AnalyzeRequest with company_name and ticker

    Returns:
        AnalyzeResponse with final report and all key metrics

    Raises:
        HTTPException 500 if pipeline fails completely
    """
    logger.info(
        f"POST /analyze — "
        f"company={request.company_name}, ticker={request.ticker}"
    )

    start_time = time.time()

    try:
        # ── Build and run the pipeline ────────────────────────────────────────
        # build_graph() creates a fresh compiled LangGraph pipeline.
        # .invoke() runs the full pipeline synchronously and returns
        # the final AgentState as a dict.
        pipeline = build_graph()

        result = pipeline.invoke({
            "company_name": request.company_name,
            "ticker":       request.ticker,
        })

        elapsed = round(time.time() - start_time, 1)
        logger.info(
            f"Pipeline complete in {elapsed}s — "
            f"agents={result.get('completed_agents')}"
        )

        # ── Check if report was generated ─────────────────────────────────────
        # If report_agent failed to generate a report, return 500
        if not result.get("final_report"):
            raise HTTPException(
                status_code = 500,
                detail      = "Pipeline completed but report generation failed"
            )

        # ── Build and return response ─────────────────────────────────────────
        # Map from AgentState fields to AnalyzeResponse fields
        return AnalyzeResponse(
            company_name        = request.company_name,
            ticker              = request.ticker.upper(),
            final_report        = result.get("final_report"),
            sentiment_label     = result.get("sentiment_label"),
            sentiment_score     = result.get("sentiment_score"),
            news_summary        = result.get("news_summary"),
            current_price       = result.get("current_price"),
            market_cap_billions = result.get("market_cap_billions"),
            pe_ratio            = result.get("pe_ratio"),
            revenue_growth_pct  = result.get("revenue_growth_pct"),
            gross_margin_pct    = result.get("gross_margin_pct"),
            completed_agents    = result.get("completed_agents"),
            errors              = result.get("errors"),
            filing_source       = result.get("filing_source"),
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Pipeline failed: {str(e)}"
        )
