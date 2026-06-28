# Pydantic schemas for API request and response bodies
"""
src/api/schemas.py
==================
Pydantic schemas for API request and response bodies.

WHAT IS PYDANTIC:
    Pydantic is a Python library for data validation.
    When FastAPI receives a request, it uses these schemas to:
    1. Validate the incoming data (is ticker a string? is it present?)
    2. Convert types automatically (string "222.14" → float 222.14)
    3. Return clear errors if data is invalid

WHY SEPARATE SCHEMAS FROM AGENTSTATE:
    AgentState is our internal pipeline whiteboard — it has 20+ fields
    that are only relevant internally (completed_agents, errors, etc.)
    The API schemas are the public interface — clean, minimal, documented.
    We never expose AgentState directly to the outside world.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA
# What the caller must send to trigger the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Request body for POST /analyze.

    Example:
        {
            "company_name": "NVIDIA Corporation",
            "ticker": "NVDA"
        }
    """
    company_name: str = Field(
        ...,
        description="Full company name",
        example="NVIDIA Corporation"
    )
    ticker: str = Field(
        ...,
        description="Stock ticker symbol",
        example="NVDA"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE SCHEMA
# What the API returns after the pipeline completes.
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    """
    Response body for POST /analyze.

    Contains the final report plus key metrics extracted during the run.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    company_name: str
    ticker:       str

    # ── Final report ──────────────────────────────────────────────────────────
    final_report: Optional[str] = Field(
        default=None,
        description="Complete markdown-formatted research report"
    )

    # ── News sentiment (from research_agent) ──────────────────────────────────
    sentiment_label: Optional[str] = Field(
        default=None,
        description="positive | neutral | negative"
    )
    sentiment_score: Optional[float] = Field(
        default=None,
        description="0.0 = very negative, 1.0 = very positive"
    )
    news_summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence summary of recent news themes"
    )

    # ── Stock metrics (from data_agent) ───────────────────────────────────────
    current_price:       Optional[float] = None
    market_cap_billions: Optional[float] = None
    pe_ratio:            Optional[float] = None
    revenue_growth_pct:  Optional[float] = None
    gross_margin_pct:    Optional[float] = None

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    completed_agents: Optional[list] = Field(
        default=None,
        description="List of agents that completed successfully"
    )
    errors: Optional[list] = Field(
        default=None,
        description="Non-fatal errors that occurred during the run"
    )
    filing_source: Optional[str] = Field(
        default=None,
        description="Source citation for SEC filing data"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status:  str = "ok"
    message: str = "Financial Research Agent API is running"
