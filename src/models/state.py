# AgentState — shared whiteboard all agents read/write
"""
src/models/state.py
===================
The shared state passed between every agent in the pipeline.

Think of it as a Google Form with many sections:
  - Orchestrator fills: company_name, ticker, research_plan
  - Research Agent fills: news_summary, sentiment_score, sentiment_label
  - Data Agent fills: current_price, pe_ratio, market_cap ...
  - Filing Agent fills: revenue_breakdown, risk_factors, management_outlook
  - Report Agent fills: final_report

Every agent receives the FULL state, reads what it needs,
adds its own output, and passes the updated state forward.

WHY Annotated[list, operator.add]:
    When Research Agent, Data Agent, and Filing Agent run in PARALLEL,
    they all try to update 'completed_agents' at the same time.
    A normal list would get overwritten.
    Annotated[list, operator.add] tells LangGraph to MERGE (append)
    instead of overwrite — so all three agents' names are collected safely.
"""

import operator
from typing import Optional, Annotated
from pydantic import BaseModel, Field


class AgentState(BaseModel):

    # ──────────────────────────────────────────────────────────────────────────
    # INPUT
    # Set once by the caller before the pipeline starts.
    # Never modified by any agent.
    # ──────────────────────────────────────────────────────────────────────────

    company_name: str = Field(
        ...,
        description="Full company name. e.g. 'NVIDIA Corporation'"
    )
    ticker: str = Field(
        ...,
        description="Stock ticker symbol. e.g. 'NVDA'"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # ORCHESTRATOR OUTPUT
    # Filled by: orchestrator.py
    # Read by: all agents (optional — for context)
    # ──────────────────────────────────────────────────────────────────────────

    research_plan: Optional[list[str]] = Field(
        default=None,
        description="Step-by-step plan created by the orchestrator"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # RESEARCH AGENT OUTPUT
    # Filled by: research_agent.py
    # Read by: report_agent.py
    # ──────────────────────────────────────────────────────────────────────────

    news_summary: Optional[str] = Field(
        default=None,
        description="2-3 sentence summary of recent news themes"
    )
    sentiment_score: Optional[float] = Field(
        default=None,
        description="0.0 = very negative, 1.0 = very positive"
    )
    sentiment_label: Optional[str] = Field(
        default=None,
        description="'positive' | 'neutral' | 'negative'"
    )
    articles_count: Optional[int] = Field(
        default=None,
        description="Number of news articles analyzed"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # DATA AGENT OUTPUT
    # Filled by: data_agent.py (calls yfinance)
    # Read by: report_agent.py
    # ──────────────────────────────────────────────────────────────────────────

    current_price: Optional[float] = Field(
        default=None,
        description="Current stock price in USD"
    )
    market_cap_billions: Optional[float] = Field(
        default=None,
        description="Market capitalization in billions USD"
    )
    pe_ratio: Optional[float] = Field(
        default=None,
        description="Price-to-earnings ratio (trailing)"
    )
    revenue_growth_pct: Optional[float] = Field(
        default=None,
        description="Revenue growth year-over-year in percentage"
    )
    gross_margin_pct: Optional[float] = Field(
        default=None,
        description="Gross margin in percentage"
    )
    week_52_high: Optional[float] = Field(
        default=None,
        description="Highest stock price in the last 52 weeks"
    )
    week_52_low: Optional[float] = Field(
        default=None,
        description="Lowest stock price in the last 52 weeks"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # FILING AGENT OUTPUT
    # Filled by: filing_agent.py (calls SEC EDGAR + RAG pipeline)
    # Read by: report_agent.py
    # ──────────────────────────────────────────────────────────────────────────

    revenue_breakdown: Optional[str] = Field(
        default=None,
        description="Revenue by business segment extracted from 10-K"
    )
    risk_factors: Optional[list[str]] = Field(
        default=None,
        description="Top 3-5 risk factors from SEC filing"
    )
    management_outlook: Optional[str] = Field(
        default=None,
        description="Forward guidance from management discussion section"
    )
    filing_source: Optional[str] = Field(
        default=None,
        description="Citation: form type, date. e.g. 'SEC 10-K · FY2024'"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # REPORT AGENT OUTPUT
    # Filled by: report_agent.py
    # Read by: API, dashboard, email sender
    # ──────────────────────────────────────────────────────────────────────────

    final_report: Optional[str] = Field(
        default=None,
        description="Complete markdown-formatted investment research report"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PIPELINE METADATA
    # Used internally to track progress and errors.
    #
    # Annotated[list, operator.add] → safe parallel writes.
    # When 3 agents run at the same time and each appends their name,
    # LangGraph merges all three instead of overwriting with just one.
    # ──────────────────────────────────────────────────────────────────────────

    completed_agents: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Tracks which agents have finished — used by the router"
    )
    errors: Annotated[list[str], operator.add] = Field(
        default_factory=list,
        description="Non-fatal errors logged during the run"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run this file directly to verify the schema works correctly.
#
# Usage:
#   python src/models/state.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Create initial state (what caller passes in)
    state = AgentState(
        company_name="NVIDIA Corporation",
        ticker="NVDA",
    )
    print("── Initial State ───────────────────────────────")
    print(f"  company_name     : {state.company_name}")
    print(f"  ticker           : {state.ticker}")
    print(f"  news_summary     : {state.news_summary}")   # None
    print(f"  current_price    : {state.current_price}")  # None
    print(f"  completed_agents : {state.completed_agents}")

    # 2. Simulate Research Agent filling its fields
    state = state.model_copy(update={
        "news_summary":    "NVIDIA continues strong momentum in AI chips.",
        "sentiment_score": 0.78,
        "sentiment_label": "positive",
        "articles_count":  12,
        "completed_agents": ["research_agent"],
    })
    print("\n── After Research Agent ────────────────────────")
    print(f"  news_summary     : {state.news_summary}")
    print(f"  sentiment_label  : {state.sentiment_label}")
    print(f"  completed_agents : {state.completed_agents}")

    # 3. Simulate Data Agent filling its fields
    state = state.model_copy(update={
        "current_price":      321.50,
        "market_cap_billions": 786.0,
        "pe_ratio":            38.5,
        "revenue_growth_pct":  122.4,
        "completed_agents":   ["data_agent"],
    })
    print("\n── After Data Agent ────────────────────────────")
    print(f"  current_price    : ${state.current_price}")
    print(f"  pe_ratio         : {state.pe_ratio}x")
    print(f"  completed_agents : {state.completed_agents}")

    # 4. Simulate Filing Agent filling its fields
    state = state.model_copy(update={
        "revenue_breakdown":  "Data Center 83%, Gaming 11%, Other 6%",
        "risk_factors":       ["Export controls", "Competition", "Supply chain"],
        "management_outlook": "Management expects continued AI-driven demand.",
        "filing_source":      "SEC 10-K · FY2024",
        "completed_agents":   ["filing_agent"],
    })
    print("\n── After Filing Agent ──────────────────────────")
    print(f"  revenue_breakdown : {state.revenue_breakdown}")
    print(f"  risk_factors      : {state.risk_factors}")
    print(f"  completed_agents  : {state.completed_agents}")

    # 5. Show final state is ready for Report Agent
    print("\n── Ready for Report Agent ──────────────────────")
    all_done = {"research_agent", "data_agent", "filing_agent"}
    is_ready = all_done.issubset(set(state.completed_agents))
    print(f"  All parallel agents done? {is_ready}")
    print(f"  final_report: {state.final_report}")  # still None — Report Agent fills this
    print()