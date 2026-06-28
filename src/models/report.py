# ResearchReport — Pydantic schema for the final report output
"""
src/models/report.py
====================
Defines the structure of the final research report.

This is what the Report Agent produces at the end of the pipeline.
Instead of a plain string, the report is a structured Pydantic object
with typed, validated fields — every section clearly defined.

3 classes are defined here:

    KeyMetrics        → groups all financial numbers
    SentimentSummary  → groups all sentiment info
    ResearchReport    → the full report (contains the two above)

HOW IT CONNECTS TO state.py:
    state.py has:   final_report: Optional[str] = None
    In the real pipeline this becomes: Optional[ResearchReport] = None
    Report Agent fills it. API and dashboard read it.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1 — KeyMetrics
# Groups all financial numbers in one place.
# Data Agent fills the AgentState fields.
# Report Agent maps them into this object.
# ─────────────────────────────────────────────────────────────────────────────

class KeyMetrics(BaseModel):
    """
    All financial numbers for the company.
    Every field is Optional because not every ticker
    has every metric available in yfinance.
    """

    current_price: Optional[float] = Field(
        default=None,
        description="Current stock price in USD"
    )
    market_cap_billions: Optional[float] = Field(
        default=None,
        description="Market cap in billions. e.g. 786.0 means $786 billion"
    )
    pe_ratio: Optional[float] = Field(
        default=None,
        description="Price-to-earnings ratio. How much investors pay per $1 of earnings"
    )
    revenue_growth_pct: Optional[float] = Field(
        default=None,
        description="Revenue growth year-over-year in percentage. e.g. 122.4 means +122.4%"
    )
    gross_margin_pct: Optional[float] = Field(
        default=None,
        description="Gross margin in percentage. e.g. 55.4 means 55.4%"
    )
    week_52_high: Optional[float] = Field(
        default=None,
        description="Highest stock price in the last 52 weeks"
    )
    week_52_low: Optional[float] = Field(
        default=None,
        description="Lowest stock price in the last 52 weeks"
    )
    data_source: str = Field(
        default="Yahoo Finance via yfinance",
        description="Where this data came from — always cite the source"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 — SentimentSummary
# Groups all news sentiment information in one place.
# Research Agent fills the AgentState fields.
# Report Agent maps them into this object.
# ─────────────────────────────────────────────────────────────────────────────

class SentimentSummary(BaseModel):
    """
    News sentiment analysis results.
    """

    score: float = Field(
        ...,
        ge=0.0,    # ge = greater than or equal to → score cannot be below 0.0
        le=1.0,    # le = less than or equal to    → score cannot be above 1.0
        description="Sentiment score. 0.0 = very negative, 1.0 = very positive"
    )
    label: str = Field(
        ...,
        description="Human readable label: 'positive' | 'neutral' | 'negative'"
    )
    summary: str = Field(
        ...,
        description="2-3 sentence summary of the key news themes"
    )
    articles_count: int = Field(
        default=0,
        description="Number of news articles analyzed to produce this sentiment"
    )
    data_source: str = Field(
        default="Tavily News API",
        description="Where the news articles came from"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3 — ResearchReport
# The main report object. Contains KeyMetrics and SentimentSummary.
# This is what gets emailed, shown on dashboard, and returned by the API.
# ─────────────────────────────────────────────────────────────────────────────

class ResearchReport(BaseModel):
    """
    The complete investment research report.

    Produced by the Report Agent at the end of the pipeline.
    Every field is required (marked with ...) except where
    explicitly marked Optional — forcing the agent to fill
    everything before the report is considered complete.
    """

    # ── Header ────────────────────────────────────────────────────────────────
    # Basic info about the report itself

    company_name: str = Field(
        ...,
        description="Full company name. e.g. 'NVIDIA Corporation'"
    )
    ticker: str = Field(
        ...,
        description="Stock ticker. e.g. 'NVDA'"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this report was generated. Auto-filled."
    )
    data_sources: list[str] = Field(
        default_factory=lambda: ["SEC EDGAR", "Yahoo Finance", "Tavily News"],
        description="All data sources used in this report"
    )

    # ── Section 1: Executive Summary ──────────────────────────────────────────
    # First thing a reader sees. 3-4 sentences max.
    # A busy executive should understand the key takeaway from this alone.

    executive_summary: str = Field(
        ...,
        description="3-4 sentence high-level summary of the company and investment thesis"
    )

    # ── Section 2: Key Metrics ────────────────────────────────────────────────
    # All financial numbers grouped together using KeyMetrics class above.
    # Written as: key_metrics: KeyMetrics (not a flat list of fields)
    # This makes the JSON output nested and readable.

    key_metrics: KeyMetrics = Field(
        ...,
        description="All financial metrics from Yahoo Finance"
    )

    # ── Section 3: Sentiment ──────────────────────────────────────────────────
    # News sentiment grouped using SentimentSummary class above.

    sentiment: SentimentSummary = Field(
        ...,
        description="News sentiment analysis from Tavily"
    )

    # ── Section 4: SEC Filing Insights ────────────────────────────────────────
    # What the Filing Agent extracted from the 10-K via RAG.
    # Must include citation — which filing, which year.

    filing_insights: str = Field(
        ...,
        description="Key findings from the SEC 10-K filing with citations"
    )

    # ── Section 5: Risk Factors ───────────────────────────────────────────────
    # List of risks. Each risk is a dict with title and description.
    # Max 5 — more than 5 dilutes the signal and loses the reader.

    risk_factors: list[dict] = Field(
        ...,
        max_length=5,
        description="Top 3-5 risks. Each is a dict with 'title' and 'description' keys"
    )

    # ── Section 6: Recommendation ─────────────────────────────────────────────
    # BUY, HOLD, or SELL — with a one sentence reason.
    # This is what most readers skip to first.

    recommendation: str = Field(
        ...,
        description="BUY | HOLD | SELL followed by a one sentence rationale"
    )

    # ── Section 7: Gaps and Limitations ──────────────────────────────────────
    # What the agent COULD NOT verify.
    # This is non-negotiable in finance — honesty about limitations
    # is what separates a trustworthy report from a dangerous one.

    gaps_and_limitations: list[str] = Field(
        ...,
        description="List of what this report could not verify or is uncertain about"
    )

    # ── Section 8: Disclaimer ─────────────────────────────────────────────────
    # Legal requirement. Always included. Never removed.

    disclaimer: str = Field(
        default=(
            "This report is generated by an AI system for informational "
            "purposes only and does not constitute financial advice. "
            "Always consult a qualified financial advisor before making "
            "investment decisions."
        ),
        description="Legal disclaimer — always present in every report"
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to verify the schema works.
#
# Usage:
#   python src/models/report.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Build a sample report — exactly what Report Agent will produce
    report = ResearchReport(
        company_name="NVIDIA Corporation",
        ticker="NVDA",

        executive_summary=(
            "NVIDIA continues to dominate the AI chip market with exceptional "
            "revenue growth driven by data center demand. News sentiment is "
            "strongly positive. Export control risks remain the key headwind."
        ),

        key_metrics=KeyMetrics(
            current_price=321.50,
            market_cap_billions=786.0,
            pe_ratio=38.5,
            revenue_growth_pct=122.4,
            gross_margin_pct=55.4,
            week_52_high=346.40,
            week_52_low=173.20,
        ),

        sentiment=SentimentSummary(
            score=0.78,
            label="positive",
            summary=(
                "News coverage is predominantly positive, driven by strong "
                "Blackwell GPU demand and new cloud partnerships. "
                "Export control headlines add minor negative sentiment."
            ),
            articles_count=12,
        ),

        filing_insights=(
            "Data Center segment accounts for 83% of total revenue ($47.5B). "
            "Gaming declined 12% YoY. Management expects AI-driven demand to "
            "continue accelerating globally. [Source: SEC 10-K FY2024, Page 42]"
        ),

        risk_factors=[
            {
                "title": "Export Controls",
                "description": "US restrictions on chip exports to China could reduce addressable market.",
            },
            {
                "title": "Competition",
                "description": "AMD MI300 and custom Google TPUs gaining enterprise traction.",
            },
            {
                "title": "Valuation",
                "description": "P/E of 38.5x assumes continued hypergrowth — any slowdown compresses multiples.",
            },
        ],

        recommendation=(
            "HOLD — Strong fundamentals but current valuation leaves limited "
            "margin of safety. Monitor export control developments closely."
        ),

        gaps_and_limitations=[
            "Insider trading data not fetched — requires SEC Form 4 scraping",
            "Competitor analysis not included in this run",
            "Sentiment based on 12 articles — broader coverage would improve accuracy",
        ],
    )

    # ── Print each section ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  {report.company_name} ({report.ticker})")
    print(f"  Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')} UTC")
    print("=" * 60)

    print("\n── Executive Summary ───────────────────────────────────")
    print(f"  {report.executive_summary}")

    print("\n── Key Metrics ─────────────────────────────────────────")
    m = report.key_metrics
    print(f"  Price          : ${m.current_price}")
    print(f"  Market Cap     : ${m.market_cap_billions}B")
    print(f"  P/E Ratio      : {m.pe_ratio}x")
    print(f"  Revenue Growth : {m.revenue_growth_pct}%")
    print(f"  Gross Margin   : {m.gross_margin_pct}%")
    print(f"  52W High/Low   : ${m.week_52_high} / ${m.week_52_low}")
    print(f"  Source         : {m.data_source}")

    print("\n── Sentiment ────────────────────────────────────────────")
    s = report.sentiment
    print(f"  Label          : {s.label.upper()}")
    print(f"  Score          : {s.score:.0%}")
    print(f"  Articles       : {s.articles_count}")
    print(f"  Summary        : {s.summary}")

    print("\n── SEC Filing Insights ──────────────────────────────────")
    print(f"  {report.filing_insights}")

    print("\n── Risk Factors ─────────────────────────────────────────")
    for i, risk in enumerate(report.risk_factors, 1):
        print(f"  {i}. {risk['title']}: {risk['description']}")

    print("\n── Recommendation ───────────────────────────────────────")
    print(f"  {report.recommendation}")

    print("\n── Gaps and Limitations ─────────────────────────────────")
    for gap in report.gaps_and_limitations:
        print(f"  • {gap}")

    print("\n── Disclaimer ───────────────────────────────────────────")
    print(f"  {report.disclaimer}")

    print("\n── Pydantic Validation Test ─────────────────────────────")
    # Test that Pydantic catches an invalid sentiment score
    try:
        bad_sentiment = SentimentSummary(
            score=1.8,        # invalid — must be between 0.0 and 1.0
            label="positive",
            summary="test",
        )
    except Exception as e:
        print(f"  ✅ Caught invalid score correctly: {e.errors()[0]['msg']}")

    print("\n── JSON Output (first 300 chars) ────────────────────────")
    print(f"  {report.model_dump_json()[:300]}...")
    print()