# Synthesizes all outputs into final report and delivers it
"""
src/agents/report_agent.py
==========================
Agent 4 of 4: Synthesizes all agent outputs into a final research report.

WHAT IT READS FROM AGENTSTATE:
    From research_agent → news_summary, sentiment_score, sentiment_label
    From data_agent     → current_price, market_cap_billions, pe_ratio,
                          revenue_growth_pct, gross_margin_pct, week_52_high, week_52_low
    From filing_agent   → revenue_breakdown, risk_factors, management_outlook

WHAT IT WRITES:
    final_report → complete markdown-formatted research report string

FLOW:
    1. Validate all required data is present in state
    2. Build a structured prompt with all data as context
    3. Send to llama3 → generates full report
    4. Write report to state.final_report
    5. Send email via SendGrid (if configured)

TEMPERATURE 0.3:
    Higher than other agents because this is writing, not extraction.
    We want professional, natural-sounding prose not robotic output.
    0.3 gives consistency with readable variation.

HOW TO RUN:
    cd financial-research-agent
    python -m src.agents.report_agent
"""

from PIL import report

from src.utils.llm import call_llm
from src.models.state import AgentState
from src.tools.email_sender import send_report_email
from src.utils.logger import get_logger

logger = get_logger(__name__)




# ─────────────────────────────────────────────────────────────────────────────
# AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def report_agent(state: AgentState) -> dict:
    """
    Synthesize all agent outputs into a final research report and email it.

    Args:
        state: Full AgentState with outputs from all 3 previous agents

    Returns:
        dict with final_report, completed_agents, errors
    """
    logger.info(
        f"Report Agent starting for "
        f"{state.company_name} ({state.ticker})"
    )

    errors = []

    # ── Step 1: Build the data context block ──────────────────────────────────
    # Assemble all state data into a structured text block.
    # This becomes the "context" we hand to llama3.
    # We format None values as "Not available" so the report
    # doesn't contain raw Python None strings.
    context = _build_context(state)

    # ── Step 2: Generate report with llama3 ───────────────────────────────────
    logger.info("Generating report with llama3...")

    report = _generate_report(
        context      = context,
        company_name = state.company_name,
        ticker       = state.ticker,
    )

    if not report:
        logger.error("Report generation failed — llama3 returned empty response")
        return {
            "final_report":      None,
            "completed_agents":  ["report_agent"],
            "errors": ["Report Agent: report generation failed"],
        }

    logger.info(f"Report generated — {len(report)} characters")

    # ── Step 3: Send email ────────────────────────────────────────────────────
    # send_report_email() gracefully skips if SendGrid is not configured.
    # We log the result but don't fail the pipeline if email fails.
    try:
        email_result = send_report_email(
            report_text = report,
            ticker      = state.ticker,
        )
        if email_result.get("success"):
            logger.info(f"Report emailed to {email_result.get('recipient')}")
        elif email_result.get("success") is None:
            logger.warning("Email skipped — SendGrid not configured")
        else:
            logger.warning(f"Email failed: {email_result.get('message')}")
            errors.append(f"Report Agent: email failed — {email_result.get('message')}")
    except Exception as e:
        logger.warning(f"Email sending failed: {e}")
        errors.append(f"Report Agent: email error — {e}")

    logger.info("Report Agent complete ✅")

    return {
        "final_report":     report,
        "completed_agents": ["report_agent"],
        "errors":           errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _build_context
# Assembles all state data into a structured text block for llama3.
#
# WHY FORMAT DATA THIS WAY:
#   llama3 reads this as plain text. Clear labels and sections help it
#   understand what each piece of data means and where it should appear
#   in the report. Without clear labels, llama3 might confuse sentiment
#   score with P/E ratio or mix up sections.
# ─────────────────────────────────────────────────────────────────────────────

def _build_context(state: AgentState) -> str:
    """
    Format all AgentState data into a readable context block for llama3.

    Args:
        state: Full AgentState after all agents have run

    Returns:
        Formatted string with all research data
    """

    def fmt(value, suffix=""):
        """Format a value — returns 'Not available' if None."""
        if value is None:
            return "Not available"
        return f"{value}{suffix}"

    # Format risk factors list into numbered text
    risk_text = "Not available"
    if state.risk_factors:
        risk_text = "\n".join(
            f"  {i+1}. {risk}"
            for i, risk in enumerate(state.risk_factors)
        )

    context = f"""
COMPANY: {state.company_name} ({state.ticker})

━━━ NEWS SENTIMENT ━━━
Summary        : {fmt(state.news_summary)}
Sentiment      : {fmt(state.sentiment_label)} (score: {fmt(state.sentiment_score)})
Articles       : {fmt(state.articles_count)} articles analyzed

━━━ MARKET DATA ━━━
Current Price  : ${fmt(state.current_price)}
Market Cap     : ${fmt(state.market_cap_billions)}B
P/E Ratio      : {fmt(state.pe_ratio)}x
Revenue Growth : {fmt(state.revenue_growth_pct)}% YoY
Gross Margin   : {fmt(state.gross_margin_pct)}%
52W High       : ${fmt(state.week_52_high)}
52W Low        : ${fmt(state.week_52_low)}

━━━ BUSINESS SEGMENTS (from SEC 10-K) ━━━
{fmt(state.revenue_breakdown)}

━━━ KEY RISK FACTORS (from SEC 10-K) ━━━
{risk_text}

━━━ MANAGEMENT OUTLOOK (from SEC 10-K) ━━━
{fmt(state.management_outlook)}

━━━ SOURCE ━━━
{fmt(state.filing_source)}
""".strip()

    return context


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _generate_report
# Sends context to llama3 with a structured prompt and returns the report.
#
# PROMPT DESIGN:
#   We give llama3 a clear role, all the data as context, and an exact
#   template to follow. Without the template, reports are inconsistent.
#   With the template, every report has the same professional structure.
#
# WHY num_predict=1000:
#   A full research report needs more tokens than a sentiment summary.
#   1000 tokens ≈ 750 words — enough for a complete professional report.
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report(
    context:      str,
    company_name: str,
    ticker:       str,
) -> str:
    """
    Generate a complete research report using llama3.

    Args:
        context:      Formatted data block from _build_context()
        company_name: Full company name
        ticker:       Stock ticker

    Returns:
        Complete report as a string, or None if generation failed
    """

    prompt = f"""You are a professional financial analyst writing an investment research report.

Here is all the research data collected for {company_name} ({ticker}):

{context}

Write a professional financial research report using EXACTLY this structure:

# {company_name} ({ticker}) — Investment Research Report

## Executive Summary
[2-3 sentences summarizing the overall investment thesis based on all the data]

## Market Sentiment
[Summarize the news sentiment and what it means for the stock]

## Financial Snapshot
[Present the key financial metrics: price, market cap, P/E, revenue growth, margins]

## Business Segments & Revenue
[Describe the company's business segments and revenue performance]

## Key Risk Factors
[List and briefly explain the top 3-4 risks]

## Management Outlook
[Summarize management's forward guidance and strategic direction]

## Investment Considerations
[Balanced assessment: 2 bull points and 2 bear points]

## Disclaimer
This report is generated by an AI system for informational purposes only and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

Write the report now. Use the data provided. Be specific and professional."""

    try:
        report = call_llm(prompt=prompt, temperature=0.3, max_tokens=1000)
        if not report:
            return None
        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Report Agent — Sanity Check")
    print(f"{'='*60}")

    # Build a mock state with realistic data from previous agents
    state = AgentState(
        company_name = "NVIDIA Corporation",
        ticker       = "NVDA",

        # From research_agent
        news_summary    = "NVIDIA continues to dominate the AI chip market with strong analyst sentiment ahead of Q1 FY2027 earnings. Multiple analysts have raised price targets citing hyperscaler demand.",
        sentiment_score = 0.85,
        sentiment_label = "positive",
        articles_count  = 10,

        # From data_agent
        current_price       = 225.32,
        market_cap_billions = 5457.37,
        pe_ratio            = 46.08,
        revenue_growth_pct  = 73.2,
        gross_margin_pct    = 71.07,
        week_52_high        = 236.54,
        week_52_low         = 129.16,

        # From filing_agent
        revenue_breakdown  = "NVIDIA operates two segments: Compute & Networking (driven by H100/H200 GPU demand from hyperscalers) and Graphics (year-over-year increase driven by Blackwell architecture sales). Compute & Networking represents the majority of revenue.",
        risk_factors       = [
            "Export controls restricting chip sales to China impacted ~$4.5B in revenue",
            "Competition from AMD MI300X and custom silicon from Google, Amazon, Microsoft",
            "Supply chain disruptions and long manufacturing lead times",
            "Concentration risk — significant revenue from a small number of hyperscaler customers",
        ],
        management_outlook = "Management expects continued growth driven by data center compute and networking platforms for accelerated computing and AI. Availability of data centers and energy infrastructure is crucial for sustained demand.",
        filing_source      = "SEC 10-K FY2026 — NVIDIA CORP",

        completed_agents = ["research_agent", "data_agent", "filing_agent"],
    )

    # ── Test 1: Context building ──────────────────────────────────────────────
    print(f"\n── Test 1: Context block ─────────────────────────────────────\n")
    context = _build_context(state)
    print(context)

    # ── Test 2: Full report generation ───────────────────────────────────────
    print(f"\n── Test 2: Full report generation ───────────────────────────\n")
    print("Generating report with llama3 (may take 1-2 minutes)...")

    result = report_agent(state)

    print(f"\n  completed_agents : {result.get('completed_agents')}")
    print(f"  errors           : {result.get('errors')}")
    print(f"  report length    : {len(result.get('final_report', '') or '')} chars")
    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"{'='*60}\n")
    print(result.get("final_report", "Report generation failed"))
    print()