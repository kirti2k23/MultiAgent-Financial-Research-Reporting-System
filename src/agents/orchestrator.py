# Master coordinator — creates plan, routes to agents
"""
src/agents/orchestrator.py
==========================
Master coordinator — validates inputs, creates research plan, routes to agents.

ROLE IN THE PIPELINE:
    orchestrator.py   → validates + creates plan   ← YOU ARE HERE
    ↓ (parallel)
    research_agent.py → news + sentiment
    data_agent.py     → stock metrics
    filing_agent.py   → SEC 10-K via RAG
    ↓ (after all 3 complete)
    report_agent.py   → final report + email

WHY AN ORCHESTRATOR EXISTS:
    Without it, graph.py would hardcode which agents run every time.
    The orchestrator makes the pipeline dynamic:

    1. VALIDATION — catches bad inputs early
       e.g. empty ticker, missing company name
       Fails fast with clear error instead of crashing deep in pipeline

    2. PLANNING — decides which agents to run
       e.g. if company has no SEC filing → skip filing_agent
       e.g. if user wants quick sentiment only → skip filing_agent
       Currently we always run all 4, but the plan makes this extensible

    3. LOGGING — single entry point for the full pipeline run
       Every pipeline run starts here — easy to trace in LangSmith

    4. STATE INITIALIZATION — sets up any shared state before agents run
       e.g. normalizes ticker to uppercase, trims company name whitespace

WHAT IT WRITES TO AGENTSTATE:
    research_plan → list of steps the pipeline will execute
                    e.g. ["Fetch news", "Get stock data", "Read SEC filing", "Generate report"]

USED BY:
    src/pipeline/graph.py → first node in the LangGraph state graph

HOW TO RUN:
    cd financial-research-agent
    python -m src.agents.orchestrator
"""

from src.models.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def orchestrator(state: AgentState) -> dict:
    """
    Validate inputs, create research plan, and initialize pipeline run.

    This is the FIRST node in the LangGraph graph. Every pipeline run
    passes through here before any agent does any work.

    WHAT IT DOES:
        1. Validates company_name and ticker are present and non-empty
        2. Normalizes inputs (uppercase ticker, stripped whitespace)
        3. Creates a research plan — list of steps to execute
        4. Logs the start of the pipeline run for LangSmith tracing
        5. Returns updated state fields

    WHAT IT DOES NOT DO:
        - Does not call any external APIs
        - Does not make any LLM calls
        - Does not do any heavy computation
        The orchestrator is intentionally lightweight — it just sets things up.

    Args:
        state: Initial AgentState with company_name and ticker set by caller

    Returns:
        dict with research_plan and normalized inputs
    """

    logger.info(
        f"{'='*50}\n"
        f"Pipeline starting\n"
        f"Company : {state.company_name}\n"
        f"Ticker  : {state.ticker}\n"
        f"{'='*50}"
    )

    # ── Step 1: Validate inputs ───────────────────────────────────────────────
    # Catch obvious problems before any agent wastes time on bad inputs
    errors = []

    if not state.ticker or not state.ticker.strip():
        errors.append("Orchestrator: ticker is required but was empty")

    if not state.company_name or not state.company_name.strip():
        errors.append("Orchestrator: company_name is required but was empty")

    if errors:
        logger.error(f"Validation failed: {errors}")
        return {
            "research_plan":    None,
            "completed_agents": ["orchestrator"],
            "errors":           errors,
        }

    # ── Step 2: Normalize inputs ──────────────────────────────────────────────
    # Ticker should always be uppercase: "nvda" → "NVDA"
    # Company name should be stripped of extra whitespace
    ticker       = state.ticker.strip().upper()
    company_name = state.company_name.strip()

    # ── Step 3: Create research plan ─────────────────────────────────────────
    # The plan is a human-readable list of steps this pipeline will execute.
    # Currently we always run all 4 agents — but this is where you'd add
    # conditional logic in a production system:
    #
    #   if company_is_private(ticker):
    #       plan = [news, stock_data, report]  # skip SEC filing
    #   elif quick_mode:
    #       plan = [news, report]              # sentiment only
    #   else:
    #       plan = [news, stock_data, SEC, report]  # full research
    #
    # For now: always run full pipeline
    research_plan = [
        f"1. Fetch recent news and analyze sentiment for {company_name}",
        f"2. Pull live stock metrics for {ticker} from Yahoo Finance",
        f"3. Download and analyze SEC 10-K filing for {ticker} via RAG pipeline",
        f"4. Synthesize all research into final report and deliver via email",
    ]

    logger.info(f"Research plan created for {ticker}:")
    for step in research_plan:
        logger.info(f"  {step}")

    logger.info(
        f"Routing to parallel agents: "
        f"research_agent, data_agent, filing_agent"
    )

    return {
        "ticker":           ticker,
        "company_name":     company_name,
        "research_plan":    research_plan,
        "completed_agents": ["orchestrator"],
        "errors":           [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Orchestrator — Sanity Check")
    print(f"{'='*60}")

    # ── Test 1: Valid input ───────────────────────────────────────────────────
    print(f"\n── Test 1: Valid input ───────────────────────────────────────\n")

    state  = AgentState(company_name="NVIDIA Corporation", ticker="nvda")
    result = orchestrator(state)

    print(f"  ticker           : {result.get('ticker')}  (normalized to uppercase)")
    print(f"  company_name     : {result.get('company_name')}")
    print(f"  completed_agents : {result.get('completed_agents')}")
    print(f"  errors           : {result.get('errors')}")
    print(f"\n  Research plan:")
    for step in (result.get("research_plan") or []):
        print(f"    {step}")

    # ── Test 2: Missing ticker ────────────────────────────────────────────────
    print(f"\n── Test 2: Missing ticker (validation) ──────────────────────\n")

    state_bad  = AgentState(company_name="NVIDIA Corporation", ticker="")
    result_bad = orchestrator(state_bad)

    print(f"  research_plan    : {result_bad.get('research_plan')}  (expected None)")
    print(f"  errors           : {result_bad.get('errors')}")

    # ── Test 3: Lowercase ticker normalization ────────────────────────────────
    print(f"\n── Test 3: Ticker normalization ─────────────────────────────\n")

    state_lower  = AgentState(company_name="Apple Inc", ticker="aapl")
    result_lower = orchestrator(state_lower)

    print(f"  input ticker     : aapl")
    print(f"  output ticker    : {result_lower.get('ticker')}  (expected AAPL)")
    print()