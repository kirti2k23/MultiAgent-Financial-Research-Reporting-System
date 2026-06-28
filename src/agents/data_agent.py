# Pulls live financial metrics from Yahoo Finance
"""
src/agents/data_agent.py
========================
Agent 2 of 4: Fetches live stock metrics from Yahoo Finance.

ROLE IN THE PIPELINE:
    orchestrator.py    → creates plan, routes to agents
    research_agent.py  → fetches news + analyzes sentiment
    data_agent.py      → pulls live stock metrics           ← YOU ARE HERE
    filing_agent.py    → reads SEC 10-K via RAG pipeline
    report_agent.py    → synthesizes everything into final report

WHAT THIS AGENT DOES:
    1. Calls get_stock_metrics() from stock_data.py
    2. Reads the returned metrics dict
    3. Writes 7 key metrics to AgentState

WHY NO LLM HERE:
    Unlike research_agent which needs llama3 to analyze sentiment,
    this agent is purely mechanical — call the tool, read the values,
    write to state. Stock metrics are already structured numbers.
    No language understanding needed.

WHY A SEPARATE AGENT:
    In the LangGraph pipeline, data_agent, research_agent, and
    filing_agent run in PARALLEL. Each is independent:
        - data_agent  doesn't need news or SEC filings
        - research_agent doesn't need stock data
        - filing_agent doesn't need either
    Running in parallel cuts total pipeline time significantly.

WHAT IT WRITES TO AGENTSTATE:
    current_price       → 221.50  (USD)
    market_cap_billions → 5400.0  (billions USD)
    pe_ratio            → 38.5    (trailing P/E)
    revenue_growth_pct  → 122.4   (YoY %)
    gross_margin_pct    → 72.7    (%)
    week_52_high        → 153.13  (USD)
    week_52_low         → 86.22   (USD)

ERROR HANDLING:
    If yfinance fails (invalid ticker, network issue):
    → Write None values to state
    → Add error to state.errors
    → Mark agent completed anyway
    Pipeline continues — a report without stock data is better
    than no report at all.

USED BY:
    src/pipeline/graph.py → registered as a node in LangGraph

HOW TO RUN:
    cd financial-research-agent
    python -m src.agents.data_agent
"""

from src.models.state import AgentState
from src.tools.stock_data import get_stock_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT FUNCTION — data_agent
#
# WHY THIS FUNCTION SIGNATURE:
#   LangGraph requires every agent node to:
#       - Accept AgentState as input
#       - Return a DICT of only the fields it changed
#   LangGraph merges this dict back into the full state automatically.
# ─────────────────────────────────────────────────────────────────────────────

def data_agent(state: AgentState) -> dict:
    """
    Fetch live stock metrics for the company and write to AgentState.

    FLOW:
        state.ticker
            ↓
        get_stock_metrics(ticker) → dict of financial metrics
            ↓
        extract 7 key metrics
            ↓
        return dict of state updates

    WHY THESE 7 METRICS:
        current_price       → what the stock costs right now
        market_cap_billions → total company value — context for the report
        pe_ratio            → is the stock cheap or expensive vs earnings?
        revenue_growth_pct  → how fast is the company growing?
        gross_margin_pct    → how profitable is the core business?
        week_52_high        → where has the stock been at its best?
        week_52_low         → where has it been at its worst?

        Together these give a complete snapshot of the company's
        financial health and market valuation for the report.

    Args:
        state: Current AgentState — reads ticker

    Returns:
        dict with stock metric fields + completed_agents + errors
    """
    logger.info(
        f"Data Agent starting for "
        f"{state.company_name} ({state.ticker})"
    )

    # ── Fetch stock metrics ───────────────────────────────────────────────────
    # get_stock_metrics() calls Yahoo Finance via yfinance.
    # Returns a dict with 14+ fields — we extract only what we need.
    # The @retry decorator in stock_data.py handles transient failures.
    try:
        metrics = get_stock_metrics(state.ticker)

    except Exception as e:
        logger.error(f"Stock metrics fetch failed for {state.ticker}: {e}")
        return {
            "current_price":       None,
            "market_cap_billions": None,
            "pe_ratio":            None,
            "revenue_growth_pct":  None,
            "gross_margin_pct":    None,
            "week_52_high":        None,
            "week_52_low":         None,
            "completed_agents":    ["data_agent"],
            "errors": [f"Data Agent: stock metrics fetch failed — {e}"],
        }

    # ── Validate we got real data ─────────────────────────────────────────────
    # yfinance returns an empty dict for invalid tickers.
    # current_price being None is the clearest signal of failure.
    if not metrics.get("current_price"):
        logger.warning(
            f"No price data for {state.ticker} — "
            f"ticker may be invalid or market is closed"
        )
        return {
            "current_price":       None,
            "market_cap_billions": None,
            "pe_ratio":            None,
            "revenue_growth_pct":  None,
            "gross_margin_pct":    None,
            "week_52_high":        None,
            "week_52_low":         None,
            "completed_agents":    ["data_agent"],
            "errors": [f"Data Agent: no price data for {state.ticker}"],
        }

    # ── Log what we found ─────────────────────────────────────────────────────
    logger.info(
        f"Data Agent complete — "
        f"price=${metrics.get('current_price')}, "
        f"market_cap=${metrics.get('market_cap_billions')}B, "
        f"P/E={metrics.get('pe_ratio')}x, "
        f"revenue_growth={metrics.get('revenue_growth_pct')}%"
    )

    # ── Return state updates ──────────────────────────────────────────────────
    # We extract only the 7 fields AgentState expects.
    # get_stock_metrics() returns more fields (sector, industry, etc.)
    # but those aren't in AgentState so we don't include them here.
    return {
        "current_price":       metrics.get("current_price"),
        "market_cap_billions": metrics.get("market_cap_billions"),
        "pe_ratio":            metrics.get("pe_ratio"),
        "revenue_growth_pct":  metrics.get("revenue_growth_pct"),
        "gross_margin_pct":    metrics.get("gross_margin_pct"),
        "week_52_high":        metrics.get("week_52_high"),
        "week_52_low":         metrics.get("week_52_low"),
        "completed_agents":    ["data_agent"],
        "errors":              [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
#
# WHAT THIS TESTS:
#   1. yfinance is working — can fetch real stock data
#   2. All 7 metrics are present and have sensible values
#   3. Full agent function works end to end with AgentState
#
# HOW TO RUN:
#   cd financial-research-agent
#   python -m src.agents.data_agent
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Data Agent — Sanity Check")
    print(f"{'='*60}")

    # ── Test 1: Direct metrics fetch ──────────────────────────────────────────
    print(f"\n── Test 1: Fetch stock metrics directly ─────────────────────\n")

    metrics = get_stock_metrics("NVDA")

    print(f"  Company          : {metrics.get('company_name')}")
    print(f"  Current Price    : ${metrics.get('current_price')}")
    print(f"  Market Cap       : ${metrics.get('market_cap_billions')}B")
    print(f"  P/E Ratio        : {metrics.get('pe_ratio')}x")
    print(f"  Revenue Growth   : {metrics.get('revenue_growth_pct')}%")
    print(f"  Gross Margin     : {metrics.get('gross_margin_pct')}%")
    print(f"  52W High         : ${metrics.get('week_52_high')}")
    print(f"  52W Low          : ${metrics.get('week_52_low')}")

    # ── Test 2: Full agent run ────────────────────────────────────────────────
    print(f"\n── Test 2: Full agent run via AgentState ────────────────────\n")

    state  = AgentState(company_name="NVIDIA Corporation", ticker="NVDA")
    result = data_agent(state)

    print(f"  current_price       : ${result.get('current_price')}")
    print(f"  market_cap_billions : ${result.get('market_cap_billions')}B")
    print(f"  pe_ratio            : {result.get('pe_ratio')}x")
    print(f"  revenue_growth_pct  : {result.get('revenue_growth_pct')}%")
    print(f"  gross_margin_pct    : {result.get('gross_margin_pct')}%")
    print(f"  week_52_high        : ${result.get('week_52_high')}")
    print(f"  week_52_low         : ${result.get('week_52_low')}")
    print(f"  completed_agents    : {result.get('completed_agents')}")
    print(f"  errors              : {result.get('errors')}")

    # ── Test 3: Invalid ticker ────────────────────────────────────────────────
    print(f"\n── Test 3: Invalid ticker graceful failure ───────────────────\n")

    state_bad  = AgentState(company_name="Fake Company", ticker="INVALIDXYZ")
    result_bad = data_agent(state_bad)

    print(f"  current_price    : {result_bad.get('current_price')}  (expected None)")
    print(f"  completed_agents : {result_bad.get('completed_agents')}")
    print(f"  errors           : {result_bad.get('errors')}")
    print()