# Fetches live stock price and metrics from yfinance
"""
src/tools/stock_data.py
=======================
Tool: Fetches live stock data and financial metrics from Yahoo Finance.

ROLE IN THE PIPELINE:
    Called by the Data Agent (src/agents/data_agent.py).
    Returns current price, market cap, P/E ratio, revenue growth,
    gross margin, and 52-week range for a given ticker.

TWO FUNCTIONS:
    get_stock_metrics(ticker)         -> current snapshot of financials
    get_price_history(ticker, period) -> historical closing prices

USED BY:
    src/agents/data_agent.py  -> calls get_stock_metrics()
    dashboard/app.py          -> calls get_price_history() for chart

HOW TO RUN:
    cd financial-research-agent
    python -m src.tools.stock_data
"""

import yfinance as yf
from src.utils.logger import get_logger
from src.utils.retry import retry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — get_stock_metrics
# Fetches the current financial snapshot for a given ticker.
# Called by Data Agent every time the pipeline runs.
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2)
def get_stock_metrics(ticker: str) -> dict:
    """
    Fetch core financial metrics for a stock ticker from Yahoo Finance.

    HOW IT WORKS:
        yf.Ticker("NVDA")  → creates a Ticker object for NVIDIA
        .info              → returns a large dict with 100+ fields
        .get("field")      → safely picks the field we need
                             returns None if field doesn't exist

    WHY .get() INSTEAD OF ["key"]:
        Not every ticker has every metric.
        A small company may not have a P/E ratio.
        .get() returns None for missing fields — safe.
        ["key"] raises a KeyError and crashes — unsafe.

    Args:
        ticker: Stock symbol e.g. "NVDA", "AAPL", "MSFT"

    Returns:
        dict with these keys:
            ticker              -> the ticker symbol
            company_name        -> full company name
            current_price       -> latest stock price in USD
            market_cap          -> market cap in raw dollars
            market_cap_billions -> market cap divided by 1 billion
            pe_ratio            -> price-to-earnings ratio
            revenue_growth      -> YoY revenue growth as decimal (0.122 = 12.2%)
            revenue_growth_pct  -> YoY revenue growth as percentage (12.2)
            gross_margin        -> gross margin as decimal (0.554 = 55.4%)
            gross_margin_pct    -> gross margin as percentage (55.4)
            week_52_high        -> highest price in last 52 weeks
            week_52_low         -> lowest price in last 52 weeks
            sector              -> e.g. "Technology"
            industry            -> e.g. "Semiconductors"
    """
    logger.info(f"Fetching stock metrics for {ticker}")

    # ── Step 1: Create Ticker object and fetch info ───────────────────────────
    stock = yf.Ticker(ticker)
    info  = stock.info

    # ── Step 2: Validate we got real data ─────────────────────────────────────
    # yfinance returns an empty or minimal dict for invalid tickers
    # Check for currentPrice or regularMarketPrice as a signal of valid data
    if not info.get("currentPrice") and not info.get("regularMarketPrice"):
        logger.warning(f"No price data found for {ticker} — ticker may be invalid")

    # ── Step 3: Extract the fields we need ────────────────────────────────────
    # Use .get() for every field — never direct access
    # This way missing fields return None instead of crashing

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap    = info.get("marketCap")
    pe_ratio      = info.get("trailingPE")
    revenue_growth = info.get("revenueGrowth")   # decimal e.g. 1.224
    gross_margin   = info.get("grossMargins")    # decimal e.g. 0.554
    week_52_high   = info.get("fiftyTwoWeekHigh")
    week_52_low    = info.get("fiftyTwoWeekLow")
    company_name   = info.get("longName") or info.get("shortName") or ticker
    sector         = info.get("sector")
    industry       = info.get("industry")

    # ── Step 4: Convert market cap to billions for readability ────────────────
    # Raw market cap is a huge number like 786000000000
    # Dividing by 1e9 gives 786.0 — much more readable
    market_cap_billions = round(market_cap / 1e9, 2) if market_cap else None

    # ── Step 5: Convert decimals to percentages ───────────────────────────────
    # yfinance returns revenue_growth as 1.224 (meaning 122.4%)
    # We store both — the raw decimal for calculations
    # and the percentage for display in the report
    revenue_growth_pct = round(revenue_growth * 100, 2) if revenue_growth else None
    gross_margin_pct   = round(gross_margin * 100, 2)   if gross_margin   else None

    # ── Step 6: Build and return the result dict ──────────────────────────────
    metrics = {
        "ticker":              ticker.upper(),
        "company_name":        company_name,
        "current_price":       current_price,
        "market_cap":          market_cap,
        "market_cap_billions": market_cap_billions,
        "pe_ratio":            round(pe_ratio, 2) if pe_ratio else None,
        "revenue_growth":      revenue_growth,
        "revenue_growth_pct":  revenue_growth_pct,
        "gross_margin":        gross_margin,
        "gross_margin_pct":    gross_margin_pct,
        "week_52_high":        week_52_high,
        "week_52_low":         week_52_low,
        "sector":              sector,
        "industry":            industry,
    }

    logger.info(
        f"Metrics fetched for {ticker} — "
        f"price=${current_price}, "
        f"market_cap=${market_cap_billions}B, "
        f"P/E={pe_ratio}"
    )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — get_price_history
# Fetches historical closing prices for a given period.
# Used by the dashboard to draw a price chart.
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2)
def get_price_history(ticker: str, period: str = "6mo") -> dict:
    """
    Fetch historical daily closing prices for a stock.

    Args:
        ticker: Stock symbol e.g. "NVDA"
        period: Time period for history.
                Valid values: "1mo" "3mo" "6mo" "1y" "2y" "5y"
                Default is "6mo" — last 6 months

    Returns:
        dict with these keys:
            ticker  -> the ticker symbol
            period  -> the period requested
            dates   -> list of date strings ["2026-01-02", "2026-01-03", ...]
            closes  -> list of closing prices [310.5, 315.2, 318.0, ...]
            high    -> highest closing price in the period
            low     -> lowest closing price in the period
            change_pct -> % change from first to last price in period
    """
    logger.info(f"Fetching price history for {ticker} ({period})")

    stock = yf.Ticker(ticker)

    # .history() returns a Pandas DataFrame with columns:
    # Open, High, Low, Close, Volume, Dividends, Stock Splits
    hist = stock.history(period=period)

    if hist.empty:
        logger.warning(f"No price history found for {ticker}")
        return {
            "ticker":     ticker.upper(),
            "period":     period,
            "dates":      [],
            "closes":     [],
            "high":       None,
            "low":        None,
            "change_pct": None,
        }

    # Extract closing prices and format dates as strings
    closes = hist["Close"].round(2).tolist()
    dates  = hist.index.strftime("%Y-%m-%d").tolist()

    # Calculate period high, low, and % change
    high       = round(max(closes), 2)
    low        = round(min(closes), 2)
    change_pct = round(((closes[-1] - closes[0]) / closes[0]) * 100, 2)

    logger.info(
        f"Price history fetched for {ticker} — "
        f"{len(closes)} days, "
        f"high=${high}, low=${low}, "
        f"change={change_pct}%"
    )

    return {
        "ticker":     ticker.upper(),
        "period":     period,
        "dates":      dates,
        "closes":     closes,
        "high":       high,
        "low":        low,
        "change_pct": change_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to test both functions with a real ticker.
#
# Usage:
#   cd financial-research-agent
#   python -m src.tools.stock_data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_TICKER = "NVDA"

    print(f"\n{'='*55}")
    print(f"  Stock Data Tool — Testing with {TEST_TICKER}")
    print(f"{'='*55}")

    # ── Test 1: get_stock_metrics ─────────────────────────────────────────────
    print(f"\n── get_stock_metrics('{TEST_TICKER}') ──────────────────────\n")

    metrics = get_stock_metrics(TEST_TICKER)

    print(f"  Company        : {metrics['company_name']}")
    print(f"  Sector         : {metrics['sector']}")
    print(f"  Industry       : {metrics['industry']}")
    print(f"  Current Price  : ${metrics['current_price']}")
    print(f"  Market Cap     : ${metrics['market_cap_billions']}B")
    print(f"  P/E Ratio      : {metrics['pe_ratio']}x")
    print(f"  Revenue Growth : {metrics['revenue_growth_pct']}%")
    print(f"  Gross Margin   : {metrics['gross_margin_pct']}%")
    print(f"  52W High       : ${metrics['week_52_high']}")
    print(f"  52W Low        : ${metrics['week_52_low']}")

    # ── Test 2: get_price_history ─────────────────────────────────────────────
    print(f"\n── get_price_history('{TEST_TICKER}', '3mo') ───────────────\n")

    history = get_price_history(TEST_TICKER, period="3mo")

    print(f"  Ticker         : {history['ticker']}")
    print(f"  Period         : {history['period']}")
    print(f"  Days fetched   : {len(history['dates'])}")
    print(f"  Period High    : ${history['high']}")
    print(f"  Period Low     : ${history['low']}")
    print(f"  Change         : {history['change_pct']}%")
    print(f"  First date     : {history['dates'][0] if history['dates'] else 'N/A'}")
    print(f"  Last date      : {history['dates'][-1] if history['dates'] else 'N/A'}")
    print(f"  Last 5 closes  : {history['closes'][-5:]}")

    # ── Test 3: Invalid ticker ────────────────────────────────────────────────
    print(f"\n── get_stock_metrics('INVALIDTICKER') ──────────────────\n")

    bad_metrics = get_stock_metrics("INVALIDTICKER")
    print(f"  current_price  : {bad_metrics['current_price']}")
    print(f"  pe_ratio       : {bad_metrics['pe_ratio']}")
    print(f"  (None values are expected for invalid ticker)")
    print()