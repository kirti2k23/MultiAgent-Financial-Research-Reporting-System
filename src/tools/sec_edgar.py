"""
src/tools/sec_edgar.py
======================
Tool: Fetches the latest SEC 10-K annual filing for a given stock ticker.

ROLE IN THE PIPELINE:
    Called by the Filing Agent (src/agents/filing_agent.py).
    Returns the full text of the company's most recent 10-K filing.
    That text is then chunked and stored in ChromaDB for RAG retrieval.

TWO FUNCTIONS:
    get_company_cik(ticker)    -> converts ticker to SEC's internal CIK number
    fetch_latest_10k(ticker)   -> fetches the full 10-K filing text + metadata

WHY NO API KEY:
    SEC EDGAR is a US government public database — completely free.
    No authentication required. You only need a User-Agent header
    to identify yourself (SEC requires this by policy).

USED BY:
    src/agents/filing_agent.py -> calls fetch_latest_10k()
    src/rag/document_loader.py -> receives the text for chunking

HOW TO RUN:
    cd financial-research-agent
    python -m src.tools.sec_edgar

DEPENDENCIES:
    pip install requests beautifulsoup4
"""

import requests
from requests.exceptions import RequestException
from src.utils.logger import get_logger
from src.utils.retry import retry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR BASE URLS
#
# SEC has two main APIs we use:
#
# 1. company_tickers.json
#    A single JSON file mapping every public company ticker → CIK number.
#    We download this once to convert "NVDA" → "0001045810".
#
# 2. submissions endpoint
#    Given a CIK, returns all filings for that company (10-K, 10-Q, 8-K etc.)
#    We filter for form type "10-K" and grab the most recent one.
#
# 3. Archives endpoint
#    Given a CIK and accession number, returns the actual filing documents.
#    We fetch the primary document (the 10-K HTML/TXT file).
# ─────────────────────────────────────────────────────────────────────────────

EDGAR_BASE_URL       = "https://data.sec.gov"
EDGAR_ARCHIVES_URL   = "https://www.sec.gov/Archives/edgar/full-index"
EDGAR_COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# ─────────────────────────────────────────────────────────────────────────────
# USER-AGENT HEADER
#
# SEC EDGAR requires every request to include a User-Agent header
# that identifies who is making the request.
# Format: "Your Name your@email.com"
# Without this, SEC will block your requests (HTTP 403).
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "FinancialResearchAgent contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — get_company_cik
# Converts a stock ticker to SEC's internal CIK (Central Index Key) number.
#
# WHY CIK EXISTS:
#   SEC EDGAR doesn't index filings by ticker symbol.
#   It uses CIK numbers — a unique identifier assigned to each company.
#   NVIDIA's CIK is 0001045810.
#   Apple's CIK is 0000320193.
#   You cannot query SEC without the CIK.
#
# HOW WE GET IT:
#   SEC provides a public JSON file at /files/company_tickers.json
#   that maps every company's ticker → CIK + company name.
#   We download that file and look up our ticker.
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2, exceptions=(RequestException,))
def get_company_cik(ticker: str) -> dict:
    """
    Convert a stock ticker symbol to SEC's CIK number.

    HOW IT WORKS:
        SEC maintains a public JSON file with all listed companies:
        {
            "0": {"cik_str": 320193,  "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019,  "ticker": "MSFT", "title": "MICROSOFT CORP"},
            "2": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
            ...
        }
        We download it, search for our ticker, and return the CIK.

    CIK PADDING:
        SEC CIK numbers are always 10 digits, zero-padded on the left.
        Raw CIK:    1045810
        Padded CIK: 0001045810
        The padded format is required in API URLs.

    Args:
        ticker: Stock symbol e.g. "NVDA", "AAPL", "MSFT"

    Returns:
        dict with:
            ticker       -> the ticker symbol (uppercased)
            cik          -> zero-padded 10-digit CIK string e.g. "0001045810"
            company_name -> official SEC company name e.g. "NVIDIA CORP"

    Raises:
        ValueError: if the ticker is not found in SEC's database
    """
    logger.info(f"Looking up CIK for ticker: {ticker.upper()}")

    # ── Step 1: Download the full company tickers JSON from SEC ───────────────
    # This file has ~10,000+ companies and is about 2MB
    # We download it fresh each time — it updates as companies list/delist
    response = requests.get(
        EDGAR_COMPANY_TICKERS,
        headers={"User-Agent": "FinancialResearchAgent contact@example.com"},
    )
    response.raise_for_status()   # raises HTTPError if status is 4xx or 5xx

    companies = response.json()

    # ── Step 2: Search for our ticker ────────────────────────────────────────
    # The JSON is a dict of dicts (not a list), keyed by index "0", "1", "2"...
    # We iterate all companies and match on ticker field (case-insensitive)
    ticker_upper = ticker.upper()

    for company in companies.values():
        if company.get("ticker", "").upper() == ticker_upper:

            # ── Step 3: Zero-pad the CIK to 10 digits ────────────────────────
            # Raw CIK from JSON: 1045810 (integer)
            # Padded CIK needed: "0001045810" (string, 10 chars)
            # str().zfill(10) adds leading zeros to reach 10 characters
            raw_cik     = str(company["cik_str"])
            padded_cik  = raw_cik.zfill(10)
            company_name = company.get("title", ticker_upper)

            logger.info(
                f"CIK found for {ticker_upper}: "
                f"CIK={padded_cik}, company='{company_name}'"
            )

            return {
                "ticker":       ticker_upper,
                "cik":          padded_cik,
                "company_name": company_name,
            }

    # ── Ticker not found ──────────────────────────────────────────────────────
    logger.error(f"Ticker '{ticker_upper}' not found in SEC EDGAR database")
    raise ValueError(
        f"Ticker '{ticker_upper}' not found in SEC EDGAR. "
        f"Make sure it is a US-listed public company."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — fetch_latest_10k
# Fetches the full text of the most recent 10-K annual filing.
#
# HOW THE SEC FILING SYSTEM WORKS:
#
#   Every filing has an "accession number" — a unique ID like:
#       0001045810-24-000029
#   It encodes: CIK - year - sequence number
#
#   The submissions endpoint returns all filings for a company:
#       https://data.sec.gov/submissions/CIK0001045810.json
#   We filter for form type "10-K" and get the most recent accession number.
#
#   Then we build the filing index URL:
#       https://www.sec.gov/Archives/edgar/data/{CIK}/{accession}/
#   And fetch the primary document (the actual 10-K HTML or TXT file).
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2, exceptions=(RequestException,))
def fetch_latest_10k(ticker: str) -> dict:
    """
    Fetch the full text of the most recent 10-K filing for a company.

    WHAT IS A 10-K:
        A 10-K is the annual report every US public company must file with the SEC.
        It contains:
        - Exact revenue breakdown by business segment
        - Management Discussion & Analysis (MD&A)
        - Risk factors (official list of what could go wrong)
        - Balance sheet, income statement, cash flows
        - Future outlook and guidance

        It is the most authoritative document about a company's financials.
        More reliable than any news article or analyst summary.

    Args:
        ticker: Stock symbol e.g. "NVDA"

    Returns:
        dict with:
            ticker        -> the ticker symbol
            company_name  -> official SEC company name
            cik           -> SEC CIK number
            filing_date   -> date the 10-K was filed e.g. "2024-02-21"
            accession_no  -> SEC accession number e.g. "0001045810-24-000029"
            filing_url    -> direct URL to the filing index page
            text          -> full raw text content of the 10-K document
            text_length   -> character count of the text
    """
    logger.info(f"Fetching latest 10-K for {ticker.upper()}")

    # ── Step 1: Get the CIK for this ticker ───────────────────────────────────
    cik_data     = get_company_cik(ticker)
    cik          = cik_data["cik"]
    company_name = cik_data["company_name"]

    # ── Step 2: Fetch all filings for this company ────────────────────────────
    # The submissions endpoint returns a JSON with ALL filings ever made
    # by this company — 10-K, 10-Q, 8-K, DEF 14A, etc.
    submissions_url = f"{EDGAR_BASE_URL}/submissions/CIK{cik}.json"

    logger.info(f"Fetching submissions for CIK {cik} from: {submissions_url}")

    submissions_response = requests.get(submissions_url, headers=HEADERS)
    submissions_response.raise_for_status()
    submissions = submissions_response.json()

    # ── Step 3: Find the most recent 10-K filing ──────────────────────────────
    # submissions["filings"]["recent"] has parallel arrays:
    # {
    #   "form":          ["10-K",         "10-Q",         "8-K",  ...],
    #   "filingDate":    ["2024-02-21",   "2023-11-20",   "...", ...],
    #   "accessionNumber":["0001045810-24-000029", "...", "...", ...],
    #   "primaryDocument":["nvda-20240128.htm",    "...", "...", ...],
    # }
    # All arrays are aligned by index — index 0 is the same filing across all arrays.
    # We find the first index where form == "10-K" (most recent first).

    recent_filings = submissions.get("filings", {}).get("recent", {})

    forms            = recent_filings.get("form", [])
    filing_dates     = recent_filings.get("filingDate", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])

    # Find the index of the most recent 10-K
    ten_k_index = None
    for i, form in enumerate(forms):
        if form == "10-K":
            ten_k_index = i
            break   # first match = most recent (filings are newest-first)

    if ten_k_index is None:
        logger.error(f"No 10-K filing found for {ticker.upper()} (CIK: {cik})")
        raise ValueError(f"No 10-K filing found for {ticker.upper()}")

    # ── Step 4: Extract filing metadata ──────────────────────────────────────
    filing_date      = filing_dates[ten_k_index]
    accession_no     = accession_numbers[ten_k_index]
    primary_document = primary_documents[ten_k_index]

    logger.info(
        f"Found 10-K for {ticker.upper()}: "
        f"filed={filing_date}, accession={accession_no}"
    )

    # ── Step 5: Build the filing document URL ────────────────────────────────
    # Accession number format: "0001045810-24-000029"
    # URL format needs dashes removed: "000104581024000029"
    # Full URL: https://www.sec.gov/Archives/edgar/data/{CIK}/{accession_nodash}/{primary_doc}
    accession_nodash = accession_no.replace("-", "")
    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_nodash}/{primary_document}"
    )

    logger.info(f"Fetching 10-K document from: {filing_url}")

    # ── Step 6: Download the actual 10-K document ─────────────────────────────
    # The primary document is usually an HTML file (.htm)
    # It can be very large — 500KB to 5MB of raw HTML
    doc_headers = {
        "User-Agent": "FinancialResearchAgent contact@example.com",
        "Accept-Encoding": "gzip, deflate",
    }
    doc_response = requests.get(filing_url, headers=doc_headers)
    doc_response.raise_for_status()

    raw_content = doc_response.text

    # ── Step 7: Strip HTML tags to get plain text ─────────────────────────────
    # The 10-K is HTML — full of <table>, <div>, <span> tags
    # We only want the readable text for our RAG pipeline
    # BeautifulSoup extracts just the text content
    text = _extract_text_from_html(raw_content)

    logger.info(
        f"10-K fetched for {ticker.upper()} — "
        f"{len(text):,} characters of text extracted"
    )

    return {
        "ticker":       ticker.upper(),
        "company_name": company_name,
        "cik":          cik,
        "filing_date":  filing_date,
        "accession_no": accession_no,
        "filing_url":   filing_url,
        "text":         text,
        "text_length":  len(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _extract_text_from_html
# Strips HTML tags from raw SEC filing HTML and returns clean plain text.
# Private function (prefixed with _) — only used inside this file.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_from_html(html: str) -> str:
    """
    Extract readable plain text from raw HTML content.

    WHY BEAUTIFULSOUP:
        SEC filings are HTML documents with complex nested tags.
        BeautifulSoup parses HTML and lets us call .get_text()
        to extract just the visible text — no tags, no scripts, no styles.

    WHY separator="\n":
        Without a separator, all text runs together: "RevenueNet income..."
        With separator="\n", each block gets its own line — much cleaner
        for chunking in the RAG pipeline later.

    Args:
        html: Raw HTML string from the SEC filing

    Returns:
        Clean plain text string
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style tags — their content is not readable text
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Extract text — separator="\n" puts each block on its own line
        # strip=True removes leading/trailing whitespace from each text block
        text = soup.get_text(separator="\n", strip=True)

        # Remove excessive blank lines (more than 2 consecutive newlines)
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    except Exception as e:
        logger.warning(f"HTML parsing failed: {e} — returning raw text")
        # Fallback: return raw HTML if BeautifulSoup fails
        # Better to have messy text than no text
        return html


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to test both functions with a real ticker.
# No API key needed — SEC EDGAR is a free public service.
#
# Usage:
#   cd financial-research-agent
#   python -m src.tools.sec_edgar
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_TICKER = "NVDA"

    print(f"\n{'='*60}")
    print(f"  SEC EDGAR Tool — Testing with {TEST_TICKER}")
    print(f"{'='*60}")

    # ── Test 1: get_company_cik ───────────────────────────────────────────────
    print(f"\n── get_company_cik('{TEST_TICKER}') ──────────────────────────\n")

    cik_result = get_company_cik(TEST_TICKER)

    print(f"  Ticker       : {cik_result['ticker']}")
    print(f"  CIK          : {cik_result['cik']}")
    print(f"  Company Name : {cik_result['company_name']}")

    # ── Test 2: fetch_latest_10k ──────────────────────────────────────────────
    print(f"\n── fetch_latest_10k('{TEST_TICKER}') ─────────────────────────\n")

    filing = fetch_latest_10k(TEST_TICKER)

    print(f"  Ticker       : {filing['ticker']}")
    print(f"  Company      : {filing['company_name']}")
    print(f"  CIK          : {filing['cik']}")
    print(f"  Filing Date  : {filing['filing_date']}")
    print(f"  Accession No : {filing['accession_no']}")
    print(f"  Filing URL   : {filing['filing_url']}")
    print(f"  Text Length  : {filing['text_length']:,} characters")

    # Print first 500 characters of the text to verify it's readable
    print(f"\n── First 500 characters of extracted text ───────────────────\n")
    print(filing["text"][:500])
    print("...")

    # ── Test 3: Invalid ticker ────────────────────────────────────────────────
    print(f"\n── get_company_cik('INVALIDTICKER') ─────────────────────────\n")

    try:
        get_company_cik("INVALIDTICKER")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print()