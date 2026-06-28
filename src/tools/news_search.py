# Searches financial news via Tavily API
"""
src/tools/news_search.py
========================
Tool: Fetches live financial news and computes sentiment for a given stock.

ROLE IN THE PIPELINE:
    Called by the Research Agent (src/agents/research_agent.py).
    Returns recent news articles + sentiment score + label
    for a given company/ticker.

TWO FUNCTIONS:
    search_company_news(ticker, company_name) -> articles + raw sentiment data
    analyze_sentiment(articles)              -> sentiment score + label + summary

USED BY:
    src/agents/research_agent.py -> calls search_company_news()

HOW TO RUN:
    cd financial-research-agent
    python -m src.tools.news_search

DEPENDENCIES:
    pip install tavily-python textblob
    textblob also needs: python -m textblob.download_corpora
"""

from tavily import TavilyClient
from textblob import TextBlob
from src.utils.config import config
from src.utils.logger import get_logger
from src.utils.retry import retry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — search_company_news
# Calls Tavily to fetch recent financial news articles for a company.
# ─────────────────────────────────────────────────────────────────────────────

@retry(max_attempts=3, initial_wait=2)
def search_company_news(ticker: str, company_name: str, max_results: int = 10) -> dict:
    """
    Search for recent financial news about a company using Tavily.

    HOW TAVILY WORKS:
        Tavily is a search API built for LLMs.
        Unlike Google, it returns clean structured results — no HTML parsing needed.
        Each result has: title, content (cleaned), url, published_date.

    WHY NOT GOOGLE?
        Google returns raw HTML full of ads, menus, and tracking scripts.
        You'd need a scraper + parser to extract the actual text.
        Tavily does all of that for you and returns LLM-ready text.

    QUERY STRATEGY:
        We build a specific query: "NVIDIA Corporation NVDA financial news 2026"
        The more specific the query, the better the results.
        Including both company name AND ticker catches articles that use either.

    Args:
        ticker:       Stock symbol e.g. "NVDA"
        company_name: Full company name e.g. "NVIDIA Corporation"
        max_results:  Max number of articles to fetch. Default 10.

    Returns:
        dict with these keys:
            ticker          -> the ticker symbol
            company_name    -> the company name
            query           -> the search query we used (useful for debugging)
            articles        -> list of article dicts (see below)
            article_count   -> how many articles were returned
            sentiment_score -> float 0.0 to 1.0 (computed from article text)
            sentiment_label -> "positive" | "neutral" | "negative"

        Each article dict:
            title          -> headline
            content        -> cleaned article body (Tavily handles extraction)
            url            -> source URL for citation
            published_date -> when the article was published (string)
    """
    logger.info(f"Searching financial news for {company_name} ({ticker})")

    # ── Step 1: Initialize Tavily client ──────────────────────────────────────
    # TavilyClient takes the API key from config
    # We do this inside the function (not at module level) so the @retry
    # decorator can reconnect cleanly on transient errors
    client = TavilyClient(api_key=config.tavily_api_key)

    # ── Step 2: Build a targeted search query ─────────────────────────────────
    # "NVIDIA Corporation NVDA financial news 2026"
    # Including the year helps surface recent articles over old ones
    from datetime import datetime
    current_year = datetime.now().year
    query = f"{company_name} {ticker} financial news {current_year}"

    logger.info(f"Tavily query: '{query}'")

    # ── Step 3: Call Tavily search ─────────────────────────────────────────────
    # search_depth="advanced" → Tavily fetches and cleans full article content
    # search_depth="basic"    → Tavily returns just snippets (cheaper, less detail)
    # We use "advanced" because we need full article text for sentiment analysis
    #
    # topic="news" → filters to news sources only (not blogs, Wikipedia, etc.)
    #
    # max_results → number of articles to return (we default to 10)
    response = client.search(
        query=query,
        search_depth="advanced",
        topic="news",
        max_results=max_results,
    )

    # ── Step 4: Extract results ───────────────────────────────────────────────
    # Tavily response format:
    # {
    #   "results": [
    #       {
    #           "title":            "NVIDIA beats Q4 earnings...",
    #           "content":          "NVIDIA reported quarterly revenue of...",
    #           "url":              "https://reuters.com/...",
    #           "published_date":   "2026-04-28",
    #           "score":            0.92   ← Tavily relevance score, not sentiment
    #       },
    #       ...
    #   ]
    # }
    raw_results = response.get("results", [])

    logger.info(f"Tavily returned {len(raw_results)} articles for {ticker}")

    if not raw_results:
        logger.warning(f"No news articles found for {ticker} — query: '{query}'")

    # ── Step 5: Clean and structure each article ──────────────────────────────
    # We only keep the fields we actually use — drop Tavily's internal score
    articles = []
    for result in raw_results:
        articles.append({
            "title":          result.get("title", "No title"),
            "content":        result.get("content", ""),
            "url":            result.get("url", ""),
            "published_date": result.get("published_date", "Unknown"),
        })

    # ── Step 6: Run sentiment analysis on the articles ────────────────────────
    # analyze_sentiment() is defined below — we call it here
    sentiment_result = analyze_sentiment(articles)

    logger.info(
        f"News fetched for {ticker} — "
        f"{len(articles)} articles, "
        f"sentiment={sentiment_result['label']} ({sentiment_result['score']:.2f})"
    )

    return {
        "ticker":          ticker.upper(),
        "company_name":    company_name,
        "query":           query,
        "articles":        articles,
        "article_count":   len(articles),
        "sentiment_score": sentiment_result["score"],
        "sentiment_label": sentiment_result["label"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — analyze_sentiment
# Computes sentiment score from a list of news articles using TextBlob.
# ─────────────────────────────────────────────────────────────────────────────

def analyze_sentiment(articles: list[dict]) -> dict:
    """
    Compute an overall sentiment score from a list of news articles.

    HOW TEXTBLOB WORKS:
        TextBlob is a lightweight NLP library.
        It analyzes text and returns a "polarity" score:
            -1.0 → very negative
             0.0 → neutral
            +1.0 → very positive

        Example:
            TextBlob("NVIDIA crushes earnings, stock surges").sentiment.polarity
            → 0.6  (positive)

            TextBlob("NVIDIA faces lawsuit over chip export ban").sentiment.polarity
            → -0.3  (negative)

    WHY NOT GPT-4o FOR SENTIMENT?
        We could use GPT-4o but:
        - TextBlob runs instantly (milliseconds vs seconds)
        - TextBlob is free (no API cost per call)
        - For simple positive/negative labeling, TextBlob is accurate enough
        - GPT-4o is used in the Research Agent ABOVE this tool to interpret
          and contextualize the sentiment — not to compute raw scores

    NORMALIZATION:
        TextBlob returns polarity in range [-1.0, +1.0]
        We normalize to [0.0, 1.0] for easier use:
            normalized = (polarity + 1.0) / 2.0
            -1.0 → 0.0 (very negative)
             0.0 → 0.5 (neutral)
            +1.0 → 1.0 (very positive)

    Args:
        articles: List of article dicts with "title" and "content" keys

    Returns:
        dict with:
            score -> float 0.0 to 1.0
            label -> "positive" | "neutral" | "negative"
            raw_polarity -> TextBlob raw score for debugging (-1.0 to 1.0)
    """

    if not articles:
        logger.warning("analyze_sentiment called with empty article list")
        return {
            "score":        0.5,
            "label":        "neutral",
            "raw_polarity": 0.0,
        }

    # ── Step 1: Analyze each article's title + content ───────────────────────
    # We concatenate title and content — title carries strong sentiment signal
    # (headlines are written to be attention-grabbing)
    polarities = []

    for article in articles:
        text = f"{article.get('title', '')} {article.get('content', '')}"

        if not text.strip():
            continue

        # TextBlob.sentiment returns a named tuple: Sentiment(polarity, subjectivity)
        # .polarity is the sentiment score we want
        # .subjectivity (0=objective, 1=subjective) — we don't use this here
        polarity = TextBlob(text).sentiment.polarity
        polarities.append(polarity)

    # ── Step 2: Average polarity across all articles ──────────────────────────
    # This gives us a single number representing overall news sentiment
    if not polarities:
        avg_polarity = 0.0
    else:
        avg_polarity = sum(polarities) / len(polarities)

    # ── Step 3: Normalize from [-1, +1] to [0.0, 1.0] ────────────────────────
    # Formula: (polarity + 1) / 2
    normalized_score = round((avg_polarity + 1.0) / 2.0, 4)

    # ── Step 4: Map score to human-readable label ─────────────────────────────
    # Thresholds chosen based on financial sentiment conventions:
    #   > 0.6 → positive   (clearly bullish coverage)
    #   < 0.4 → negative   (clearly bearish/bad news coverage)
    #   else  → neutral    (mixed or balanced coverage)
    if normalized_score > 0.6:
        label = "positive"
    elif normalized_score < 0.4:
        label = "negative"
    else:
        label = "neutral"

    logger.info(
        f"Sentiment analysis complete — "
        f"raw polarity={avg_polarity:.4f}, "
        f"normalized={normalized_score:.4f}, "
        f"label={label}"
    )

    return {
        "score":        normalized_score,
        "label":        label,
        "raw_polarity": round(avg_polarity, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to test both functions with a real ticker.
# Requires TAVILY_API_KEY in your .env file.
#
# Usage:
#   cd financial-research-agent
#   python -m src.tools.news_search
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    TEST_TICKER       = "NVDA"
    TEST_COMPANY_NAME = "NVIDIA Corporation"

    print(f"\n{'='*60}")
    print(f"  News Search Tool — Testing with {TEST_TICKER}")
    print(f"{'='*60}")

    # ── Test 1: search_company_news ───────────────────────────────────────────
    print(f"\n── search_company_news('{TEST_TICKER}') ─────────────────────\n")

    result = search_company_news(TEST_TICKER, TEST_COMPANY_NAME, max_results=5)

    print(f"  Ticker         : {result['ticker']}")
    print(f"  Company        : {result['company_name']}")
    print(f"  Query          : {result['query']}")
    print(f"  Articles found : {result['article_count']}")
    print(f"  Sentiment      : {result['sentiment_label'].upper()} ({result['sentiment_score']:.2f})")

    print(f"\n── Article Headlines ────────────────────────────────────────\n")
    for i, article in enumerate(result["articles"], 1):
        print(f"  {i}. {article['title']}")
        print(f"     Published : {article['published_date']}")
        print(f"     URL       : {article['url']}")
        print(f"     Preview   : {article['content'][:120]}...")
        print()

    # ── Test 2: analyze_sentiment directly with mock articles ─────────────────
    print(f"\n── analyze_sentiment() — mock test (no API needed) ─────────\n")

    mock_articles = [
        {
            "title":   "NVIDIA crushes earnings, stock surges 15%",
            "content": "NVIDIA reported record revenue driven by explosive AI chip demand.",
        },
        {
            "title":   "NVIDIA faces new export restrictions on China sales",
            "content": "US regulators announced tighter controls on NVIDIA chip exports.",
        },
        {
            "title":   "NVIDIA partners with Microsoft for cloud AI infrastructure",
            "content": "NVIDIA and Microsoft expand their strategic cloud AI partnership.",
        },
    ]

    sentiment = analyze_sentiment(mock_articles)
    print(f"  Raw polarity   : {sentiment['raw_polarity']}")
    print(f"  Score (0-1)    : {sentiment['score']}")
    print(f"  Label          : {sentiment['label'].upper()}")

    # ── Test 3: Edge case — empty articles ────────────────────────────────────
    print(f"\n── analyze_sentiment([]) — empty articles edge case ─────────\n")
    empty_sentiment = analyze_sentiment([])
    print(f"  Score          : {empty_sentiment['score']}")
    print(f"  Label          : {empty_sentiment['label']} (expected: neutral)")
    print()