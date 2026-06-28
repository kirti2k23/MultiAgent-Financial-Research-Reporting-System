"""
src/agents/research_agent.py
============================
Agent 1 of 4: Fetches recent news and analyzes sentiment.

ROLE IN THE PIPELINE:
    orchestrator.py    → creates plan, routes to agents
    research_agent.py  → fetches news + analyzes sentiment  ← YOU ARE HERE
    data_agent.py      → pulls live stock metrics
    filing_agent.py    → reads SEC 10-K via RAG pipeline
    report_agent.py    → synthesizes everything into final report

WHAT THIS AGENT DOES:
    1. Calls news_search.py → fetches top 10 recent news articles
    2. Formats all article titles + summaries into one text block
    3. Sends to llama3 with a structured prompt asking for:
       - Overall sentiment score (0.0 to 1.0)
       - Sentiment label (positive / neutral / negative)
       - 2-3 sentence summary of key news themes
    4. Parses llama3's JSON response
    5. Writes results to AgentState

WHY THIS AGENT EXISTS:
    News sentiment is a leading indicator — it reflects what the market
    is thinking RIGHT NOW about a company, before it shows up in
    financial statements. A company with great fundamentals but terrible
    recent news (regulatory probe, CEO resignation, product recall)
    needs that context in the research report.

WHAT IT WRITES TO AGENTSTATE:
    news_summary    → "NVIDIA dominates AI chip market with strong H100 demand..."
    sentiment_score → 0.78  (0.0 = very negative, 1.0 = very positive)
    sentiment_label → "positive" | "neutral" | "negative"
    articles_count  → 10

TEMPERATURE CHOICE — 0.2:
    We use temperature=0.2 for sentiment analysis.
    Not 0.0 because the summary should read naturally, not robotically.
    Not 0.5+ because we want consistent, factual analysis not creativity.
    0.2 gives us consistent results with natural-sounding prose.

ERROR HANDLING PHILOSOPHY:
    If Tavily returns 0 articles OR llama3 fails:
    → Write None values to state (not crash)
    → Add error message to state.errors
    → Mark agent as completed anyway
    The pipeline continues — a report without news is better than no report.

LANGSMITH TRACING:
    LangSmith traces all LLM calls automatically when these env vars are set:
        LANGCHAIN_API_KEY     → your LangSmith API key
        LANGCHAIN_TRACING_V2  → "true"
        LANGCHAIN_PROJECT      → project name in LangSmith dashboard
    No code changes needed — LangSmith hooks into the Ollama calls.
    You'll see token counts, latency, and prompts in the LangSmith UI.

USED BY:
    src/pipeline/graph.py → registered as a node in the LangGraph graph

HOW TO RUN:
    cd financial-research-agent
    python -m src.agents.research_agent
"""

from curses import raw
import json
from src.utils.llm import call_llm
from src.models.state import AgentState
from src.tools.news_search import search_company_news
from src.utils.logger import get_logger

logger = get_logger(__name__)
# ─────────────────────────────────────────────────────────────────────────────
# AGENT FUNCTION — research_agent
#
# WHY THIS FUNCTION SIGNATURE:
#   LangGraph requires every agent node to:
#       - Accept AgentState as input
#       - Return a DICT of only the fields it changed
#   LangGraph merges this dict back into the full state automatically.
#   We never return the full AgentState — only our updates.
# ─────────────────────────────────────────────────────────────────────────────

def research_agent(state: AgentState) -> dict:
    """
    Fetch recent news for the company and analyze sentiment using llama3.

    FLOW:
        state.company_name + state.ticker
            ↓
        search_company_news() → list of articles [{title, content, url}]
            ↓
        _format_articles() → single text block of all articles
            ↓
        _analyze_sentiment() → llama3 → {score, label, summary}
            ↓
        return dict of state updates

    Args:
        state: Current AgentState — reads company_name and ticker

    Returns:
        dict with keys: news_summary, sentiment_score, sentiment_label,
                        articles_count, completed_agents, errors
    """
    logger.info(
        f"Research Agent starting for "
        f"{state.company_name} ({state.ticker})"
    )

    # ── Step 1: Fetch news articles ───────────────────────────────────────────
    # search_company_news() calls Tavily API and returns top 10 articles.
    # Each article is a dict with: title, content, url, score
    try:
        result   = search_company_news(
            company_name = state.company_name,
            ticker       = state.ticker,
        )
        # search_company_news() returns a dict with an 'articles' key
        # not a plain list — extract the articles list from it
        articles = result.get('articles', []) if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        return {
            "news_summary":    None,
            "sentiment_score": None,
            "sentiment_label": None,
            "articles_count":  0,
            "completed_agents": ["research_agent"],
            "errors": [f"Research Agent: news fetch failed — {e}"],
        }

    if not articles:
        logger.warning(
            f"No news articles found for {state.ticker} — "
            f"returning neutral sentiment"
        )
        return {
            "news_summary":    "No recent news articles found.",
            "sentiment_score": 0.5,
            "sentiment_label": "neutral",
            "articles_count":  0,
            "completed_agents": ["research_agent"],
            "errors": [f"Research Agent: no articles found for {state.ticker}"],
        }

    logger.info(f"Fetched {len(articles)} articles for {state.ticker}")

    # ── Step 2: Format articles into one text block ───────────────────────────
    # We combine all article titles and content into a single string.
    # This gives llama3 all the context it needs in one prompt.
    articles_text = _format_articles(articles)

    # ── Step 3: Analyze sentiment with llama3 ─────────────────────────────────
    # Send the articles to llama3 and ask for structured JSON response.
    # Returns dict with: sentiment_score, sentiment_label, summary
    sentiment = _analyze_sentiment(
        articles_text = articles_text,
        company_name  = state.company_name,
        ticker        = state.ticker,
    )

    # ── Step 4: Return state updates ──────────────────────────────────────────
    # LangGraph merges this dict into AgentState automatically.
    # We include "completed_agents" so the orchestrator knows we're done.
    # operator.add in AgentState means ["research_agent"] gets APPENDED
    # to the existing list, not replacing it.
    logger.info(
        f"Research Agent complete — "
        f"sentiment={sentiment['label']} ({sentiment['score']}), "
        f"articles={len(articles)}"
    )

    return {
        "news_summary":    sentiment["summary"],
        "sentiment_score": sentiment["score"],
        "sentiment_label": sentiment["label"],
        "articles_count":  len(articles),
        "completed_agents": ["research_agent"],
        "errors": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _format_articles
# Formats a list of article dicts into a single readable text block.
# This text block is what gets sent to llama3.
#
# WHY FORMAT THIS WAY:
#   llama3 reads the articles as plain text — not as a Python list.
#   A numbered list with clear Article X / Title / Content labels
#   helps llama3 understand the structure and analyze each article
#   individually before forming an overall judgment.
# ─────────────────────────────────────────────────────────────────────────────

def _format_articles(articles: list[dict]) -> str:
    """
    Format article dicts into a numbered text block for llama3.

    WHY TRUNCATE CONTENT TO 500 CHARS:
        A full article can be 5,000+ chars.
        10 articles × 5,000 chars = 50,000 chars — too long for llama3.
        The first 500 chars of each article captures the key facts.
        Headlines and opening paragraphs contain 80% of the news value.

    Args:
        articles: List of dicts with keys: title, content, url

    Returns:
        Formatted string with all articles numbered and labeled
    """
    formatted = []

    for i, article in enumerate(articles, 1):
        title   = article.get("title", "No title")
        content = article.get("content", "No content")

        # Truncate content to first 500 chars
        # First paragraph always has the most important information
        content_preview = content[:500] if len(content) > 500 else content

        formatted.append(
            f"Article {i}:\n"
            f"Title: {title}\n"
            f"Content: {content_preview}\n"
        )

    return "\n---\n".join(formatted)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _analyze_sentiment
# Sends formatted articles to llama3 and asks for structured sentiment analysis.
#
# WHY JSON RESPONSE FORMAT:
#   We need to read sentiment_score (a float) and sentiment_label (a string)
#   in Python code. Plain English responses are unparseable — is "mostly
#   positive with some concerns" positive or neutral? JSON is unambiguous.
#
# WHY TEMPERATURE 0.2:
#   Not 0.0 — the summary should read naturally, not robotically.
#   Not 0.5+ — we want consistent, factual analysis not creativity.
#   0.2 gives consistent results with natural-sounding prose.
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_sentiment(
    articles_text: str,
    company_name:  str,
    ticker:        str,
) -> dict:
    """
    Send articles to llama3 and get structured sentiment analysis back.

    PROMPT DESIGN:
        We give llama3:
            1. Clear role: "You are a financial analyst"
            2. The articles to analyze
            3. Exact JSON format to respond in
            4. Field definitions so it knows what we mean by each field

        We ask for:
            sentiment_score → float 0.0 to 1.0
                0.0 = extremely negative (bankruptcy, fraud, disaster)
                0.5 = neutral (mixed news, no clear direction)
                1.0 = extremely positive (record revenue, major wins)

            sentiment_label → one of: "positive" | "neutral" | "negative"
                positive  → score >= 0.6
                neutral   → score 0.4 to 0.6
                negative  → score <= 0.4

            summary → 2-3 sentences capturing the key news themes
                This goes directly into the final research report.
                Should be professional, factual, concise.

    FALLBACK ON FAILURE:
        If llama3 returns malformed JSON or the request times out,
        we return neutral sentiment (0.5) rather than crashing.
        The report will note that sentiment analysis was unavailable.

    Args:
        articles_text: Formatted string of all news articles
        company_name:  Full company name e.g. "NVIDIA Corporation"
        ticker:        Stock ticker e.g. "NVDA"

    Returns:
        dict with: score (float), label (str), summary (str)
    """

    # ── Build the prompt ──────────────────────────────────────────────────────
    # Key design decisions:
    #   - Give it a clear role ("financial news analyst")
    #   - Show the exact JSON structure with field descriptions
    #   - Tell it explicitly: no markdown, no preamble, JSON only
    #   - Define the sentiment scale so it's not ambiguous
    prompt = f"""You are a financial news analyst. Analyze the following recent news articles about {company_name} ({ticker}) and provide a sentiment assessment.

NEWS ARTICLES:
{articles_text}

Based on these articles, analyze the overall market sentiment toward {company_name}.

Respond ONLY in this exact JSON format with no extra text, no markdown backticks, no explanation:
{{
    "sentiment_score": <float between 0.0 and 1.0 where 0.0=very negative, 0.5=neutral, 1.0=very positive>,
    "sentiment_label": <"positive" or "neutral" or "negative">,
    "summary": "<2-3 sentences summarizing the key news themes and their implications for the company>"
}}"""

    # ── Call llama3 via Ollama ────────────────────────────────────────────────
    try:
        raw = call_llm(prompt=prompt, temperature=0.2, max_tokens=200)
        if not raw:
            raise RuntimeError("LLM returned empty response")
        logger.info(f"llama3 sentiment raw response: {raw[:200]}")

        # ── Strip markdown code fences if present ─────────────────────────────
        # llama3 sometimes wraps JSON in ```json ... ``` despite instructions
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # ── Extract JSON object using regex ───────────────────────────────────
        # WHY REGEX INSTEAD OF json.loads() DIRECTLY:
        #   llama3 sometimes adds text after the closing } or includes
        #   an unescaped character inside the summary string that breaks
        #   json.loads(). Extracting just the {...} block first makes
        #   parsing much more robust.
        #   re.DOTALL → makes . match newlines too (summary spans multiple lines)
        import re as _re
        json_match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found", raw, 0)
        
        # ── Clean common llama3 JSON issues ──────────────────────────────────
        # llama3 sometimes puts unescaped newlines inside string values
        # which breaks json.loads(). We clean the extracted JSON block.
        json_str = json_match.group()
        
        # Parse JSON response
        parsed = json.loads(json_str)

        score   = float(parsed.get("sentiment_score", 0.5))
        label   = parsed.get("sentiment_label", "neutral").lower()
        summary = parsed.get("summary", "Sentiment analysis unavailable.")

        # Truncate summary to 500 chars — prevents excessively long
        # summaries that cause JSON parsing issues on subsequent calls
        summary = summary[:500] if len(summary) > 500 else summary

        # ── Validate score is in range ────────────────────────────────────────
        # Clamp to [0.0, 1.0] in case llama3 returns something out of range
        score = max(0.0, min(1.0, score))

        # ── Validate label is one of the expected values ──────────────────────
        if label not in ("positive", "neutral", "negative"):
            # Derive label from score if llama3 returned unexpected string
            if score >= 0.6:
                label = "positive"
            elif score <= 0.4:
                label = "negative"
            else:
                label = "neutral"

        return {
            "score":   round(score, 3),
            "label":   label,
            "summary": summary,
        }

    except json.JSONDecodeError as e:
        # llama3 returned non-JSON despite instructions
        logger.warning(f"Sentiment JSON parse failed: {e} — returning neutral")
        return {
            "score":   0.5,
            "label":   "neutral",
            "summary": "Sentiment analysis unavailable — JSON parse error.",
        }

    except requests.exceptions.Timeout:
        logger.warning("llama3 timed out during sentiment analysis")
        return {
            "score":   0.5,
            "label":   "neutral",
            "summary": "Sentiment analysis unavailable — request timed out.",
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {
            "score":   0.5,
            "label":   "neutral",
            "summary": f"Sentiment analysis unavailable — {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
#
# WHAT THIS TESTS:
#   1. Tavily API key is working — can fetch real news
#   2. llama3 is running — can analyze sentiment
#   3. JSON parsing works — structured output is correct
#   4. Full agent function works end to end
#
# HOW TO RUN:
#   cd financial-research-agent
#   python -m src.agents.research_agent
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
 
    print(f"\n{'='*60}")
    print(f"  Research Agent — Sanity Check")
    print(f"{'='*60}")
 
    # ── Test 1: news fetch ────────────────────────────────────────────────────
    print(f"\n── Test 1: Fetch news articles ──────────────────────────────\n")
 
    from src.tools.news_search import search_company_news
    result   = search_company_news("NVIDIA Corporation", "NVDA")
    articles = result.get("articles", []) if isinstance(result, dict) else result
 
    print(f"  Articles fetched : {len(articles)}")
    if articles:
        print(f"  First title      : {articles[0].get('title', 'N/A')[:80]}")
        print(f"  First URL        : {articles[0].get('url', 'N/A')[:80]}")
 
    # ── Test 2: Sentiment analysis ────────────────────────────────────────────
    print(f"\n── Test 2: Sentiment analysis via llama3 ────────────────────\n")
 
    if articles:
        articles_text = _format_articles(articles)
        sentiment     = _analyze_sentiment(articles_text, "NVIDIA Corporation", "NVDA")
 
        print(f"  Sentiment score  : {sentiment['score']}")
        print(f"  Sentiment label  : {sentiment['label']}")
        print(f"  Summary          : {sentiment['summary'][:200]}")
 
    # ── Test 3: Full agent function ───────────────────────────────────────────
    print(f"\n── Test 3: Full agent run via AgentState ────────────────────\n")
 
    state  = AgentState(company_name="NVIDIA Corporation", ticker="NVDA")
    result = research_agent(state)
 
    print(f"  news_summary     : {str(result.get('news_summary', ''))[:150]}")
    print(f"  sentiment_score  : {result.get('sentiment_score')}")
    print(f"  sentiment_label  : {result.get('sentiment_label')}")
    print(f"  articles_count   : {result.get('articles_count')}")
    print(f"  completed_agents : {result.get('completed_agents')}")
    print(f"  errors           : {result.get('errors')}")
    print()