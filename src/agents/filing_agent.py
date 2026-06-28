"""
src/agents/filing_agent.py
==========================
Agent 3 of 4: Extracts insights from SEC 10-K filing via RAG pipeline.

WHAT IT WRITES TO AGENTSTATE:
    revenue_breakdown  → "Data Center 88%, Gaming 8%..."
    risk_factors       → ["Export controls", "Competition from AMD"...]
    management_outlook → "Management expects continued AI-driven demand..."
    filing_source      → "SEC 10-K FY2026 — NVIDIA CORP"

FLOW:
    1. Check ChromaDB — if NVDA chunks exist, skip embedding
    2. If not: fetch 10-K → clean → chunk → embed → store
    3. Run 3 targeted retrieval queries (revenue, risks, outlook)
    4. For each query: send retrieved chunks to llama3 for extraction
    5. Write structured answers to AgentState

HOW TO RUN:
    cd financial-research-agent
    python -m src.agents.filing_agent
"""

import json
from src.utils.llm import call_llm
from src.models.state import AgentState
from src.tools.sec_edgar import fetch_latest_10k
from src.rag.document_loader import load_document
from src.rag.chunker import chunk_document
from src.rag.embedder import embed_chunks
from src.rag.vector_store import store, ticker_exists
from src.rag.retriever import retrieve
from src.utils.logger import get_logger

logger = get_logger(__name__)



# ─────────────────────────────────────────────────────────────────────────────
# AGENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def filing_agent(state: AgentState) -> dict:
    """
    Run the full RAG pipeline and extract 3 key insights from the 10-K.

    Args:
        state: reads company_name, ticker

    Returns:
        dict with revenue_breakdown, risk_factors, management_outlook,
        filing_source, completed_agents, errors
    """
    logger.info(f"Filing Agent starting for {state.company_name} ({state.ticker})")

    filing_source = None
    errors        = []
    eval_data = [] 
    # ── Step 1: Embed and store if not already in ChromaDB ───────────────────
    # ticker_exists() checks if chunks are already stored.
    # Skipping re-embedding saves 2-3 minutes on repeated runs.
    if ticker_exists(state.ticker):
        logger.info(f"Chunks already in ChromaDB for {state.ticker} — skipping embedding")
    else:
        logger.info(f"No chunks found for {state.ticker} — running full RAG pipeline")

        try:
            # Fetch → clean → chunk → embed → store
            filing        = fetch_latest_10k(state.ticker)
            filing_source = filing.get("source") or f"SEC 10-K — {state.company_name}"
            doc           = load_document(filing)
            chunks        = chunk_document(doc)

            logger.info(f"Chunked into {len(chunks)} chunks — embedding now")

            embedded = embed_chunks(chunks)
            if not embedded:
                raise RuntimeError("Embedding returned 0 chunks — Ollama may not be running")

            store(embedded)
            logger.info(f"Stored {len(embedded)} chunks in ChromaDB")

        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return {
                "revenue_breakdown":  None,
                "risk_factors":       None,
                "management_outlook": None,
                "filing_source":      None,
                "completed_agents":   ["filing_agent"],
                "errors": [f"Filing Agent: RAG pipeline failed — {e}"],
            }

    # ── Step 2: Retrieve and extract 3 insights ───────────────────────────────
    # Each question targets a different section of the filing.
    # We ask 3 focused questions instead of one broad query so each
    # retrieval returns clean, relevant chunks without mixing topics.

    queries = [
        {
            "key":     "revenue_breakdown",
            "question": f"What are {state.company_name}'s main business segments and how did each segment perform? Describe revenue trends, growth drivers, and segment contributions.",
            # No section filter — revenue numbers may be split across sections
            # Searching all sections gives better chance of finding actual figures
            "section":  None,
        },
        {
            "key":     "risk_factors",
            "question": f"What are the top risk factors facing {state.company_name}? List the most significant risks.",
            "section":  "risk_factors",
        },
        {
            "key":     "management_outlook",
            "question": f"What is {state.company_name} management's outlook and forward guidance for future performance?",
            "section":  "management_discussion",
        },
    ]

    results = {}

    for query in queries:
        logger.info(f"Retrieving context for: {query['key']}")

        try:
            # Retrieve relevant chunks from ChromaDB
            retrieval = retrieve(
                question       = query["question"],
                ticker         = state.ticker,
                section_filter = query["section"],
            )

            if not retrieval.context:
                logger.warning(f"No context retrieved for {query['key']}")
                results[query["key"]] = None
                errors.append(f"Filing Agent: no context for {query['key']}")
                continue

            # Extract insight using llama3
            insight = _extract_insight(
                question = query["question"],
                context  = retrieval.formatted_context(),
                key      = query["key"],
            )

            results[query["key"]] = insight
            # Save evaluation data for RAGAS
            eval_data.append({
                "question": query["question"],
                "answer":   insight or "",
                "contexts": retrieval.contexts,  # list of individual chunk texts
            })
            logger.info(f"Extracted {query['key']}: {str(insight)[:100]}")

        except Exception as e:
            logger.error(f"Extraction failed for {query['key']}: {e}")
            results[query["key"]] = None
            errors.append(f"Filing Agent: {query['key']} extraction failed — {e}")

    # ── Step 3: Format risk_factors as a list ─────────────────────────────────
    # risk_factors in AgentState is list[str] not a plain string.
    # If llama3 returned a string, split it into a list by newlines or bullets.
    risk_raw = results.get("risk_factors")
    if isinstance(risk_raw, str):
        # Split on newlines, bullets, or numbered lists
        import re
        items = re.split(r"\n+|•|\d+\.", risk_raw)
        items = [item.strip() for item in items if len(item.strip()) > 20]
        results["risk_factors"] = items[:5]  # top 5 risks only

    logger.info(
        f"Filing Agent complete — "
        f"revenue={'✅' if results.get('revenue_breakdown') else '❌'}, "
        f"risks={'✅' if results.get('risk_factors') else '❌'}, "
        f"outlook={'✅' if results.get('management_outlook') else '❌'}"
    )

    return {
        "revenue_breakdown":  results.get("revenue_breakdown"),
        "risk_factors":       results.get("risk_factors"),
        "management_outlook": results.get("management_outlook"),
        "filing_source":      filing_source or f"SEC 10-K — {state.company_name}",
        "completed_agents":   ["filing_agent"],
        "errors":             errors,
        "eval_data":          eval_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _extract_insight
# Sends retrieved context to llama3 and extracts a specific insight.
# ─────────────────────────────────────────────────────────────────────────────

def _extract_insight(
    question: str,
    context:  str,
    key:      str,
) -> str:
    """
    Send retrieved SEC filing context to llama3 and extract a specific insight.

    TEMPERATURE 0.2:
        We want accurate extraction with natural wording.
        Lower than sentiment (0.2 vs 0.2) for the same reason —
        factual extraction needs consistency not creativity.

    Args:
        question: The specific question to answer
        context:  Retrieved chunks from ChromaDB (formatted)
        key:      Which insight we're extracting (for logging)

    Returns:
        Extracted insight as a string, or None if extraction failed
    """
    # Truncate context to prevent exceeding llama3's context window
    # 3000 chars ≈ 750 tokens — enough for meaningful extraction
    context_preview = context[:3000] if len(context) > 3000 else context

    prompt = f"""You are a financial analyst extracting specific information from an SEC 10-K filing.

{context_preview}

Based ONLY on the context above, answer this question concisely:
{question}

Rules:
- Use only information from the context above
- Be specific — include numbers, percentages, dollar amounts where available
- Keep your answer to 3-5 sentences maximum
- Do not make up information not in the context
- If the context doesn't contain enough information, say so briefly

Answer:"""

    try:
        answer = call_llm(prompt=prompt, temperature=0.2, max_tokens=250)
        if not answer:
            return None

        # Remove any "Answer:" prefix llama3 might include
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        return answer

    except Exception as e:
        logger.error(f"llama3 extraction failed for {key}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Filing Agent — Sanity Check")
    print(f"{'='*60}")

    # ── Test 1: Full agent run ────────────────────────────────────────────────
    print(f"\n── Test 1: Full agent run via AgentState ────────────────────\n")

    state  = AgentState(company_name="NVIDIA Corporation", ticker="NVDA")
    result = filing_agent(state)

    print(f"\n  filing_source    : {result.get('filing_source')}")
    print(f"  completed_agents : {result.get('completed_agents')}")
    print(f"  errors           : {result.get('errors')}")

    print(f"\n  revenue_breakdown:")
    print(f"  {str(result.get('revenue_breakdown', 'None'))[:300]}")

    print(f"\n  risk_factors:")
    risks = result.get("risk_factors") or []
    for i, risk in enumerate(risks[:3], 1):
        print(f"  {i}. {str(risk)[:120]}")

    print(f"\n  management_outlook:")
    print(f"  {str(result.get('management_outlook', 'None'))[:300]}")

    print()