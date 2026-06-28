# Agentic retrieval — loops until enough context is found
"""
src/rag/retriever.py
====================
RAG Pipeline Step 5 (Final): Agentic retrieval loop.

ROLE IN THE RAG PIPELINE:
    sec_edgar.py       → fetches raw 10-K text
    document_loader.py → cleans and structures the text
    chunker.py         → splits into overlapping chunks
    embedder.py        → converts chunks to vectors
    vector_store.py    → saves/queries vectors in ChromaDB
    retriever.py       → loops until sufficient context found  ← YOU ARE HERE

WHAT THIS FILE DOES (main goal):
    The Filing Agent needs to answer questions like:
        "What is NVIDIA's revenue breakdown by segment?"
        "What are the main risk factors?"

    This file finds the most relevant chunks from ChromaDB to answer
    those questions, and keeps trying with better queries until it
    has enough context — or hits the max iteration limit.

HOW THE LOOP WORKS:
    1. Search ChromaDB with the question → get top 4 chunks
    2. Send question + chunks to llama3:
           "Is this enough to answer the question? yes/no"
    3. If YES → stop, return everything collected
       If NO  → llama3 also tells us what better query to try next
    4. Search ChromaDB again with the better query
    5. Add new chunks to what we already have (skip duplicates)
    6. Repeat up to MAX_ITERATIONS times

WHY LLAMA3 FOR SUFFICIENCY CHECK:
    llama3 reads the question AND the retrieved context together.
    It understands language well enough to judge:
        - "Revenue breakdown by segment" needs ALL segments
        - If only Data Center is in the context → not sufficient
        - Suggest: "NVIDIA gaming automotive revenue 2024"
    This is just prompt engineering — we give it both pieces and ask
    it to fill in a JSON object telling us if one answers the other.

ONE FUNCTION:
    retrieve(question, ticker, section_filter) -> RetrievalResult

USED BY:
    src/agents/filing_agent.py -> calls retrieve() to get SEC filing context

HOW TO RUN:
    ollama serve      ← start Ollama first
    cd financial-research-agent
    python -m src.rag.retriever

DEPENDENCIES:
    pip install requests chromadb
    ollama pull mxbai-embed-large
    ollama pull llama3
"""

import json

from PIL.ImagePalette import raw
from src.utils.llm import call_llm
from dataclasses import dataclass, field
from typing import Optional
from src.rag.vector_store import search
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
#
# WHY CONSTANTS AT THE TOP:
#   Single source of truth — change the model or URL in one place,
#   not scattered across the file.
# ─────────────────────────────────────────────────────────────────────────────

# Ollama's chat endpoint — used for the sufficiency check


# Maximum number of retrieval loops before we stop regardless
# 3 is enough: first query gets obvious stuff, second fills gaps,
# third is the last attempt. Without this limit, a bad document
# could loop forever.
MAX_ITERATIONS = 3

# How many chunks to fetch per ChromaDB search query
# 4 chunks × 3 iterations = up to 12 unique chunks maximum
# That's enough context for a detailed answer without overwhelming the LLM
CHUNKS_PER_QUERY = 4


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL RESULT DATACLASS
#
# WHY A DATACLASS INSTEAD OF A PLAIN DICT:
#   The Filing Agent needs several things from retrieval:
#       - the context text (to pass to llama3 for report writing)
#       - citations (to include in the report)
#       - source URLs (for attribution)
#       - metadata (iterations, sufficient, etc. for debugging)
#
#   Packing all of this into one structured object means:
#       - The Filing Agent imports one thing and has everything
#       - Fields are named and typed — no guessing what's in the dict
#       - formatted_context() method assembles the final prompt block
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Everything the Filing Agent needs from the retrieval process.

    Produced by retrieve() and consumed by filing_agent.py.
    """

    # The original question that was asked
    question:     str

    # The company ticker we searched for
    ticker:       str

    # All retrieved chunk texts joined together
    # This is the "context" that gets passed to llama3 for report writing
    context:      str

    # Citation string for each chunk used
    # e.g. "SEC 10-K FY2024 — NVIDIA CORP | risk_factors | chunk 3/12"
    citations:    list[str]

    contexts: list[str]  # individual chunk texts for RAGAS evaluation

    # How many unique chunks were collected across all iterations
    chunks_used:  int

    # How many retrieval loops actually ran (1 to MAX_ITERATIONS)
    iterations:   int

    # Did we find enough context? True if llama3 said sufficient,
    # or if we hit MAX_ITERATIONS (we return what we have regardless)
    sufficient:   bool

    # All queries we used — original + any follow-up queries
    # Useful for debugging: "what did the retriever actually search for?"
    queries_used: list[str]

    # Direct URLs to the SEC filing documents for attribution
    filing_urls:  list[str]

    def formatted_context(self) -> str:
        """
        Assembles context + citations into a clean block ready to
        paste directly into the Filing Agent's LLM prompt.

        WHY A METHOD HERE INSTEAD OF IN filing_agent.py:
            The retriever knows the structure of its own output.
            It makes sense for it to also know how to format it.
            The Filing Agent just calls this and gets a ready-to-use string.

        Example output:
            RETRIEVED CONTEXT FROM SEC 10-K FILING (NVDA):
            ────────────────────────────────────────────────────────────
            Data Center revenue grew 217%...

            ---

            Export controls restrict sales to China...
            ────────────────────────────────────────────────────────────
            CITATIONS:
            [1] SEC 10-K FY2024 — NVIDIA CORP | management_discussion | chunk 1/12
            [2] SEC 10-K FY2024 — NVIDIA CORP | risk_factors | chunk 2/12
        """
        citations_block = "\n".join(
            f"[{i+1}] {cite}" for i, cite in enumerate(self.citations)
        )
        return (
            f"RETRIEVED CONTEXT FROM SEC 10-K FILING ({self.ticker}):\n"
            f"{'─' * 60}\n"
            f"{self.context}\n"
            f"{'─' * 60}\n"
            f"CITATIONS:\n{citations_block}\n"
            f"{'─' * 60}\n"
            f"Source: SEC EDGAR · {self.ticker} · 10-K Annual Filing\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — retrieve
# The main agentic retrieval loop.
# This is the only function filing_agent.py calls.
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    question:       str,
    ticker:         str,
    section_filter: Optional[str] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> RetrievalResult:
    """
    Search ChromaDB for relevant chunks, check if sufficient,
    and retry with a better query if not.

    DEDUPLICATION — WHY A DICT KEYED BY TEXT:
        Across multiple iterations, the same chunk can appear
        in multiple search results. We use a dict keyed by chunk text
        to automatically skip chunks we already have.

        Why dict and not set?
            A set stores only the text.
            A dict stores text → full result (score, citation, url, etc.)
            We need the full result, not just the text.

        Example:
            Iteration 1 finds: [chunk_A, chunk_B, chunk_C, chunk_D]
            Iteration 2 finds: [chunk_B, chunk_C, chunk_E, chunk_F]
            After dedup:       [chunk_A, chunk_B, chunk_C, chunk_D, chunk_E, chunk_F]
            chunk_B and chunk_C appear only once in the final context.

    Args:
        question:       What the Filing Agent needs to know
                        e.g. "What is revenue breakdown by business segment?"
        ticker:         Company ticker — only search chunks from this company
        section_filter: Optional — restrict search to one section
                        e.g. "risk_factors", "management_discussion"
        max_iterations: Safety limit on retry loops (default 3)

    Returns:
        RetrievalResult with context, citations, and metadata
    """
    logger.info(
        f"Agentic retrieval starting — "
        f"ticker={ticker}, "
        f"question='{question[:80]}', "
        f"max_iterations={max_iterations}"
    )

    # ── collected: our deduplication dict ────────────────────────────────────
    # Key   = chunk text (string)
    # Value = full result dict {text, score, citation, metadata, filing_url}
    # New chunks are only added if their text is not already a key here.
    collected    = {}
    queries_used = []

    # Start with the original question as the first query
    current_query = question

    # ── Main retrieval loop ───────────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):

        logger.info(
            f"Iteration {iteration}/{max_iterations} — "
            f"query: '{current_query[:80]}'"
        )

        # ── Step 1: Search ChromaDB ───────────────────────────────────────────
        # search() converts current_query to a vector using mxbai-embed-large
        # then finds the CHUNKS_PER_QUERY most similar chunks in ChromaDB
        results = search(
            query          = current_query,
            ticker         = ticker,
            n_results      = CHUNKS_PER_QUERY,
            section_filter = section_filter,
        )

        # Record this query so we can debug what was searched later
        queries_used.append(current_query)

        if not results:
            logger.warning(
                f"Iteration {iteration}: ChromaDB returned 0 results. "
                f"Are chunks stored for {ticker}? Run vector_store.py first."
            )
            break

        # ── Step 2: Add new chunks to collected (deduplicate) ─────────────────
        new_count = 0
        for result in results:
            # If this chunk text is already in collected, skip it
            # dict key lookup is O(1) — fast even with many chunks
            if result["text"] not in collected:
                collected[result["text"]] = result
                new_count += 1

        logger.info(
            f"Iteration {iteration}: found {len(results)} chunks, "
            f"{new_count} new — total unique: {len(collected)}"
        )

        # ── Step 3: On the last iteration, stop without sufficiency check ──────
        # No point asking llama3 "is this sufficient?" on the last iteration
        # because we're returning regardless — saves one LLM call.
        if iteration == max_iterations:
            logger.info(
                f"Max iterations ({max_iterations}) reached — "
                f"returning {len(collected)} chunks"
            )
            break

        # ── Step 4: Check sufficiency with llama3 ─────────────────────────────
        # Build the context string from everything collected so far
        # and ask llama3 if it's enough to answer the original question
        context_so_far = "\n\n".join(r["text"] for r in collected.values())

        sufficient, next_query = _check_sufficiency(
            question  = question,
            context   = context_so_far,
            iteration = iteration,
        )

        if sufficient:
            logger.info(
                f"Sufficiency check: SUFFICIENT — "
                f"stopping after {iteration} iteration(s)"
            )
            break
        else:
            # Not sufficient — use the follow-up query llama3 suggested
            logger.info(
                f"Sufficiency check: NOT SUFFICIENT — "
                f"next query: '{next_query[:80]}'"
            )
            current_query = next_query

    # ── Assemble the final RetrievalResult ────────────────────────────────────

    # Convert collected dict values to a sorted list
    # Sort by score descending — most relevant chunks appear first in context
    # This matters because LLMs pay more attention to early context
    all_chunks = list(collected.values())
    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    # Join all chunk texts with a separator so the LLM can see boundaries
    context   = "\n\n---\n\n".join(r["text"]      for r in all_chunks)
    citations = [r["citation"]                     for r in all_chunks]

    # Collect unique filing URLs (use a set to deduplicate, convert back to list)
    filing_urls = list({r["filing_url"] for r in all_chunks if r["filing_url"]})
    contexts = [r["text"] for r in all_chunks] 
    result = RetrievalResult(
        question     = question,
        ticker       = ticker,
        context      = context,
        contexts     = contexts, 
        citations    = citations,
        chunks_used  = len(all_chunks),
        iterations   = len(queries_used),
        sufficient   = len(all_chunks) >= CHUNKS_PER_QUERY,
        queries_used = queries_used,
        filing_urls  = filing_urls,
    )

    logger.info(
        f"Retrieval complete — "
        f"{result.chunks_used} chunks, "
        f"{result.iterations} iterations, "
        f"sufficient={result.sufficient}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — _check_sufficiency
# Asks llama3 whether the current context answers the question.
# If not, asks it to suggest a better follow-up query.
#
# WHY PRIVATE (prefixed with _):
#   This is an implementation detail of retrieve().
#   External code should never call this directly.
#
# HOW IT WORKS:
#   We send llama3 a prompt containing:
#       1. The original question
#       2. The context retrieved so far
#       3. Instructions to respond in JSON only
#
#   llama3 reads both, understands what's missing, and fills in:
#       {"sufficient": true/false, "follow_up_query": "..."}
#
#   This is pure prompt engineering — no special API, no fine-tuning.
#   Just a well-structured prompt that guides the model to give us
#   a parseable, actionable response.
# ─────────────────────────────────────────────────────────────────────────────

def _check_sufficiency(
    question:  str,
    context:   str,
    iteration: int,
) -> tuple[bool, str]:
    """
    Ask llama3: "Is this context enough to answer the question?"

    TEMPERATURE=0 — WHY:
        Temperature controls how random the model's word choices are.
        0.0 = always picks the most likely word = deterministic output.
        We use 0.0 here because this is a yes/no judgment call.
        We want the same answer every time for the same input.
        Randomness would make sufficiency checks inconsistent.

    WHY JSON RESPONSE FORMAT:
        We need to read the model's answer in Python code.
        Plain English like "Yes I think it's sufficient" is hard to parse —
        do we search for "yes"? What if it says "yes, but..."?
        JSON is unambiguous: {"sufficient": true} is always true.
        We instruct the model to respond ONLY in JSON, then parse it.

    STRIPPING MARKDOWN BACKTICKS:
        llama3 sometimes wraps JSON in markdown code fences:
            ```json
            {"sufficient": true}
            ```
        We strip those before calling json.loads() to avoid parse errors.

    FALLBACK ON FAILURE:
        If llama3 returns malformed JSON or the request fails entirely,
        we assume NOT sufficient and repeat the original question.
        Better to do one extra search than to stop too early and miss
        important context.

    Args:
        question:  The original question from the Filing Agent
        context:   All context collected so far (joined chunk texts)
        iteration: Current iteration number (for logging only)

    Returns:
        Tuple of:
            sufficient    (bool)  → True if context is enough
            follow_up     (str)   → Better query to try if not sufficient
    """

    # Truncate context to 3000 chars before sending to llama3
    # WHY: We only need enough context for llama3 to judge sufficiency.
    # Sending the full context (could be 10,000+ chars) wastes tokens
    # and slows down the response. 3000 chars ≈ 750 tokens — plenty
    # for a yes/no judgment.
    context_preview = context[:3000] if len(context) > 3000 else context

    # ── Build the prompt ──────────────────────────────────────────────────────
    # Key design decisions:
    #   - Give it the question first so it knows what to evaluate against
    #   - Give it the context second so it can compare
    #   - Be explicit: "respond ONLY in this JSON format"
    #   - Show the exact JSON structure with field names
    #   - Explain what follow_up_query should contain
    prompt = f"""You are evaluating whether retrieved context from a SEC 10-K filing is sufficient to answer a research question.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_preview}

Is this context sufficient to give a detailed and accurate answer to the question above?

Respond ONLY in this exact JSON format with no extra text, no markdown, no explanation:
{{
    "sufficient": true or false,
    "reason": "one sentence explaining why sufficient or not",
    "follow_up_query": "a better search query to find missing information, or empty string if sufficient"
}}"""

    # ── Call Ollama's chat API ────────────────────────────────────────────────
    try:
        raw = call_llm(prompt=prompt, temperature=0, max_tokens=200)

        if not raw:
            raise RuntimeError("LLM returned empty response")
        
        logger.info(f"llama3 sufficiency raw response: {raw[:200]}")

        # ── Strip markdown code fences if present ─────────────────────────────
        # llama3 sometimes wraps JSON in ```json ... ``` even when told not to
        # We strip those to get clean JSON before parsing
        if "```" in raw:
            # Split on ``` and take the middle part
            parts = raw.split("```")
            # parts[1] is the content between the first pair of backticks
            raw = parts[1]
            # If it starts with "json", remove that language identifier
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # ── Parse the JSON response ───────────────────────────────────────────
        parsed      = json.loads(raw)
        sufficient  = bool(parsed.get("sufficient", False))
        follow_up   = parsed.get("follow_up_query", "").strip()
        reason      = parsed.get("reason", "")

        logger.info(
            f"Sufficiency check (iteration {iteration}): "
            f"sufficient={sufficient}, reason='{reason}'"
        )

        # If not sufficient but no follow-up query provided,
        # append "details" to the original question as a fallback
        if not sufficient and not follow_up:
            follow_up = f"{question} detailed breakdown"

        return sufficient, follow_up

    except json.JSONDecodeError as e:
        # llama3 returned something that isn't valid JSON
        # This happens occasionally — the model ignores formatting instructions
        logger.warning(
            f"Sufficiency check JSON parse failed (iteration {iteration}): {e} — "
            f"assuming NOT sufficient, retrying with original question"
        )
        return False, f"{question} additional details"

    except requests.exceptions.Timeout:
        # llama3 took too long — skip sufficiency check, keep going
        logger.warning(
            f"Sufficiency check timed out (iteration {iteration}) — "
            f"assuming NOT sufficient"
        )
        return False, f"{question} more information"

    except Exception as e:
        # Any other error — log it and continue rather than crashing
        logger.warning(
            f"Sufficiency check failed (iteration {iteration}): {e} — "
            f"assuming NOT sufficient"
        )
        return False, f"{question} additional details"


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
#
# WHAT THIS TESTS:
#   1. Full retrieval loop with mock chunks already in ChromaDB
#   2. Sufficiency check works (llama3 responds with valid JSON)
#   3. Deduplication works (same chunk doesn't appear twice)
#   4. formatted_context() output looks correct
#
# PREREQUISITES:
#   - Ollama must be running:         ollama serve
#   - llama3 must be pulled:          ollama pull llama3
#   - mxbai-embed-large must exist:   ollama pull mxbai-embed-large
#   - Run vector_store.py first to store mock chunks in ChromaDB
#
# HOW TO RUN:
#   cd financial-research-agent
#   python -m src.rag.retriever
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Retriever — Sanity Check")
    print(f"{'='*60}")

    # ── Setup: store mock chunks in ChromaDB first ────────────────────────────
    print(f"\n── Setup: storing mock chunks ───────────────────────────────\n")

    from src.rag.chunker import Chunk
    from src.rag.embedder import embed_chunks
    from src.rag.vector_store import store

    mock_chunks = [
        Chunk(
            text         = "NVIDIA Data Center revenue grew 217% to $47.5B. Compute $45.1B, Networking $2.4B. Growth driven by H100 and H200 GPU sales to Microsoft, Google, Amazon, Meta.",
            token_count  = 38,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 0,
            total_chunks = 5,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Gaming segment revenue decreased 12% to $6.2B in fiscal 2024. Decline driven by inventory normalization after crypto mining boom.",
            token_count  = 30,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 1,
            total_chunks = 5,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Export control regulations restrict NVIDIA's ability to sell A100, H100 chips to China. Impacted approximately $5B in potential revenue in fiscal 2024.",
            token_count  = 32,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "risk_factors",
            chunk_index  = 0,
            total_chunks = 5,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Competition from AMD MI300X and custom silicon from Google TPUs and Amazon Trainium represents long-term substitution risk for NVIDIA data center dominance.",
            token_count  = 34,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "risk_factors",
            chunk_index  = 1,
            total_chunks = 5,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Gross margin for fiscal 2024 was 72.7%, up from 56.9% in fiscal 2023. Improvement driven by product mix shift toward high-margin Data Center GPUs.",
            token_count  = 33,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 2,
            total_chunks = 5,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
    ]

    print(f"  Embedding {len(mock_chunks)} chunks...")
    embedded     = embed_chunks(mock_chunks)
    store_result = store(embedded)
    print(f"  Stored {store_result['stored_count']} chunks for NVDA ✅")

    # ── Test 1: Retrieve revenue breakdown ────────────────────────────────────
    print(f"\n── Test 1: Revenue breakdown query ──────────────────────────\n")

    result = retrieve(
        question = "What is NVIDIA's revenue breakdown by business segment?",
        ticker   = "NVDA",
    )

    print(f"  Chunks used   : {result.chunks_used}")
    print(f"  Iterations    : {result.iterations}")
    print(f"  Sufficient    : {result.sufficient}")
    print(f"  Queries used  :")
    for i, q in enumerate(result.queries_used, 1):
        print(f"    {i}. {q}")

    print(f"\n  Context preview (first 300 chars):")
    print(f"  {result.context[:300]}...")

    print(f"\n  Citations:")
    for cite in result.citations:
        print(f"    • {cite}")

    # ── Test 2: Section filtered retrieval ───────────────────────────────────
    print(f"\n── Test 2: Risk factors (section filter) ────────────────────\n")

    risk_result = retrieve(
        question       = "What are the main risks facing NVIDIA?",
        ticker         = "NVDA",
        section_filter = "risk_factors",
    )

    print(f"  Chunks used   : {risk_result.chunks_used}")
    print(f"  All from risk_factors: {all('risk_factors' in c for c in risk_result.citations)}")

    # ── Test 3: formatted_context output ─────────────────────────────────────
    print(f"\n── Test 3: formatted_context() ──────────────────────────────\n")
    print(result.formatted_context()[:500])
    print("...")
    print()