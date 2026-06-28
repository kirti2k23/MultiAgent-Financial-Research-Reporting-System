# Saves and queries embeddings in ChromaDB
"""
src/rag/vector_store.py
=======================
RAG Pipeline Step 4: Saves and queries embeddings in ChromaDB.

ROLE IN THE RAG PIPELINE:
    sec_edgar.py       → fetches raw 10-K text
    document_loader.py → cleans and structures the text
    chunker.py         → splits into overlapping chunks
    embedder.py        → converts chunks to vectors
    vector_store.py    → saves vectors to ChromaDB     ← YOU ARE HERE

WHAT IS CHROMADB:
    ChromaDB is a local vector database — like SQLite but for embeddings.
    Instead of querying by exact match (WHERE ticker = 'NVDA'),
    you query by semantic similarity ("find chunks about export controls").

    It stores:
        - The embedding vector (1536 floats)
        - The chunk text (so you can read what was found)
        - Metadata (ticker, section, source, etc. — for filtering)

    And lets you ask:
        "Find the 5 chunks most semantically similar to this query embedding"

WHY LOCAL (not Pinecone, Weaviate, etc.):
    No API key, no cloud cost, no network latency.
    ChromaDB persists to disk — embeddings survive restarts.
    For single-company research, local is fast enough.

TWO MAIN OPERATIONS:
    store(embedded_chunks) → saves chunks to ChromaDB
    search(query_text, ticker, section_filter) → returns top-k relevant chunks

USED BY:
    src/rag/retriever.py -> calls search() to find relevant chunks

HOW TO RUN:
    cd financial-research-agent
    python -m src.rag.vector_store

DEPENDENCIES:
    pip install chromadb openai
"""

import chromadb
from chromadb.config import Settings
from src.rag.embedder import EmbeddedChunk, _embed_batch
from src.utils.config import config
from src.utils.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)

# ChromaDB collection name — all filings go into one collection
# Filtered by ticker metadata when searching
COLLECTION_NAME = "sec_filings"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — get_collection
# Returns the ChromaDB collection, creating it if it doesn't exist.
# Called at the start of both store() and search().
#
# WHY NOT A MODULE-LEVEL CLIENT:
#   If we create the client at import time, tests can't override the
#   persist directory. Creating it inside the function lets callers
#   (and tests) control where data is stored via config.
# ─────────────────────────────────────────────────────────────────────────────

def get_collection() -> chromadb.Collection:
    """
    Get or create the ChromaDB collection for SEC filings.

    HOW CHROMADB PERSISTENCE WORKS:
        chromadb.PersistentClient(path="./chroma_db")
            → Creates a folder at ./chroma_db
            → Stores all embeddings and metadata there
            → Survives Python restarts — no need to re-embed on every run

        client.get_or_create_collection(name)
            → Returns existing collection if it exists
            → Creates a new empty one if it doesn't
            → Safe to call multiple times — idempotent

    DISTANCE METRIC — cosine:
        ChromaDB supports: "cosine", "l2" (Euclidean), "ip" (inner product)
        We use cosine — it measures the ANGLE between vectors, not magnitude.
        Cosine is standard for text embeddings: two similar texts score ~1.0
        regardless of their length differences.

    Returns:
        chromadb.Collection ready for storing or querying
    """
    client = chromadb.PersistentClient(
        path     = config.chroma_persist_dir,
        settings = Settings(anonymized_telemetry=False),  # disable usage tracking
    )

    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},   # use cosine distance
    )

    logger.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready — "
        f"{collection.count()} documents stored at '{config.chroma_persist_dir}'"
    )

    return collection


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — store
# Saves a list of EmbeddedChunk objects into ChromaDB.
# ─────────────────────────────────────────────────────────────────────────────

def store(embedded_chunks: list[EmbeddedChunk]) -> dict:
    """
    Store embedded chunks in ChromaDB.

    HOW CHROMADB ADD WORKS:
        collection.add(
            ids        → unique string ID per document (required)
            embeddings → the vector for each document
            documents  → the text content (stored for retrieval)
            metadatas  → dict of filterable fields per document
        )

    ID FORMAT:
        {ticker}_{section}_{chunk_index}
        e.g. "NVDA_risk_factors_3"
        Must be unique. If you re-run for the same ticker, we first
        delete old documents for that ticker to avoid duplicates.

    METADATA FIELDS:
        ChromaDB lets you filter by metadata in search queries.
        We store: ticker, section, source, filing_date, filing_url, token_count
        This lets the retriever do: "find chunks for NVDA in risk_factors only"

    Args:
        embedded_chunks: List of EmbeddedChunk objects from embed_chunks()

    Returns:
        dict with: ticker, stored_count, collection_total
    """
    if not embedded_chunks:
        logger.warning("store() called with empty list — nothing to store")
        return {"stored_count": 0}

    ticker     = embedded_chunks[0].ticker
    collection = get_collection()

    # ── Step 1: Delete existing chunks for this ticker ────────────────────────
    # If we're re-running for the same company, remove old embeddings first.
    # This prevents duplicate chunks accumulating over multiple runs.
    try:
        existing = collection.get(where={"ticker": ticker})
        if existing["ids"]:
            collection.delete(where={"ticker": ticker})
            logger.info(
                f"Deleted {len(existing['ids'])} existing chunks for {ticker} "
                f"before re-storing"
            )
    except Exception as e:
        logger.warning(f"Could not check/delete existing chunks for {ticker}: {e}")

    # ── Step 2: Build the parallel arrays ChromaDB expects ────────────────────
    # ChromaDB's add() takes 4 parallel lists — all must be the same length
    # and in the same order.
    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk in embedded_chunks:
        # ID must be unique — use ticker + section + index
        chunk_id = f"{chunk.ticker}_{chunk.section}_{chunk.chunk_index}"

        ids.append(chunk_id)
        embeddings.append(chunk.embedding)
        documents.append(chunk.text)
        metadatas.append({
            "ticker":       chunk.ticker,
            "section":      chunk.section,
            "source":       chunk.source,
            "filing_date":  chunk.filing_date,
            "filing_url":   chunk.filing_url or "",
            "token_count":  chunk.token_count,
            "chunk_index":  chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
        })

    # ── Step 3: Add to ChromaDB ───────────────────────────────────────────────
    # ChromaDB handles batching internally — we can pass all chunks at once
    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    total = collection.count()

    logger.info(
        f"Stored {len(embedded_chunks)} chunks for {ticker} — "
        f"collection total: {total}"
    )

    return {
        "ticker":           ticker,
        "stored_count":     len(embedded_chunks),
        "collection_total": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — search
# Queries ChromaDB to find the most relevant chunks for a query.
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query:          str,
    ticker:         str,
    n_results:      int = 5,
    section_filter: str = None,
) -> list[dict]:
    """
    Find the top-k chunks most semantically similar to the query.

    HOW VECTOR SEARCH WORKS:
        1. Embed the query text → query vector (1536 floats)
        2. ChromaDB computes cosine similarity between the query vector
           and every stored embedding for this ticker
        3. Returns the top n_results chunks with highest similarity

    FILTERING:
        where={"ticker": ticker}
            → only search chunks from this company
        where={"ticker": ticker, "section": section_filter}
            → only search within a specific section (e.g. "risk_factors")

        This is important: without the ticker filter, you'd get chunks
        from completely different companies contaminating the results.

    RESULT FORMAT:
        Returns a list of dicts, each with:
            text        → the chunk text
            score       → similarity score (0.0 to 1.0, higher = more relevant)
            citation    → "SEC 10-K FY2024 | risk_factors | chunk 3/12"
            metadata    → all stored metadata fields
            filing_url  → direct link to the SEC filing

    Args:
        query:          The question or topic to search for
        ticker:         Only search chunks from this company
        n_results:      Number of chunks to return (default 5)
        section_filter: Optional section name to restrict search
                        e.g. "risk_factors", "management_discussion"

    Returns:
        List of result dicts sorted by relevance (most relevant first)
    """
    collection = get_collection()

    # ── Step 1: Check we have chunks for this ticker ──────────────────────────
    existing = collection.get(where={"ticker": ticker})
    if not existing["ids"]:
        logger.warning(f"No chunks found in ChromaDB for ticker {ticker}")
        return []

    logger.info(
        f"Searching for: '{query[:80]}...' "
        f"ticker={ticker}, section={section_filter or 'all'}, "
        f"n_results={n_results}"
    )

    # ── Step 2: Embed the query ───────────────────────────────────────────────
    # We embed the query using the same model used for the chunks.
    # The query vector must be in the same space as the chunk vectors.
    # client          = OpenAI(api_key=config.openai_api_key)
    query_embedding = _embed_batch([query])[0]

    # ── Step 3: Build metadata filter ────────────────────────────────────────
    # ChromaDB where clauses use dict syntax
    # Single condition: {"ticker": "NVDA"}
    # Multiple conditions: {"$and": [{"ticker": "NVDA"}, {"section": "risk_factors"}]}
    if section_filter:
        where = {
            "$and": [
                {"ticker":  {"$eq": ticker}},
                {"section": {"$eq": section_filter}},
            ]
        }
    else:
        where = {"ticker": {"$eq": ticker}}

    # ── Step 4: Query ChromaDB ────────────────────────────────────────────────
    # query_embeddings → list of query vectors (we send just one)
    # n_results        → how many chunks to return
    # where            → metadata filter
    # include          → what fields to include in the response
    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = min(n_results, len(existing["ids"])),
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    # ── Step 5: Format results ────────────────────────────────────────────────
    # ChromaDB returns nested lists (one per query — we sent one query)
    # results["documents"][0] → list of texts for our single query
    # results["distances"][0] → cosine distances (lower = more similar)
    # We convert distance to similarity: similarity = 1 - distance
    documents_list = results["documents"][0]
    metadatas_list = results["metadatas"][0]
    distances_list = results["distances"][0]

    formatted = []
    for text, metadata, distance in zip(documents_list, metadatas_list, distances_list):

        # Convert cosine distance to similarity score
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Similarity: 1 - (distance / 2) → maps to [0, 1]
        similarity = round(1 - (distance / 2), 4)

        citation = (
            f"{metadata.get('source', 'Unknown')} | "
            f"{metadata.get('section', 'Unknown')} | "
            f"chunk {metadata.get('chunk_index', 0) + 1}/"
            f"{metadata.get('total_chunks', '?')}"
        )

        formatted.append({
            "text":       text,
            "score":      similarity,
            "citation":   citation,
            "metadata":   metadata,
            "filing_url": metadata.get("filing_url", ""),
        })

    logger.info(
        f"Search returned {len(formatted)} chunks — "
        f"top score: {formatted[0]['score'] if formatted else 'N/A'}"
    )

    return formatted


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — ticker_exists
# Quick check: does ChromaDB have any chunks for this ticker?
# Used by the filing agent to skip re-embedding if already stored.
# ─────────────────────────────────────────────────────────────────────────────

def ticker_exists(ticker: str) -> bool:
    """
    Check if chunks for a ticker are already stored in ChromaDB.

    Used by the Filing Agent to avoid re-embedding on every pipeline run.
    If the ticker is already stored, we skip the embed → store steps
    and go straight to retrieval.

    Args:
        ticker: Stock symbol e.g. "NVDA"

    Returns:
        True if at least one chunk exists for this ticker, False otherwise
    """
    try:
        collection = get_collection()
        existing   = collection.get(where={"ticker": ticker})
        count      = len(existing["ids"])
        logger.info(f"ticker_exists({ticker}): {count} chunks in ChromaDB")
        return count > 0
    except Exception as e:
        logger.warning(f"ticker_exists check failed for {ticker}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to test store and search with mock data.
# Requires OPENAI_API_KEY in .env (for query embedding).
#
# Usage:
#   cd financial-research-agent
#   python -m src.rag.vector_store
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Vector Store — Sanity Check")
    print(f"{'='*60}")

    from src.rag.chunker import Chunk
    from src.rag.embedder import embed_chunks

    # ── Test 1: Build mock chunks and embed them ──────────────────────────────
    print(f"\n── Test 1: Store mock embedded chunks ───────────────────────\n")

    mock_chunks = [
        Chunk(
            text         = "NVIDIA Data Center revenue grew 217% year over year to $47.5 billion, driven by H100 GPU demand from cloud hyperscalers.",
            token_count  = 28,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 0,
            total_chunks = 4,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Export controls imposed by the US government restrict sales of advanced AI chips to China. This could materially reduce future revenue.",
            token_count  = 27,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "risk_factors",
            chunk_index  = 0,
            total_chunks = 4,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "NVIDIA was incorporated in the State of Delaware in April 1993. Principal executive offices are located in Santa Clara, California.",
            token_count  = 24,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "business",
            chunk_index  = 0,
            total_chunks = 4,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Competition from AMD MI300X and custom silicon from Google (TPUs) and Amazon (Trainium) represents an ongoing risk to NVIDIA's market share.",
            token_count  = 26,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "risk_factors",
            chunk_index  = 1,
            total_chunks = 4,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
    ]

    print(f"  Embedding {len(mock_chunks)} mock chunks...")
    embedded = embed_chunks(mock_chunks)
    print(f"  Embedded: {len(embedded)} chunks")

    result = store(embedded)
    print(f"  Stored: {result['stored_count']} chunks")
    print(f"  Collection total: {result['collection_total']}")

    # ── Test 2: Search — revenue query ────────────────────────────────────────
    print(f"\n── Test 2: Search — 'revenue breakdown by segment' ──────────\n")

    results = search(
        query   = "revenue breakdown by segment",
        ticker  = "NVDA",
        n_results = 3,
    )

    for i, r in enumerate(results, 1):
        print(f"  Result {i}:")
        print(f"    Score    : {r['score']}")
        print(f"    Section  : {r['metadata']['section']}")
        print(f"    Citation : {r['citation']}")
        print(f"    Preview  : {r['text'][:120]}...")
        print()

    # ── Test 3: Search with section filter ────────────────────────────────────
    print(f"\n── Test 3: Search filtered to risk_factors only ─────────────\n")

    risk_results = search(
        query          = "what are the main risks facing the company",
        ticker         = "NVDA",
        n_results      = 3,
        section_filter = "risk_factors",
    )

    for i, r in enumerate(risk_results, 1):
        print(f"  Result {i} (section={r['metadata']['section']}):")
        print(f"    Score   : {r['score']}")
        print(f"    Preview : {r['text'][:120]}...")
        print()

    # ── Test 4: ticker_exists ─────────────────────────────────────────────────
    print(f"\n── Test 4: ticker_exists ────────────────────────────────────\n")
    print(f"  NVDA exists: {ticker_exists('NVDA')}   (expected: True)")
    print(f"  AAPL exists: {ticker_exists('AAPL')}  (expected: False)")
    print()