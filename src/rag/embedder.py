# Converts text chunks into vectors using OpenAI embeddings
"""
src/rag/embedder.py
===================
RAG Pipeline Step 3: Converts Chunk objects into vector embeddings.

ROLE IN THE RAG PIPELINE:
    sec_edgar.py       → fetches raw 10-K text
    document_loader.py → cleans and structures the text
    chunker.py         → splits into overlapping chunks
    embedder.py        → converts chunks to vectors      ← YOU ARE HERE
    vector_store.py    → saves vectors to ChromaDB

WHAT IS AN EMBEDDING:
    An embedding is a list of floating point numbers that represents
    the "meaning" of a piece of text in a high-dimensional space.

    Example:
        "NVIDIA revenue grew 122%" → [0.023, -0.451, 0.882, ...]  (1024 numbers)
        "NVIDIA sales increased"   → [0.021, -0.448, 0.879, ...]  (very similar)
        "Apple launches iPhone"    → [0.341,  0.212, -0.103, ...]  (very different)

    Two pieces of text that mean the same thing have similar vectors.
    That's how the retriever finds relevant chunks given a query.

MODEL USED:
    mxbai-embed-large (via Ollama)
    - 1024 dimensions
    - Runs fully locally — no API key, no cost, no internet required
    - Strong retrieval quality, better than smaller open-source alternatives
    - Ollama serves it over a local HTTP API at http://localhost:11434

WHY OLLAMA:
    Ollama runs open-source models locally as an HTTP server.
    Instead of sending your data to OpenAI's servers, everything
    stays on your machine. The API is simple: POST request in,
    embedding vectors out.

BATCHING:
    A 10-K filing produces 300-500 chunks.
    We batch chunks into groups of 50 and embed each batch together.
    Ollama processes batches sequentially on CPU — smaller batches (50)
    prevent memory pressure compared to sending all 500 at once.

TWO THINGS THIS FILE EXPORTS:
    EmbeddedChunk   → dataclass: Chunk + its embedding vector
    embed_chunks()  → takes list[Chunk], returns list[EmbeddedChunk]

USED BY:
    src/rag/vector_store.py → receives EmbeddedChunk objects for storage

HOW TO RUN:
    # Make sure Ollama is running first:
    ollama serve

    # Then run:
    cd financial-research-agent
    python -m src.rag.embedder

DEPENDENCIES:
    pip install requests
    ollama pull mxbai-embed-large
"""

import time
import requests
from dataclasses import dataclass
from typing import Optional
from src.rag.chunker import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
#
# WHY CONSTANTS AT THE TOP:
#   If you want to switch models later (e.g. nomic-embed-text),
#   you change ONE line here instead of hunting through the whole file.
#   This is called the "single source of truth" principle.
# ─────────────────────────────────────────────────────────────────────────────

# Ollama runs a local HTTP server on this address by default.
# If you changed Ollama's port, update this.
OLLAMA_BASE_URL  = "http://localhost:11434"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

# The embedding model to use.
# Must be pulled first: ollama pull mxbai-embed-large
EMBED_MODEL = "mxbai-embed-large"

# mxbai-embed-large produces 1024-dimensional vectors.
# We validate this after the first batch — if dimensions don't match,
# a different model is likely running and we fail immediately.
EMBEDDING_DIMS = 1024

# Number of chunks to send per Ollama request.
# Smaller than cloud APIs (100) because Ollama runs locally on CPU —
# smaller batches prevent memory pressure and keep things responsive.
BATCH_SIZE = 50


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDEDCHUNK DATACLASS
#
# WHY A SEPARATE DATACLASS INSTEAD OF MODIFYING CHUNK:
#   Chunk is the output of chunker.py — it has no embedding yet.
#   Adding an embedding field directly to Chunk would mean chunker.py
#   knows about embeddings, which is not its job.
#   EmbeddedChunk is a new object that combines both concerns cleanly.
#
#   This is called "separation of concerns" — each class does one job.
#   Chunk = splitting. EmbeddedChunk = splitting + vectorized.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddedChunk:
    """
    A Chunk object with its embedding vector attached.
    Produced by embed_chunks() and consumed by vector_store.py.
    """

    # ── All original fields from Chunk ────────────────────────────────────────
    # We copy every field from Chunk so vector_store.py only needs to
    # import EmbeddedChunk — it doesn't need Chunk at all.
    text:         str
    token_count:  int
    ticker:       str
    source:       str
    filing_date:  str
    section:      str
    chunk_index:  int
    total_chunks: int
    filing_url:   Optional[str]

    # ── New field: the embedding vector ───────────────────────────────────────
    # 1024 floats for mxbai-embed-large.
    # This is what ChromaDB stores and searches against.
    embedding: list[float]

    def citation(self) -> str:
        """
        Returns a human-readable citation string for this chunk.
        Same format as Chunk.citation() — keeps the interface consistent.

        Example output:
            "SEC 10-K FY2024 — NVIDIA CORP | risk_factors | chunk 3/12"
        """
        return (
            f"{self.source} | "
            f"{self.section} | "
            f"chunk {self.chunk_index + 1}/{self.total_chunks}"
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk, embedding: list[float]) -> "EmbeddedChunk":
        """
        Factory method: create an EmbeddedChunk from a Chunk + embedding.

        WHY A CLASSMETHOD FACTORY:
            Instead of manually copying all 9 fields from Chunk,
            this factory does it in one readable call:
                EmbeddedChunk.from_chunk(chunk, embedding)

            If Chunk ever gains new fields, we only update this method —
            not every place that creates EmbeddedChunk objects.

        Args:
            chunk:     The source Chunk object
            embedding: The 1024-float vector from Ollama

        Returns:
            EmbeddedChunk with all Chunk fields + embedding attached
        """
        return cls(
            text         = chunk.text,
            token_count  = chunk.token_count,
            ticker       = chunk.ticker,
            source       = chunk.source,
            filing_date  = chunk.filing_date,
            section      = chunk.section,
            chunk_index  = chunk.chunk_index,
            total_chunks = chunk.total_chunks,
            filing_url   = chunk.filing_url,
            embedding    = embedding,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — _check_ollama
# Verifies Ollama is running before we attempt any embedding.
#
# WHY A DEDICATED CHECK:
#   If Ollama isn't running and we go straight to embedding,
#   the error is: "ConnectionRefusedError: [Errno 111] Connection refused"
#   buried deep in a stack trace — hard to understand.
#
#   With this check, the error is:
#   "Ollama is not running. Start it with: ollama serve"
#   — immediately actionable.
#
#   This is called "fail fast" — catch problems at the entry point
#   with a clear message rather than letting them surface as confusing
#   errors deep inside the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama() -> None:
    """
    Ping Ollama's health endpoint to verify it is running.

    HOW IT WORKS:
        Ollama exposes GET http://localhost:11434 which returns
        "Ollama is running" if the server is up.
        We just check for a 200 response — content doesn't matter.

    Raises:
        RuntimeError: with a clear message if Ollama is not reachable
    """
    try:
        # timeout=3 → don't wait more than 3 seconds
        # If Ollama isn't running, this raises ConnectionRefusedError
        # which we catch below and convert to a clear RuntimeError
        response = requests.get(OLLAMA_BASE_URL, timeout=3)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama returned unexpected status {response.status_code}. "
                f"Try restarting: ollama serve"
            )

        logger.info("Ollama is running ✅")

    except requests.exceptions.ConnectionError:
        # Happens when Ollama process is not running at all
        raise RuntimeError(
            f"\n\nOllama is not running.\n"
            f"Start it with:  ollama serve\n"
            f"Then verify:    ollama list\n"
            f"Model needed:   ollama pull {EMBED_MODEL}\n"
        )

    except requests.exceptions.Timeout:
        # Happens when Ollama is starting up and not yet responsive
        raise RuntimeError(
            f"Ollama did not respond within 3 seconds. "
            f"It may still be starting up — wait a moment and try again."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — _embed_batch
# Sends one batch of texts to Ollama and returns embedding vectors.
#
# WHY PRIVATE (prefixed with _):
#   This is an implementation detail of embed_chunks().
#   External code should only call embed_chunks() — never this directly.
#   The underscore prefix is a Python convention for "internal use only".
#
# WHY NO @retry DECORATOR:
#   Unlike a remote API, Ollama runs locally.
#   If it fails, it's not a transient network blip — something is actually
#   wrong (model not loaded, out of memory, etc).
#   Retrying automatically would hide real problems.
#   embed_chunks() handles batch failures with try/except instead.
# ─────────────────────────────────────────────────────────────────────────────

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Send one batch of texts to Ollama and return their embedding vectors.

    HOW OLLAMA'S EMBED API WORKS:
        Request (POST to /api/embed):
            {
                "model": "mxbai-embed-large",
                "input": ["text1", "text2", "text3"]
            }

        Response:
            {
                "embeddings": [
                    [0.023, -0.451, 0.882, ...],   ← vector for text1 (1024 floats)
                    [0.341,  0.212, -0.103, ...],  ← vector for text2
                    [0.019, -0.388, 0.751, ...],   ← vector for text3
                ]
            }

        Ollama guarantees response order matches input order.
        So embeddings[0] always corresponds to texts[0].

    DIMENSION VALIDATION:
        After getting vectors back, we check the first vector has
        exactly EMBEDDING_DIMS (1024) dimensions.
        If wrong, a different model is running — we fail clearly
        rather than storing wrong-sized vectors into ChromaDB silently.
        Wrong dimensions in ChromaDB would corrupt all future searches.

    Args:
        texts: List of text strings to embed (max BATCH_SIZE)

    Returns:
        List of embedding vectors, one per input text.
        Each vector is a list of EMBEDDING_DIMS floats.

    Raises:
        RuntimeError: if Ollama returns wrong dimensions or empty response
        requests.HTTPError: if Ollama returns a non-200 status
    """

    # ── Build the request payload ─────────────────────────────────────────────
    # "model" → which Ollama model to use
    # "input" → list of strings (Ollama's /api/embed accepts a list)
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }

    # ── POST to Ollama ────────────────────────────────────────────────────────
    # timeout=120 → large batches on CPU can take up to 2 minutes
    # raise_for_status() → raises HTTPError for 4xx/5xx responses
    response = requests.post(
        OLLAMA_EMBED_URL,
        json    = payload,
        timeout = 120,
    )
    response.raise_for_status()

    # ── Parse the response ────────────────────────────────────────────────────
    data       = response.json()
    embeddings = data.get("embeddings", [])

    if not embeddings:
        raise RuntimeError(
            f"Ollama returned empty embeddings for model '{EMBED_MODEL}'. "
            f"Is the model fully loaded? Try: ollama run {EMBED_MODEL}"
        )

    # ── Validate dimensions on first vector ───────────────────────────────────
    # If the model loaded is different from EMBED_MODEL, dimensions
    # will be wrong and ChromaDB will reject them — better to fail here
    # with a clear message than fail silently inside ChromaDB.
    actual_dims = len(embeddings[0])
    if actual_dims != EMBEDDING_DIMS:
        raise RuntimeError(
            f"Dimension mismatch: expected {EMBEDDING_DIMS} dims "
            f"but got {actual_dims} dims from model '{EMBED_MODEL}'. "
            f"Check that the correct model is loaded in Ollama."
        )

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — embed_chunks
# Public function: takes Chunk objects, returns EmbeddedChunk objects.
# This is the ONLY function the rest of the pipeline calls.
#
# THOUGHT PROCESS BEHIND THE DESIGN:
#   1. Check Ollama is running FIRST — fail fast before doing any work
#   2. Process in batches — don't send 500 chunks in one request
#   3. If ONE batch fails, skip it and continue — partial embeddings
#      are better than crashing the whole pipeline for one bad batch
#   4. Log progress — embedding 500 chunks takes time on CPU,
#      the user should see progress not a hanging terminal
# ─────────────────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """
    Convert a list of Chunk objects into EmbeddedChunk objects.

    This is the main entry point for the embedder.
    Called by the Filing Agent after chunking the 10-K document.

    BATCHING STRATEGY:
        If we have 250 chunks and BATCH_SIZE=50:
            Batch 1: chunks[0:50]    → 1 Ollama call → 50  embeddings
            Batch 2: chunks[50:100]  → 1 Ollama call → 50  embeddings
            Batch 3: chunks[100:150] → 1 Ollama call → 50  embeddings
            Batch 4: chunks[150:200] → 1 Ollama call → 50  embeddings
            Batch 5: chunks[200:250] → 1 Ollama call → 50  embeddings
        Total: 5 Ollama calls instead of 250.

    ERROR RECOVERY:
        If batch 3 fails (e.g. Ollama runs out of memory):
            → Log the error clearly
            → Skip that batch
            → Continue with batch 4
        Result: 200 embedded chunks instead of 250.
        Better than crashing and getting 0.

    Args:
        chunks: List of Chunk objects from chunk_document()

    Returns:
        List of EmbeddedChunk objects.
        May be shorter than input if some batches failed.
    """

    if not chunks:
        logger.warning("embed_chunks called with empty list — nothing to embed")
        return []

    # ── Step 1: Verify Ollama is running before doing any work ───────────────
    # Gives a clear error message if Ollama isn't started yet.
    # Without this check, the first _embed_batch() call fails with a
    # confusing low-level ConnectionRefusedError.
    _check_ollama()

    logger.info(
        f"Embedding {len(chunks)} chunks using {EMBED_MODEL} "
        f"(batch_size={BATCH_SIZE}, dims={EMBEDDING_DIMS})"
    )

    embedded_chunks = []
    total_batches   = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    # ── Step 2: Process chunks in batches ────────────────────────────────────
    for batch_num, batch_start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):

        # Slice the current batch from the full chunks list.
        # min() handles the last batch which may be smaller than BATCH_SIZE.
        batch_end   = min(batch_start + BATCH_SIZE, len(chunks))
        batch       = chunks[batch_start:batch_end]

        # Extract just the text — that's all Ollama needs.
        # Metadata (ticker, section, etc.) stays in the Chunk object.
        batch_texts = [chunk.text for chunk in batch]

        logger.info(
            f"Embedding batch {batch_num}/{total_batches} — "
            f"chunks {batch_start}–{batch_end} ({len(batch)} chunks)"
        )

        # ── Step 3: Call Ollama for this batch ────────────────────────────────
        # Wrapped in try/except so one failed batch doesn't kill everything.
        try:
            embeddings = _embed_batch(batch_texts)

        except Exception as e:
            logger.error(
                f"Batch {batch_num}/{total_batches} failed: {e} — "
                f"skipping {len(batch)} chunks and continuing"
            )
            continue

        # ── Step 4: Attach each embedding to its chunk ────────────────────────
        # zip(batch, embeddings) pairs chunk[0] with embeddings[0], etc.
        # This is safe because Ollama preserves input order in its response.
        for chunk, embedding in zip(batch, embeddings):
            embedded_chunk = EmbeddedChunk.from_chunk(chunk, embedding)
            embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Batch {batch_num}/{total_batches} complete — "
            f"{len(embedded_chunks)} total embedded so far"
        )

        # ── Step 5: Small pause between batches ──────────────────────────────
        # Gives Ollama a moment to free memory between batches.
        # Local models benefit from slightly more breathing room than
        # cloud APIs.
        if batch_num < total_batches:
            time.sleep(0.2)

    logger.info(
        f"Embedding complete — "
        f"{len(embedded_chunks)}/{len(chunks)} chunks embedded successfully"
    )

    return embedded_chunks


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
#
# WHAT THIS TESTS:
#   1. Ollama is reachable
#   2. mxbai-embed-large is loaded and returns vectors
#   3. Vectors are exactly 1024 dimensions
#   4. Semantic similarity works — similar texts score higher than
#      unrelated texts (proves the model understands meaning)
#
# HOW TO RUN:
#   ollama serve          ← start Ollama first
#   cd financial-research-agent
#   python -m src.rag.embedder
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Embedder — Sanity Check (Ollama / {EMBED_MODEL})")
    print(f"{'='*60}")

    # ── Test 1: Ollama health check ───────────────────────────────────────────
    print(f"\n── Test 1: Ollama health check ──────────────────────────────\n")

    try:
        _check_ollama()
        print(f"  Ollama status  : Running ✅")
        print(f"  Embed URL      : {OLLAMA_EMBED_URL}")
        print(f"  Model          : {EMBED_MODEL}")
        print(f"  Expected dims  : {EMBEDDING_DIMS}")
    except RuntimeError as e:
        print(f"  ❌ {e}")
        exit(1)

    # ── Test 2: Embed 3 mock chunks ───────────────────────────────────────────
    print(f"\n── Test 2: Embed 3 mock chunks ──────────────────────────────\n")

    from src.rag.chunker import Chunk

    mock_chunks = [
        Chunk(
            text         = "NVIDIA Data Center revenue grew 217% year over year to $47.5 billion, driven by H100 GPU demand from cloud hyperscalers.",
            token_count  = 25,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 0,
            total_chunks = 3,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "Export controls imposed by the US government restrict sales of H100 chips to China. This could materially reduce future revenue.",
            token_count  = 24,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "risk_factors",
            chunk_index  = 0,
            total_chunks = 3,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
        Chunk(
            text         = "NVIDIA revenue increased significantly due to strong data center GPU sales and growing AI infrastructure demand worldwide.",
            token_count  = 22,
            ticker       = "NVDA",
            source       = "SEC 10-K FY2024 — NVIDIA CORP",
            filing_date  = "2024-02-21",
            section      = "management_discussion",
            chunk_index  = 1,
            total_chunks = 3,
            filing_url   = "https://sec.gov/archives/nvda-10k-2024.htm",
        ),
    ]

    print(f"  Embedding {len(mock_chunks)} chunks...")
    embedded = embed_chunks(mock_chunks)

    print(f"\n  Chunks in       : {len(mock_chunks)}")
    print(f"  Chunks embedded : {len(embedded)}")

    for ec in embedded:
        print(f"\n  Citation        : {ec.citation()}")
        print(f"  Section         : {ec.section}")
        print(f"  Dimensions      : {len(ec.embedding)}")
        print(f"  First 5 values  : {[round(v, 4) for v in ec.embedding[:5]]}")

    # ── Test 3: Dimension validation ──────────────────────────────────────────
    print(f"\n── Test 3: Dimension validation ─────────────────────────────\n")

    all_correct = all(len(ec.embedding) == EMBEDDING_DIMS for ec in embedded)
    print(f"  All chunks have {EMBEDDING_DIMS} dims : {'✅ Yes' if all_correct else '❌ No'}")

    # ── Test 4: Semantic similarity ───────────────────────────────────────────
    # chunk 0 (revenue) and chunk 2 (revenue) should be MORE similar
    # than chunk 0 (revenue) and chunk 1 (export controls / risk)
    print(f"\n── Test 4: Semantic similarity check ────────────────────────\n")
    print(f"  chunk 0 = revenue growth (management discussion)")
    print(f"  chunk 1 = export controls (risk factors)")
    print(f"  chunk 2 = revenue / AI demand (management discussion)")
    print(f"  Expected: chunk 0 vs chunk 2 > chunk 0 vs chunk 1\n")

    if len(embedded) == 3:
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            """
            Cosine similarity between two vectors.
            1.0  = identical meaning
            0.0  = completely unrelated
            -1.0 = opposite meaning
            """
            dot   = sum(x * y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

        sim_0_1 = cosine_similarity(embedded[0].embedding, embedded[1].embedding)
        sim_0_2 = cosine_similarity(embedded[0].embedding, embedded[2].embedding)

        print(f"  Revenue vs Export Controls : {sim_0_1:.4f}")
        print(f"  Revenue vs AI Demand       : {sim_0_2:.4f}")

        if sim_0_2 > sim_0_1:
            print(f"\n  ✅ Similarity check passed — model understands meaning")
        else:
            print(f"\n  ⚠️  Unexpected result — both scores are close, model may still be correct")

    print()