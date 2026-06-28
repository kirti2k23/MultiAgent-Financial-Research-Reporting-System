# Splits filing text into 1000-token chunks with metadata
"""
src/rag/chunker.py
==================
RAG Pipeline Step 2: Splits the cleaned Document into overlapping chunks.

ROLE IN THE RAG PIPELINE:
    sec_edgar.py       → fetches raw 10-K text
    document_loader.py → cleans and structures the text
    chunker.py         → splits into overlapping chunks  ← YOU ARE HERE
    embedder.py        → converts chunks to vectors
    vector_store.py    → saves vectors to ChromaDB

WHY CHUNKING:
    Embedding models and LLMs have token limits.
    A 10-K filing is 300,000–500,000 characters (~100,000 tokens).
    GPT-4o context window: 128,000 tokens max.
    Embedding models: 8,192 tokens max.
    We cannot process the whole document at once — we chunk it.

WHY OVERLAP:
    Without overlap, sentences at chunk boundaries get split in half.
    A sentence starting at the end of chunk 1 might finish in chunk 2 —
    neither chunk has the complete thought.
    Overlap copies the tail of each chunk into the start of the next,
    ensuring every sentence is fully captured in at least one chunk.

CHUNK SETTINGS:
    chunk_size    = 400 tokens  (reduced from 1000 to fit mxbai-embed-large context limit)
    chunk_overlap = 80   tokens  (20% overlap — standard recommendation)

SECTION-AWARE CHUNKING:
    We chunk each section (business, risk_factors, etc.) separately.
    This prevents chunks from mixing content across sections.
    Every chunk knows which section it came from → better retrieval.

USED BY:
    src/rag/embedder.py -> receives list of Chunk objects

HOW TO RUN:
    cd financial-research-agent
    python -m src.rag.chunker
"""

import tiktoken
from dataclasses import dataclass
from typing import Optional
from src.rag.document_loader import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK DATACLASS
# A single piece of the document, ready for embedding.
# Carries both the text AND metadata for citation and filtering.
#
# WHY METADATA MATTERS:
#   When the retriever finds the top-5 most relevant chunks,
#   it needs to tell the Filing Agent WHERE each chunk came from.
#   Without metadata, you get an answer but no citation.
#   With metadata, you get: "Revenue grew 217% [Source: SEC 10-K FY2024,
#   management_discussion, chunk 12/47]"
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single chunk of document text, ready for embedding and storage.
    """

    # ── Content ───────────────────────────────────────────────────────────────
    text:        str    # The actual text content of this chunk
    token_count: int    # How many tokens this chunk contains

    # ── Source metadata ───────────────────────────────────────────────────────
    ticker:      str    # e.g. "NVDA"
    source:      str    # e.g. "SEC 10-K FY2024 — NVIDIA CORP"
    filing_date: str    # e.g. "2024-02-21"
    section:     str    # e.g. "risk_factors" | "management_discussion"

    # ── Position metadata ─────────────────────────────────────────────────────
    chunk_index:  int   # Index of this chunk within its section (0-based)
    total_chunks: int   # Total chunks in this section

    # ── Optional ──────────────────────────────────────────────────────────────
    filing_url:   Optional[str] = None   # Direct link to the SEC filing

    def citation(self) -> str:
        """
        Returns a human-readable citation string for this chunk.
        Used by the Filing Agent when it quotes from this chunk.

        Example output:
            "SEC 10-K FY2024 — NVIDIA CORP | risk_factors | chunk 3/12"
        """
        return (
            f"{self.source} | "
            f"{self.section} | "
            f"chunk {self.chunk_index + 1}/{self.total_chunks}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — chunk_document
# Takes a Document and returns a list of Chunk objects.
# Processes each section separately for section-aware chunking.
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(
    doc:          Document,
    chunk_size:   int = 400,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """
    Split a Document into overlapping Chunk objects.

    STRATEGY — SECTION-AWARE CHUNKING:
        We process each section (business, risk_factors, etc.) separately.
        This means chunks never mix content from different sections.
        The full document text is also chunked as a fallback if
        sections were not successfully extracted.

    HOW TOKEN COUNTING WORKS:
        We use tiktoken — OpenAI's tokenizer — to count tokens accurately.
        tiktoken.encoding_for_model("gpt-4o") gives us GPT-4o's tokenizer.
        This ensures our chunks align with the same token limits GPT-4o uses.

        Why not just use characters?
            Characters ≠ tokens. "NVIDIA" = 1 token, "Supercalifragilistic" = 6.
            Measuring in characters would make some chunks too big for the model.
            Measuring in tokens is exact.

    Args:
        doc:          Document object from document_loader.load_document()
        chunk_size:   Max tokens per chunk. Default 1000.
        chunk_overlap: Tokens to repeat at chunk boundaries. Default 200.

    Returns:
        List of Chunk objects sorted by section and chunk_index.
    """
    logger.info(
        f"Chunking document for {doc.ticker} — "
        f"chunk_size={chunk_size}, overlap={chunk_overlap}"
    )

    # ── Step 1: Initialize tiktoken encoder ───────────────────────────────────
    # "cl100k_base" is the encoding used by GPT-4o and text-embedding-3 models
    # This ensures our chunk sizes match what these models actually see
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"tiktoken encoding failed: {e} — falling back to word split")
        encoder = None

    # ── Step 2: Decide what to chunk ──────────────────────────────────────────
    # If sections were extracted → chunk each section separately
    # If no sections → chunk the full document as one block
    sections_found = {k: v for k, v in doc.sections.items() if v.strip()}

    all_chunks = []

    if sections_found:
        logger.info(f"Chunking {len(sections_found)} sections separately")

        for section_name, section_text in sections_found.items():
            section_chunks = _chunk_text(
                text          = section_text,
                chunk_size    = chunk_size,
                chunk_overlap = chunk_overlap,
                encoder       = encoder,
                doc           = doc,
                section       = section_name,
            )
            all_chunks.extend(section_chunks)
            logger.info(
                f"  Section '{section_name}': {len(section_chunks)} chunks"
            )

        # NOTE: We do NOT chunk the full document separately.
        # The 3 extracted sections (business, risk_factors, management_discussion)
        # cover all content needed for the research report.
        # Adding full_document would contaminate search results with
        # raw financial table noise (numbers without context).

    else:
        logger.warning(
            f"No sections extracted for {doc.ticker} — "
            f"chunking full document only"
        )
        all_chunks = _chunk_text(
            text          = doc.text,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            encoder       = encoder,
            doc           = doc,
            section       = "full_document",
        )

    logger.info(
        f"Chunking complete for {doc.ticker} — "
        f"{len(all_chunks)} total chunks across all sections"
    )

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _chunk_text
# Splits a single text string into overlapping chunks.
# Returns a list of Chunk objects for that section.
# Private function — only used inside this file.
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(
    text:          str,
    chunk_size:    int,
    chunk_overlap: int,
    encoder,
    doc:           Document,
    section:       str,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using token-based splitting.

    HOW TOKEN-BASED SPLITTING WORKS:
        1. Encode the full text into a list of token IDs:
           "NVIDIA revenue" → [45, 782, 19283]  (just example numbers)

        2. Slide a window of chunk_size tokens across the token list:
           Window 1: tokens[0:1000]
           Window 2: tokens[800:1800]   ← starts 200 back (overlap)
           Window 3: tokens[1600:2600]
           ...

        3. Decode each window back to text:
           [45, 782, 19283] → "NVIDIA revenue"

        This gives us chunks of exactly chunk_size tokens with
        chunk_overlap tokens repeated between consecutive chunks.

    FALLBACK — WORD-BASED SPLITTING:
        If tiktoken is unavailable, we fall back to splitting by words.
        Less precise but still functional.

    Args:
        text:          The text to chunk
        chunk_size:    Max tokens per chunk
        chunk_overlap: Tokens of overlap between consecutive chunks
        encoder:       tiktoken encoder (or None for word-based fallback)
        doc:           Source Document (for metadata)
        section:       Section name for metadata

    Returns:
        List of Chunk objects
    """
    if not text.strip():
        return []

    chunks = []

    if encoder:
        # ── Token-based chunking ──────────────────────────────────────────────
        # Encode full text to token IDs
        token_ids = encoder.encode(text)

        # Slide window across token IDs
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))

            # Decode this window back to text
            chunk_token_ids = token_ids[start:end]
            chunk_text      = encoder.decode(chunk_token_ids)
            token_count     = len(chunk_token_ids)

            chunks.append((chunk_text, token_count))

            # Move window forward by (chunk_size - overlap)
            # This creates the overlap between consecutive chunks
            if end == len(token_ids):
                break   # reached the end
            start += (chunk_size - chunk_overlap)

    else:
        # ── Word-based fallback ───────────────────────────────────────────────
        # Approximate: 1 token ≈ 0.75 words
        word_chunk_size    = int(chunk_size * 0.75)
        word_overlap_size  = int(chunk_overlap * 0.75)

        words = text.split()
        start = 0

        while start < len(words):
            end        = min(start + word_chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            word_count = end - start

            chunks.append((chunk_text, word_count))

            if end == len(words):
                break
            start += (word_chunk_size - word_overlap_size)

    # ── Build Chunk objects with metadata ─────────────────────────────────────
    total = len(chunks)
    chunk_objects = []

    for i, (chunk_text, token_count) in enumerate(chunks):
        chunk_obj = Chunk(
            text         = chunk_text.strip(),
            token_count  = token_count,
            ticker       = doc.ticker,
            source       = doc.source,
            filing_date  = doc.filing_date,
            section      = section,
            chunk_index  = i,
            total_chunks = total,
            filing_url   = doc.filing_url,
        )
        chunk_objects.append(chunk_obj)

    return chunk_objects


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
#
# Usage:
#   cd financial-research-agent
#   python -m src.rag.chunker
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Chunker — Sanity Check")
    print(f"{'='*60}")

    from src.tools.sec_edgar import fetch_latest_10k
    from src.rag.document_loader import load_document

    # ── Step 1: Fetch and load the document ───────────────────────────────────
    print(f"\n── Fetching NVDA 10-K ───────────────────────────────────────\n")
    filing = fetch_latest_10k("NVDA")
    doc    = load_document(filing)

    print(f"  Document loaded: {doc.text_length:,} characters")

    # ── Step 2: Chunk it ──────────────────────────────────────────────────────
    print(f"\n── Chunking document ────────────────────────────────────────\n")
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=80)

    print(f"  Total chunks     : {len(chunks)}")

    # ── Step 3: Show chunk distribution by section ────────────────────────────
    print(f"\n── Chunks by section ────────────────────────────────────────\n")
    from collections import Counter
    section_counts = Counter(c.section for c in chunks)
    for section, count in sorted(section_counts.items()):
        print(f"  {section:<25} : {count} chunks")

    # ── Step 4: Preview first and last chunk ──────────────────────────────────
    print(f"\n── First chunk preview ──────────────────────────────────────\n")
    first = chunks[0]
    print(f"  Citation    : {first.citation()}")
    print(f"  Token count : {first.token_count}")
    print(f"  Text preview: {first.text[:300]}...")

    print(f"\n── Overlap check (end of chunk 0 vs start of chunk 1) ───────\n")
    if len(chunks) >= 2:
        end_of_chunk_0   = chunks[0].text[-150:].strip()
        start_of_chunk_1 = chunks[1].text[:150].strip()
        print(f"  End of chunk 0  : ...{end_of_chunk_0}")
        print(f"  Start of chunk 1: {start_of_chunk_1}...")
        print(f"\n  (Some text should repeat between the two — that's the overlap)")

    # ── Step 5: Token count stats ─────────────────────────────────────────────
    print(f"\n── Token count stats ────────────────────────────────────────\n")
    token_counts = [c.token_count for c in chunks]
    print(f"  Min tokens   : {min(token_counts)}")
    print(f"  Max tokens   : {max(token_counts)}")
    print(f"  Avg tokens   : {sum(token_counts) // len(token_counts)}")
    print(f"  Total tokens : {sum(token_counts):,}")
    print()