# Downloads and parses SEC PDF filings into clean text
"""
src/rag/document_loader.py
==========================
RAG Pipeline Step 1: Loads and cleans the raw 10-K filing text.

ROLE IN THE RAG PIPELINE:
    sec_edgar.py       → fetches raw 10-K text from SEC EDGAR
    document_loader.py → cleans and structures the text  ← YOU ARE HERE
    chunker.py         → splits into smaller overlapping chunks
    embedder.py        → converts chunks to vectors
    vector_store.py    → saves vectors to ChromaDB

WHY CLEANING MATTERS:
    Raw SEC filings contain:
    - Page numbers scattered throughout
    - Repeated legal boilerplate headers/footers
    - Excessive blank lines and whitespace
    - HTML artifacts (leftover tags, &nbsp; entities)
    - Table formatting noise (dashes, pipes, underscores)

    If we embed dirty text, the retriever returns garbage.
    Clean text → meaningful embeddings → accurate retrieval.

WHY ONLY 3 SECTIONS (not 4):
    NVIDIA's 10-K stores actual financial tables in a separate exhibit
    file, not the main HTM document. Item 8 in the main file is just:
    "The information required by this Item is set forth in our
    Consolidated Financial Statements..."
    One sentence — no actual numbers.

    We use management_discussion (Item 7 MD&A) instead — it contains
    revenue figures, segment breakdown, margins, and management outlook
    in prose form, which is better for LLM analysis than raw tables.

HOW SECTION PATTERNS WERE DERIVED:
    We analyzed the actual NVIDIA 10-K HTML directly to find exact
    heading formats. Key findings:

    1. sec_edgar.py converts \xa0 (non-breaking space) to regular space
       so patterns only need \s not [\s\xa0]

    2. "Item 1. Business" has 2 matches — match 2 is the real section
       starting with "Our Company\nNVIDIA pioneered..."

    3. "Item 1A. Risk Factors" has 8 matches — 7 are cross-references
       like "see Item 1A. Risk Factors for discussion of..."
       Real section is the one followed by "The following risk factors
       should be considered..." — we include that in the start pattern
       to match exactly 1 occurrence.

    4. "Item 7. Management" has 2 matches — match 2 is the real section.
       The "pick densest" logic handles this automatically.

USED BY:
    src/agents/filing_agent.py -> calls load_document()

HOW TO RUN:
    cd financial-research-agent
    python -m src.rag.document_loader
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    Structured container for a cleaned SEC 10-K filing.
    Passed through the entire RAG pipeline:
        document_loader → chunker → embedder → vector_store → retriever
    """
    ticker:       str
    company_name: str
    filing_date:  str
    source:       str
    text:         str
    accession_no: Optional[str] = None
    filing_url:   Optional[str] = None
    text_length:  int = 0
    sections:     dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION — load_document
# ─────────────────────────────────────────────────────────────────────────────

def load_document(filing: dict) -> Document:
    """
    Clean and structure a raw SEC 10-K filing into a Document object.

    Args:
        filing: dict returned by src.tools.sec_edgar.fetch_latest_10k()

    Returns:
        Document dataclass with cleaned text and extracted sections
    """
    ticker       = filing.get("ticker", "UNKNOWN")
    company_name = filing.get("company_name", ticker)
    filing_date  = filing.get("filing_date", "Unknown")
    raw_text     = filing.get("text", "")

    logger.info(
        f"Loading document for {ticker} — "
        f"raw text: {len(raw_text):,} characters"
    )

    cleaned_text = _clean_text(raw_text)
    sections     = _extract_sections(cleaned_text)

    try:
        fiscal_year = f"FY{filing_date[:4]}"
    except Exception:
        fiscal_year = "FY Unknown"

    source = f"SEC 10-K {fiscal_year} — {company_name}"

    doc = Document(
        ticker       = ticker,
        company_name = company_name,
        filing_date  = filing_date,
        source       = source,
        text         = cleaned_text,
        accession_no = filing.get("accession_no"),
        filing_url   = filing.get("filing_url"),
        text_length  = len(cleaned_text),
        sections     = sections,
    )

    logger.info(
        f"Document loaded for {ticker} — "
        f"cleaned: {len(cleaned_text):,} chars, "
        f"sections found: {[k for k, v in sections.items() if len(v) > 500]}"
    )

    return doc


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _clean_text
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Clean raw SEC filing text by removing noise and normalizing whitespace.

    WHAT WE REMOVE:
        - HTML entities (&nbsp; &amp; etc.)
        - Non-breaking spaces (\xa0) → converted to regular spaces
        - Page numbers (standalone digits on their own line)
        - Table separator lines (rows of dashes/equals)
        - Dot leaders from table of contents (Revenue ........ 45)
        - Repeated short lines that appear 5+ times (headers/footers)
        - Multiple consecutive blank lines
    """
    # ── Convert non-breaking spaces FIRST ────────────────────────────────────
    # sec_edgar.py may or may not do this — we do it here to be safe.
    # \xa0 appears in NVIDIA's 10-K between "Item" and the number.
    # If not converted, section patterns won't match correctly.
    text = text.replace('\xa0', ' ')

    # ── Remove HTML entities ──────────────────────────────────────────────────
    text = re.sub(r"&nbsp;",       " ",  text)
    text = re.sub(r"&amp;",        "&",  text)
    text = re.sub(r"&lt;",         "<",  text)
    text = re.sub(r"&gt;",         ">",  text)
    text = re.sub(r"&#\d+;",       " ",  text)
    text = re.sub(r"&[a-zA-Z]+;",  " ",  text)
    text = re.sub(r"<[^>]+>",      " ",  text)

    # ── Remove page numbers ───────────────────────────────────────────────────
    # Standalone digits on their own line e.g. "42" or "128"
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # ── Remove table separator lines ──────────────────────────────────────────
    text = re.sub(r"^\s*[-=_]{4,}\s*$", "", text, flags=re.MULTILINE)

    # ── Remove dot leaders from TOC ───────────────────────────────────────────
    # "Revenue ................ 45" → "Revenue  45"
    text = re.sub(r"\.{4,}", " ", text)

    # ── Normalize whitespace ──────────────────────────────────────────────────
    text = text.replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Remove repeated header/footer lines ───────────────────────────────────
    # Lines under 60 chars that appear 5+ times are headers/footers
    lines = [line.strip() for line in text.split("\n")]

    from collections import Counter
    short_lines = [l for l in lines if 0 < len(l) < 60]
    line_counts = Counter(short_lines)
    repeated    = {line for line, count in line_counts.items() if count >= 5}

    if repeated:
        logger.info(f"Removing {len(repeated)} repeated header/footer lines")
        lines = [l for l in lines if l not in repeated]

    text = "\n".join(lines)
    text = text.strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _extract_sections
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sections(text: str) -> dict:
    """
    Extract 3 key sections from the cleaned 10-K text.

    APPROACH:
        Use re.finditer() to find ALL occurrences of each section heading.
        Pick the match with the most content after it — that's the real
        section, not a TOC entry or cross-reference.

        TOC entries have ~50-200 chars before the next item.
        Real sections have thousands of chars.

    PATTERNS — HOW THEY WERE DERIVED:
        Tested against actual NVIDIA 10-K HTML file.

        business:
            Start: "Item 1. Business" — 2 matches, match 2 is real
            End:   "Item 1A. Risk Factors\nThe following..." — 1 match only
                   Including "The following" makes it unique (not a cross-ref)

        risk_factors:
            Start: "Item 1A. Risk Factors\nThe following..." — 1 match only
                   The phrase "The following risk factors" only appears at
                   the real section heading, not in cross-references
            End:   "Item 1B" or "Item 2." — marks end of risk section

        management_discussion:
            Start: "Item 7. Management" — 2 matches, match 2 is real
            End:   "Item 7A" or "Item 8" — marks end of MD&A section
    """

    sections = {
        "business":              "",
        "risk_factors":          "",
        "management_discussion": "",
    }

    section_patterns = [
        (
            "business",
            # Start: "Item 1. Business\nOur Company\nNVIDIA pioneered..."
            # WHY THIS SPECIFIC PATTERN:
            #   The real business section heading in cleaned text is:
            #   "Item 1. Business\nOur Company"
            #   The TOC entry is:
            #   "Item 1.\nBusiness\n\nItem 1A.\nRisk Factors..."
            #   By including "Our Company" in the pattern, we match
            #   ONLY the real section — not the TOC entry.
            #   This gives us exactly 1 match, no need for "pick densest".
            r"item\s+1\.\s*business\s*\nour company",

            # End: the real Item 1A heading follows immediately after business
            # In cleaned text this appears as "Item 1A. Risk Factors\nThe following"
            r"item\s+1a",
        ),
        (
            "risk_factors",
            # Start: "Item 1A. Risk Factors\nThe following risk factors..."
            # Including "The following" is critical — there are 8 matches
            # of "Item 1A. Risk Factors" in the document (cross-references)
            # but only 1 match followed by "The following" — the real section
            r"item\s+1a\.?\s*risk\s*factors\n+the following",

            # End: "Item 1B" or "Item 2." — next section after risk factors
            r"item\s+1b|item\s+2\.",
        ),
        (
            "management_discussion",
            # Start: "Item 7. Management" — 2 matches
            # TOC entry has ~200 chars, real section has 33,000+ chars
            # _find_real_section picks correctly
            r"item\s+7\.?\s*\n*management",

            # End: "Item 7A" (Quantitative Disclosures) or "Item 8"
            r"item\s+7a|item\s+8",
        ),
    ]

    for section_key, start_pattern, end_pattern in section_patterns:
        section_text = _find_real_section(
            text          = text,
            section_key   = section_key,
            start_pattern = start_pattern,
            end_pattern   = end_pattern,
        )
        sections[section_key] = section_text

        if section_text:
            logger.info(
                f"Extracted section '{section_key}': "
                f"{len(section_text):,} characters"
            )
        else:
            logger.warning(f"Section '{section_key}' not found or too short")

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _find_real_section
# Finds ALL matches of a pattern and returns the one with the most
# content after it — skipping TOC entries and cross-references.
# ─────────────────────────────────────────────────────────────────────────────

def _find_real_section(
    text:          str,
    section_key:   str,
    start_pattern: str,
    end_pattern:   str,
    min_length:    int = 500,
    max_length:    int = 50000,
) -> str:
    """
    Find the real section by picking the match with the most content.

    HOW IT WORKS:
        1. Find ALL positions where start_pattern appears
        2. For each position, find the next end_pattern occurrence
        3. Measure content between start and end
        4. Return the match with the most content
        5. If no match has > min_length chars → section not found

    Args:
        text:          Full cleaned document text
        section_key:   Section name for logging
        start_pattern: Regex matching the section heading
        end_pattern:   Regex matching the next section heading
        min_length:    Minimum chars to be considered a real section
        max_length:    Cap section at this many chars

    Returns:
        Section text, or empty string if not found
    """
    all_matches = list(re.finditer(start_pattern, text, re.IGNORECASE))

    if not all_matches:
        logger.info(f"Section '{section_key}': pattern not found in document")
        return ""

    logger.info(
        f"Section '{section_key}': found {len(all_matches)} match(es) — "
        f"selecting the one with most content (skipping TOC)"
    )

    best_text   = ""
    best_length = 0

    for match in all_matches:
        start_pos = match.start()

        # Search for end pattern starting 50 chars after start
        # The +50 offset skips past the heading line itself
        search_from = start_pos + 50
        end_match   = re.search(end_pattern, text[search_from:], re.IGNORECASE)

        if end_match:
            end_pos      = search_from + end_match.start()
            section_text = text[start_pos:end_pos].strip()
        else:
            # No end found — take up to max_length from start
            section_text = text[start_pos:start_pos + max_length].strip()

        section_text = section_text[:max_length]

        if len(section_text) > best_length:
            best_length = len(section_text)
            best_text   = section_text

    if best_length < min_length:
        logger.warning(
            f"Section '{section_key}': best match only {best_length} chars "
            f"(minimum {min_length}) — treating as not found"
        )
        return ""

    logger.info(
        f"Section '{section_key}': selected match with {best_length:,} chars"
    )

    return best_text


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Document Loader — Sanity Check (Fixed Version)")
    print(f"{'='*60}")

    print(f"\n── Fetching NVDA 10-K ───────────────────────────────────────\n")

    from src.tools.sec_edgar import fetch_latest_10k
    filing = fetch_latest_10k("NVDA")
    print(f"  Raw text length  : {filing['text_length']:,} characters")

    doc = load_document(filing)

    print(f"\n── Sections extracted ───────────────────────────────────────\n")
    for section_name, section_text in doc.sections.items():
        status = f"{len(section_text):,} chars" if section_text else "NOT FOUND ❌"
        print(f"  {section_name:<25} : {status}")

    print(f"\n── Section previews (first 300 chars each) ──────────────────\n")
    for section_name, section_text in doc.sections.items():
        if section_text:
            print(f"  [{section_name.upper()}]")
            # Skip the heading lines — show actual content
            lines         = section_text.split("\n")
            content_lines = [l for l in lines if len(l.strip()) > 60]
            preview       = content_lines[0] if content_lines else section_text[:300]
            print(f"  {preview[:300]}")
            print()