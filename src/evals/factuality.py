"""
src/evals/factuality.py
=======================
RAGAS evaluation for the RAG pipeline — measures Faithfulness and Context Precision.

RAGAS VERSION: 0.4.3

WHAT THIS FILE DOES:
    Takes the eval_data collected by filing_agent (question, answer, contexts)
    and runs RAGAS evaluation to measure:

    1. FAITHFULNESS (0.0 to 1.0):
       Are the claims in the answer supported by the retrieved chunks?
       High score → LLM is grounded in context, not hallucinating
       Low score  → LLM is making up information

    2. CONTEXT PRECISION WITHOUT REFERENCE (0.0 to 1.0):
       Are the retrieved chunks relevant to the question?
       High score → retriever is finding the right chunks
       Low score  → retriever is returning irrelevant chunks
       NOTE: Uses LLMContextPrecisionWithoutReference — no ground truth needed

HOW TO USE:
    from src.evals.factuality import evaluate
    scores = evaluate(state.eval_data, ticker="NVDA")

HOW TO RUN STANDALONE:
    cd financial-research-agent
    python -m src.evals.factuality
"""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference
from ragas.llms import llm_factory
from openai import OpenAI
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NIM_API_KEY  = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL    = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(eval_data: list, ticker: str = "NVDA") -> dict:
    """
    Run RAGAS evaluation on filing agent output.

    Args:
        eval_data: List of dicts from filing_agent, each containing:
                   - question: the query asked
                   - answer:   what the LLM extracted
                   - contexts: list of retrieved chunk texts
        ticker:    Company ticker for logging

    Returns:
        dict with faithfulness, context_precision, num_queries, details
    """
    if not eval_data:
        logger.warning("No eval_data provided — skipping evaluation")
        return {
            "faithfulness":      None,
            "context_precision": None,
            "num_queries":       0,
            "details":           [],
        }

    if not NIM_API_KEY:
        logger.error("NVIDIA_NIM_API_KEY not set — cannot run RAGAS evaluation")
        return {
            "faithfulness":      None,
            "context_precision": None,
            "num_queries":       0,
            "details":           [],
            "error":             "NVIDIA_NIM_API_KEY not configured",
        }

    logger.info(
        f"Running RAGAS evaluation for {ticker} — "
        f"{len(eval_data)} queries"
    )

    # Step 1: Prepare dataset
    dataset = _prepare_dataset(eval_data)

    # Step 2: Configure NIM as judge
    nim_llm = _setup_nim()

    # Step 3: Run RAGAS
    scores = _run_ragas(dataset, nim_llm, ticker)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _prepare_dataset
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_dataset(eval_data: list) -> Dataset:
    """
    Convert eval_data list into a RAGAS-compatible HuggingFace Dataset.

    RAGAS 0.4.3 expects columns: question, answer, contexts
    contexts must be a list of strings per row.
    """
    valid_data = []

    for item in eval_data:
        question = item.get("question", "").strip()
        answer   = item.get("answer",   "").strip()
        contexts = [c for c in item.get("contexts", []) if c.strip()]

        if not question or not answer or not contexts:
            logger.warning(f"Skipping eval item with missing data")
            continue

        valid_data.append({
            "question": question,
            "answer":   answer,
            "contexts": contexts,
        })

    if not valid_data:
        logger.warning("No valid eval data after filtering")
        return Dataset.from_dict({
            "question": [],
            "answer":   [],
            "contexts": [],
        })

    dataset_dict = {
        "question": [item["question"] for item in valid_data],
        "answer":   [item["answer"]   for item in valid_data],
        "contexts": [item["contexts"] for item in valid_data],
    }

    logger.info(f"Prepared RAGAS dataset — {len(valid_data)} valid entries")
    return Dataset.from_dict(dataset_dict)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _setup_nim
# Uses RAGAS 0.4.3 new API: llm_factory with OpenAI client
#
# WHY llm_factory:
#   In RAGAS 0.4.x, LangchainLLMWrapper is deprecated.
#   The new API uses llm_factory() with an OpenAI-compatible client.
#   NIM is OpenAI-compatible so we point the client to NIM's URL.
# ─────────────────────────────────────────────────────────────────────────────

def _setup_nim():
    """
    Configure NIM as the LLM judge using RAGAS 0.4.3 API.

    Returns:
        RAGAS LLM object configured to use NIM
    """
    # Create OpenAI client pointing to NIM
    nim_client = OpenAI(
        api_key  = NIM_API_KEY,
        base_url = NIM_BASE_URL,
    )

    # Use RAGAS 0.4.3 llm_factory with our NIM client
    nim_llm = llm_factory(
        model  = NIM_MODEL,
        client = nim_client,
    )

    logger.info(f"RAGAS configured with NIM — model={NIM_MODEL}")
    return nim_llm


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — _run_ragas
# ─────────────────────────────────────────────────────────────────────────────

def _run_ragas(dataset: Dataset, nim_llm, ticker: str) -> dict:
    """
    Run RAGAS evaluation and return formatted scores.

    METRICS USED:
        faithfulness                      → no ground truth needed
        LLMContextPrecisionWithoutReference → no ground truth needed

    Both metrics use NIM as the judge LLM.
    """
    if len(dataset) == 0:
        logger.warning("Empty dataset — cannot run RAGAS")
        return {
            "faithfulness":      None,
            "context_precision": None,
            "num_queries":       0,
            "details":           [],
        }

    try:
        logger.info("Running RAGAS evaluation (30-60 seconds)...")

        # Instantiate metrics as objects (required in RAGAS 0.4.x)
        faith_metric = Faithfulness(llm=nim_llm)
        cp_metric    = LLMContextPrecisionWithoutReference(llm=nim_llm)

        # Run evaluation
        result = ragas_evaluate(
            dataset = dataset,
            metrics = [faith_metric, cp_metric],
        )

        result_df = result.to_pandas()

        # Log available columns for debugging
        logger.info(f"RAGAS result columns: {list(result_df.columns)}")

        # Find faithfulness column
        faith_col = [c for c in result_df.columns if "faith" in c.lower()]
        cp_col    = [c for c in result_df.columns if "precision" in c.lower()]

        faith_score = float(result_df[faith_col[0]].mean()) if faith_col else 0.0
        cp_score    = float(result_df[cp_col[0]].mean())    if cp_col    else 0.0

        # Per-query breakdown
        details = []
        for i in range(len(result_df)):
            details.append({
                "question":         dataset["question"][i][:100],
                "faithfulness":     round(float(result_df[faith_col[0]].iloc[i]), 3) if faith_col else 0.0,
                "context_precision": round(float(result_df[cp_col[0]].iloc[i]), 3)   if cp_col    else 0.0,
            })

        logger.info(
            f"RAGAS complete for {ticker} — "
            f"faithfulness={faith_score:.3f}, "
            f"context_precision={cp_score:.3f}"
        )

        return {
            "faithfulness":      round(faith_score, 3),
            "context_precision": round(cp_score, 3),
            "num_queries":       len(dataset),
            "details":           details,
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {
            "faithfulness":      None,
            "context_precision": None,
            "num_queries":       len(dataset),
            "details":           [],
            "error":             str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  Factuality Evaluation — Sanity Check (RAGAS 0.4.3)")
    print(f"{'='*60}\n")

    mock_eval_data = [
        {
            "question": "What are NVIDIA's main business segments?",
            "answer":   "NVIDIA operates two segments: Compute & Networking and Graphics. Compute & Networking grew strongly driven by H100 GPU demand.",
            "contexts": [
                "NVIDIA's reportable segments are Compute & Networking and Graphics.",
                "The Compute & Networking segment saw strong growth driven by H100 and H200 GPU demand from hyperscalers.",
                "Graphics segment revenue decreased due to inventory normalization after the crypto mining boom.",
            ]
        },
        {
            "question": "What are the top risk factors facing NVIDIA?",
            "answer":   "Top risks include export controls restricting chip sales to China, competition from AMD and custom silicon, and supply chain disruptions.",
            "contexts": [
                "Export control regulations restrict NVIDIA's ability to sell A100 and H100 chips to China, impacting approximately $5B in potential revenue.",
                "Competition from AMD MI300X and custom silicon from Google TPUs represents long-term substitution risk.",
                "Long manufacturing lead times and supply chain disruptions could negatively impact financial results.",
            ]
        },
        {
            "question": "What is NVIDIA's management outlook?",
            "answer":   "Revenue growth driven by data center AI solutions. Energy infrastructure availability is crucial for sustained demand.",
            "contexts": [
                "Revenue growth was driven by data center compute and networking platforms for accelerated computing and AI solutions.",
                "Expanding energy capacity to meet demand is a complex multi-year process with regulatory and construction challenges.",
                "Blackwell architectures represented the majority of Data Center revenue in fiscal year 2026.",
            ]
        },
    ]

    print(f"Queries: {len(mock_eval_data)}")
    print("Running evaluation...\n")

    scores = evaluate(mock_eval_data, ticker="NVDA")

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}\n")
    print(f"  Faithfulness      : {scores.get('faithfulness')}")
    print(f"  Context Precision : {scores.get('context_precision')}")
    print(f"  Queries evaluated : {scores.get('num_queries')}")

    if scores.get("error"):
        print(f"  Error: {scores.get('error')}")

    print(f"\n  Per-query breakdown:")
    for d in scores.get("details", []):
        print(f"\n    Q : {d['question'][:80]}")
        print(f"    Faithfulness      : {d['faithfulness']}")
        print(f"    Context Precision : {d['context_precision']}")
    print()