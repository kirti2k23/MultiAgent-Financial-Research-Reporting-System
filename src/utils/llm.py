"""
src/utils/llm.py
================
Unified LLM caller — single entry point for all LLM calls across the system.

WHY THIS FILE EXISTS:
    Before this file, every agent had its own copy of the Ollama API call.
    The same HTTP request code was duplicated in 4 places:
        research_agent.py, filing_agent.py, report_agent.py, retriever.py

    Problems with duplication:
        - Switch to NIM → change 4 files
        - Add <think> tag stripping → change 4 files
        - Change timeout → change 4 files

    This file centralizes ALL LLM calls. Agents call one function:
        from src.utils.llm import call_llm
        response = call_llm(prompt="...", temperature=0.2)

    Now switching providers, models, or behavior = change ONE file.

SUPPORTS TWO PROVIDERS:
    ollama → local models (llama3, deepseek-r1:8b) via http://localhost:11434
    nim    → NVIDIA NIM cloud (llama-3.1-8b, llama-3.1-70b) via cloud GPU

    Switch between them with ONE line in .env:
        LLM_PROVIDER=ollama   (local, free, runs on your Mac GPU)
        LLM_PROVIDER=nim      (cloud, free tier, runs on NVIDIA H100 GPU)

HOW TO RUN:
    cd financial-research-agent
    python -m src.utils.llm
"""

import re
import os
import requests
from dotenv import load_dotenv
from src.utils.logger import get_logger

# Load .env so os.getenv() can read LLM_PROVIDER, NIM key, etc.
# Without this, env vars from .env are not available at import time.
load_dotenv()

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# Read which provider and model to use from environment variables.
#
# WHY READ FROM ENV:
#   Switching from local to cloud should not require code changes.
#   Set LLM_PROVIDER=nim in .env and the whole system uses NIM.
#   Set LLM_PROVIDER=ollama and it uses local models.
# ─────────────────────────────────────────────────────────────────────────────

# Which provider: "ollama" (local) or "nim" (NVIDIA cloud)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Ollama settings (local) ───────────────────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

# ── NVIDIA NIM settings (cloud) ───────────────────────────────────────────────
# NIM uses OpenAI-compatible API format
NIM_CHAT_URL    = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_API_KEY     = os.getenv("NVIDIA_NIM_API_KEY", "")
NIM_MODEL       = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — call_llm
# The ONLY function agents should call. Everything else is internal.
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    prompt:      str,
    temperature: float = 0.2,
    max_tokens:  int   = 500,
) -> str:
    """
    Call the configured LLM provider and return clean text.

    This is the single entry point for all LLM calls in the system.
    Agents don't need to know which provider is running or how the
    HTTP call works — they just call this function.

    WHAT IT HANDLES:
        - Routing to Ollama or NIM based on LLM_PROVIDER
        - Building the correct request format for each provider
        - Parsing the response
        - Stripping <think>...</think> reasoning blocks
        - Error handling

    Args:
        prompt:      The text prompt to send to the LLM
        temperature: 0.0 = deterministic, 1.0 = creative (default 0.2)
        max_tokens:  Maximum tokens in the response (default 500)

    Returns:
        Clean text response from the LLM (thinking blocks stripped)
        Returns empty string "" on failure.
    """
    # Log which provider and model is being used
    # This makes it clear in the logs whether Ollama or NIM is running
    if LLM_PROVIDER == "nim":
        logger.info(f"LLM call → NIM ({NIM_MODEL}) | temp={temperature} | max_tokens={max_tokens}")
    else:
        logger.info(f"LLM call → Ollama ({OLLAMA_MODEL}) | temp={temperature} | max_tokens={max_tokens}")

    # Route to the correct provider
    if LLM_PROVIDER == "nim":
        raw = _call_nim(prompt, temperature, max_tokens)
    else:
        raw = _call_ollama(prompt, temperature, max_tokens)

    # Strip <think>...</think> blocks that reasoning models add
    clean = _strip_thinking(raw)

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE — _call_ollama
# Makes the actual HTTP call to local Ollama.
# This is the same code that was previously duplicated in every agent.
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Call local Ollama and return the raw response text.

    OLLAMA API FORMAT:
        POST http://localhost:11434/api/chat
        Body: {
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "..."}],
            "stream": false,
            "options": {"temperature": 0.2, "num_predict": 500}
        }
        Response: {"message": {"content": "..."}}

    Args:
        prompt:      The text prompt
        temperature: Sampling temperature
        max_tokens:  Max tokens (Ollama calls this num_predict)

    Returns:
        Raw response text, or empty string on failure
    """
    # Reasoning models (deepseek-r1) use tokens for internal thinking
    # BEFORE producing the actual answer. If max_tokens is too low,
    # all tokens are spent reasoning and the answer comes back empty.
    # We boost the token budget for these models to leave room for
    # both the reasoning AND the answer.
    effective_max = max_tokens
    if "deepseek-r1" in OLLAMA_MODEL or "r1" in OLLAMA_MODEL:
        effective_max = max_tokens + 1000  # extra room for reasoning

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json = {
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options": {
                    "temperature": temperature,
                    "num_predict": effective_max,
                }
            },
            timeout = 180,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()

    except requests.exceptions.ConnectionError:
        logger.error(
            "Ollama not running. Start it with: ollama serve"
        )
        return ""

    except requests.exceptions.Timeout:
        logger.error(f"Ollama timed out after 180s (model={OLLAMA_MODEL})")
        return ""

    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE — _call_nim
# Makes the actual HTTP call to NVIDIA NIM cloud.
#
# NIM uses OpenAI-compatible format — different from Ollama:
#   - Different URL
#   - Bearer token auth header
#   - Response in choices[0].message.content (like OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

def _call_nim(prompt: str, temperature: float, max_tokens: int) -> str:
    """
    Call NVIDIA NIM cloud API and return the raw response text.

    NIM API FORMAT (OpenAI-compatible):
        POST https://integrate.api.nvidia.com/v1/chat/completions
        Header: Authorization: Bearer nvapi-xxxxx
        Body: {
            "model": "meta/llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": "..."}],
            "temperature": 0.2,
            "max_tokens": 500
        }
        Response: {"choices": [{"message": {"content": "..."}}]}

    WHY DIFFERENT FROM OLLAMA:
        NIM follows OpenAI's API standard. The response is in
        choices[0].message.content, not message.content.
        Auth uses a Bearer token in the header.

    Args:
        prompt:      The text prompt
        temperature: Sampling temperature
        max_tokens:  Max tokens

    Returns:
        Raw response text, or empty string on failure
    """
    if not NIM_API_KEY:
        logger.error(
            "NVIDIA_NIM_API_KEY not set in .env — cannot use NIM provider"
        )
        return ""

    try:
        response = requests.post(
            NIM_CHAT_URL,
            headers = {
                "Authorization": f"Bearer {NIM_API_KEY}",
                "Content-Type":  "application/json",
            },
            json = {
                "model":       NIM_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens":  max_tokens,
            },
            timeout = 120,
        )
        response.raise_for_status()

        # NIM response format follows OpenAI: choices[0].message.content
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        logger.error(f"NIM timed out after 120s (model={NIM_MODEL})")
        return ""

    except requests.exceptions.HTTPError as e:
        logger.error(f"NIM HTTP error: {e} — check API key and model name")
        return ""

    except Exception as e:
        logger.error(f"NIM call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE — _strip_thinking
# Removes <think>...</think> reasoning blocks from model output.
#
# WHY THIS IS NEEDED:
#   DeepSeek-R1 and other reasoning models output their thinking process
#   wrapped in <think>...</think> tags before the actual answer:
#
#       <think>
#       Let me analyze the financial data...
#       </think>
#       The revenue breakdown shows...
#
#   We only want the actual answer, not the thinking. This function
#   removes the thinking block. If no thinking block exists (smaller
#   models, simple tasks), it returns the text unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    HOW IT WORKS:
        re.DOTALL → makes . match newlines, so it removes the entire
                    multi-line thinking block, not just one line.
        We remove everything between <think> and </think> inclusive.

    Args:
        text: Raw model output that may contain thinking blocks

    Returns:
        Text with thinking blocks removed and whitespace cleaned
    """
    if not text:
        return ""

    # Remove <think>...</think> blocks (including the tags)
    # re.DOTALL makes . match newlines so multi-line blocks are removed
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Some models leave a stray closing tag if generation was cut off
    cleaned = cleaned.replace("</think>", "").replace("<think>", "")

    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"\n{'='*60}")
    print(f"  LLM Caller — Sanity Check")
    print(f"{'='*60}")

    print(f"\n  Provider : {LLM_PROVIDER}")
    print(f"  Ollama model : {OLLAMA_MODEL}")
    print(f"  NIM model    : {NIM_MODEL}")
    print(f"  NIM key set  : {'Yes' if NIM_API_KEY else 'No'}")

    # ── Test 1: Basic call ────────────────────────────────────────────────────
    print(f"\n── Test 1: Basic LLM call ────────────────────────────────────\n")

    response = call_llm(
        prompt      = "Reply with exactly one sentence: what is NVIDIA known for?",
        temperature = 0.2,
        max_tokens  = 100,
    )
    print(f"  Response: {response}")

    # ── Test 2: Thinking block stripping ──────────────────────────────────────
    print(f"\n── Test 2: <think> tag stripping ────────────────────────────\n")

    mock_with_thinking = """<think>
The user wants me to analyze this. Let me think...
This is reasoning that should be removed.
</think>

The actual answer is NVIDIA makes GPUs."""

    stripped = _strip_thinking(mock_with_thinking)
    print(f"  Input had <think> block : Yes")
    print(f"  Output: {stripped}")
    print(f"  Thinking removed: {'Yes' if '<think>' not in stripped else 'No'}")

    # ── Test 3: JSON task ─────────────────────────────────────────────────────
    print(f"\n── Test 3: JSON response ────────────────────────────────────\n")

    json_response = call_llm(
        prompt      = 'Respond ONLY in JSON: {"sentiment": "positive or negative", "score": 0.0 to 1.0} for this news: NVIDIA stock hits record high.',
        temperature = 0.0,
        max_tokens  = 100,
    )
    print(f"  Response: {json_response}")
    print()