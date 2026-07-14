# Loads and validates all environment variables
"""
src/utils/config.py
===================
Central configuration loader for the entire project.

Reads all environment variables from the .env file and stores
them in one Config object. Every other file imports from here.

RULES:
    - No other file should call os.getenv() directly
    - All API keys live here and nowhere else
    - Missing required keys fail immediately at startup
      (not 10 minutes later inside the pipeline)

USAGE:
    from src.utils.config import config

    key = config.openai_api_key
    dir = config.chroma_persist_dir

HOW TO SET UP:
    1. Copy .env.example to .env
    2. Fill in your actual API keys in .env
    3. Never commit .env to GitHub
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load the .env file into the environment
# Must be called before any os.getenv() call
# If .env file doesn't exist, it silently continues
# (useful in production where env vars are set directly)
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DATACLASS
# A simple container that holds all configuration values.
# Using @dataclass instead of Pydantic because config values
# are just strings — no complex validation needed here.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Used by all agents to call GPT-4o
    # openai_api_key: str

    # ── News Search ───────────────────────────────────────────────────────────
    # Used by news_search.py tool to fetch financial news
    tavily_api_key: str

    # ── Email Delivery ────────────────────────────────────────────────────────
    # Used by email_sender.py to send the final report
    # sendgrid_api_key: str
    # sendgrid_from_email: str
    # report_recipient_email: str

    # ── Observability ─────────────────────────────────────────────────────────
    # Used by LangSmith to trace every agent run
    # Optional — pipeline works without it, just no tracing
    langchain_api_key: str

    # ── Vector Store ──────────────────────────────────────────────────────────
    # Where ChromaDB stores embeddings on disk
    chroma_persist_dir: str

    # ── App Settings ──────────────────────────────────────────────────────────
    log_level: str   # "DEBUG" | "INFO" | "WARNING" | "ERROR"


# ─────────────────────────────────────────────────────────────────────────────
# LOADER FUNCTION
# Reads all env vars, validates required ones, returns Config object.
# Called once at the bottom of this file — result stored in `config`.
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> Config:
    """
    Read environment variables and return a Config object.

    Required keys → raises EnvironmentError immediately if missing.
    Optional keys → falls back to a sensible default if missing.
    """

    # ── Step 1: Read all values from environment ──────────────────────────────
    # openai_api_key         = os.getenv("OPENAI_API_KEY",          "")
    tavily_api_key         = os.getenv("TAVILY_API_KEY",          "")
    # sendgrid_api_key       = os.getenv("SENDGRID_API_KEY",        "")
    # sendgrid_from_email    = os.getenv("SENDGRID_FROM_EMAIL",     "")
    # report_recipient_email = os.getenv("REPORT_RECIPIENT_EMAIL",  "")
    langchain_api_key      = os.getenv("LANGCHAIN_API_KEY",       "")
    chroma_persist_dir     = os.getenv("CHROMA_PERSIST_DIR",      "./chroma_db")
    log_level              = os.getenv("LOG_LEVEL",               "INFO")

    # ── Step 2: Validate required keys ────────────────────────────────────────
    # These keys are absolutely required — pipeline cannot run without them.
    # We check all of them at once and report everything missing in one error
    # instead of failing one key at a time.

    required = {
        # "OPENAI_API_KEY":  openai_api_key,
        "TAVILY_API_KEY":  tavily_api_key,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise EnvironmentError(
            f"\n\n"
            f"  Missing required environment variables:\n"
            f"  {', '.join(missing)}\n\n"
            f"  Steps to fix:\n"
            f"  1. Copy .env.example to .env\n"
            f"  2. Open .env and add your API keys\n"
            f"  3. Run the script again\n"
        )

    # ── Step 3: Warn about optional but recommended keys ─────────────────────
    # These are not required to run the pipeline but reduce functionality.

    # if not sendgrid_api_key:
    #     print("  [Config] WARNING: SENDGRID_API_KEY not set — email delivery disabled")

    if not langchain_api_key:
        print("  [Config] WARNING: LANGCHAIN_API_KEY not set — LangSmith tracing disabled")

    # ── Step 4: Return the Config object ─────────────────────────────────────
    return Config(
        # openai_api_key         = openai_api_key,
        tavily_api_key         = tavily_api_key,
        # sendgrid_api_key       = sendgrid_api_key,
        # sendgrid_from_email    = sendgrid_from_email,
        # report_recipient_email = report_recipient_email,
        langchain_api_key      = langchain_api_key,
        chroma_persist_dir     = chroma_persist_dir,
        log_level              = log_level,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG INSTANCE
# Loaded once when this module is first imported.
# Every other file imports this single instance.
#
# Why load at import time?
# So missing keys are caught immediately when the app starts —
# not later when a specific tool first tries to use its key.
# ─────────────────────────────────────────────────────────────────────────────

config = _load_config()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to see all loaded config values.
# Masks sensitive keys so they are safe to print.
#
# Usage:
#   python src/utils/config.py
# ─────────────────────────────────────────────────────────────────────────────

def _mask(value: str) -> str:
    """Show only first 6 chars of a sensitive key, rest as asterisks."""
    if not value:
        return "NOT SET"
    if len(value) <= 6:
        return "***"
    return value[:6] + "*" * (len(value) - 6)


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("  Config Loaded Successfully")
    print("=" * 55)

    print("\n── LLM ─────────────────────────────────────────────")
    print(f"  openai_api_key         : {_mask(config.openai_api_key)}")

    print("\n── News Search ─────────────────────────────────────")
    print(f"  tavily_api_key         : {_mask(config.tavily_api_key)}")

    print("\n── Email Delivery ──────────────────────────────────")
    print(f"  sendgrid_api_key       : {_mask(config.sendgrid_api_key)}")
    print(f"  sendgrid_from_email    : {config.sendgrid_from_email or 'NOT SET'}")
    print(f"  report_recipient_email : {config.report_recipient_email or 'NOT SET'}")

    print("\n── Observability ───────────────────────────────────")
    print(f"  langchain_api_key      : {_mask(config.langchain_api_key)}")

    print("\n── App Settings ────────────────────────────────────")
    print(f"  chroma_persist_dir     : {config.chroma_persist_dir}")
    print(f"  log_level              : {config.log_level}")

    print("\n── Validation Test ─────────────────────────────────")
    print(f"  OpenAI key set?        : {'✅ Yes' if config.openai_api_key else '❌ No'}")
    print(f"  Tavily key set?        : {'✅ Yes' if config.tavily_api_key else '❌ No'}")
    print(f"  SendGrid key set?      : {'✅ Yes' if config.sendgrid_api_key else '⚠️  No (email disabled)'}")
    print(f"  LangSmith key set?     : {'✅ Yes' if config.langchain_api_key else '⚠️  No (tracing disabled)'}")
    print()