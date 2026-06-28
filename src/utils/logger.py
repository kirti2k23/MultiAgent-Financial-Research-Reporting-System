# Structured logging setup used across all modules
"""
src/utils/logger.py
===================
Centralized logging setup for the entire project.

Every file in the project gets its own named logger
by calling get_logger(__name__) at the top of the file.

WHY CENTRALIZED:
    If logging was set up separately in each file,
    you'd get inconsistent formats, duplicate handlers,
    and no central control over log level.
    One function here, consistent logs everywhere.

USAGE IN ANY FILE:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Fetching news for NVDA")
    logger.warning("API returned empty results")
    logger.error("Request failed after 3 retries")

OUTPUT FORMAT:
    2026-04-29 09:42:11 [INFO]    research_agent    : Fetching news for NVDA
    2026-04-29 09:42:15 [WARNING] news_search       : Tavily returned 0 results
    2026-04-29 09:42:20 [ERROR]   sec_edgar         : Request failed after 3 retries
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger with consistent formatting.

    Args:
        name: Name for this logger. Always pass __name__ so the
              logger is automatically named after the module.
              e.g. get_logger(__name__) in research_agent.py
              produces a logger named 'src.agents.research_agent'

    Returns:
        A configured logging.Logger instance ready to use.

    WHY CHECK handlers BEFORE ADDING:
        Python's logging module is global. If you call get_logger()
        multiple times for the same module (e.g. during testing),
        it returns the same logger object. Without the check,
        you'd add a new handler each time — causing every log
        message to print 2x, 3x, 4x times. The check prevents this.
    """

    # Get or create a logger with this name
    # If a logger with this name already exists, Python returns
    # the same one — loggers are singletons by name
    logger = logging.getLogger(name)

    # Only add handler if this logger has none yet
    # This prevents duplicate log messages on repeated calls
    if not logger.handlers:

        # ── Set log level ─────────────────────────────────────────────────────
        # Read from environment variable LOG_LEVEL (set in .env)
        # Default to INFO if not set
        # getattr converts the string "INFO" to logging.INFO (which is 20)
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level     = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)

        # ── Create handler ────────────────────────────────────────────────────
        # StreamHandler sends logs to the terminal (stdout)
        # In production you can add FileHandler or CloudWatch handler here
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # ── Create formatter ──────────────────────────────────────────────────
        # This defines exactly how each log line looks
        #
        # %(asctime)s   → timestamp:   2026-04-29 09:42:11
        # %(levelname)s → level:       INFO / WARNING / ERROR
        # %(name)s      → module name: src.agents.research_agent
        # %(message)s   → the actual message you logged
        #
        # The padding (−8s, −30s) right-pads the text so all
        # columns align neatly across different log lines
        formatter = logging.Formatter(
            fmt=(
                "%(asctime)s "
                "[%(levelname)-8s] "   # -8s = left-align in 8 char wide column
                "%(name)-35s : "       # -35s = left-align in 35 char wide column
                "%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ── Attach formatter to handler, handler to logger ────────────────────
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # ── Prevent logs from bubbling up to root logger ──────────────────────
        # Without this, some messages print twice —
        # once from our handler and once from Python's root logger
        logger.propagate = False

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to see how logs look from different modules.
#
# Usage:
#   python src/utils/logger.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Simulate loggers from different modules
    # In real code each file calls get_logger(__name__)
    orchestrator_log  = get_logger("src.agents.orchestrator")
    research_log      = get_logger("src.agents.research_agent")
    data_log          = get_logger("src.agents.data_agent")
    filing_log        = get_logger("src.agents.filing_agent")
    report_log        = get_logger("src.agents.report_agent")
    tools_log         = get_logger("src.tools.news_search")
    rag_log           = get_logger("src.rag.retriever")

    print("\n── Simulating a full pipeline run ──────────────────────\n")

    # Orchestrator
    orchestrator_log.info("Pipeline started for NVIDIA Corporation (NVDA)")
    orchestrator_log.info("Research plan created — 4 steps")

    # Parallel agents starting
    research_log.info("Fetching news articles for NVDA")
    data_log.info("Connecting to Yahoo Finance for NVDA")
    filing_log.info("Downloading SEC 10-K filing for NVDA")

    # Tool level logs
    tools_log.info("Tavily API call — query: 'NVIDIA latest news 2026'")
    tools_log.warning("Tavily returned only 8 results — expected 10")

    # RAG logs
    rag_log.info("Chunking document — 1842 tokens across 3 chunks")
    rag_log.info("Retrieval iteration 1 — query: 'revenue breakdown by segment'")
    rag_log.info("Sufficiency check: SUFFICIENT — moving to report")

    # Agents completing
    research_log.info("Done — sentiment: positive (78%), 8 articles analyzed")
    data_log.info("Done — price: $321.5, P/E: 38.5x")
    filing_log.info("Done — revenue breakdown and risk factors extracted")

    # Report
    report_log.info("All parallel agents complete — synthesizing report")
    report_log.info("Report generated — 1420 characters")
    report_log.info("Email sent to recipient@example.com")

    # Error example
    tools_log.error("SEC EDGAR request timed out — retry 1 of 3")
    tools_log.error("SEC EDGAR request timed out — retry 2 of 3")
    tools_log.warning("SEC EDGAR request succeeded on retry 3 — continuing")

    print("\n── Log level test (only INFO and above should show) ────\n")

    test_log = get_logger("test.module")
    test_log.debug("This is DEBUG — hidden when LOG_LEVEL=INFO")
    test_log.info("This is INFO — visible")
    test_log.warning("This is WARNING — visible")
    test_log.error("This is ERROR — visible")

    print("\n── Duplicate handler test ──────────────────────────────\n")

    # Calling get_logger twice with the same name should NOT duplicate messages
    logger_a = get_logger("test.duplicate")
    logger_b = get_logger("test.duplicate")
    logger_a.info("This message should appear exactly ONCE")
    logger_b.info("This message should also appear exactly ONCE")
    print()