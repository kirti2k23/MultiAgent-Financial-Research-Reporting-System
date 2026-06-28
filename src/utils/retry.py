"""
src/utils/retry.py
==================
Exponential backoff retry decorator for all external API calls.

Every tool in src/tools/ wraps its API calls with this decorator.
If an API call fails transiently, it automatically retries with
increasing wait times instead of crashing the entire pipeline.

WHAT IS EXPONENTIAL BACKOFF:
    1st failure → wait 2s  → retry
    2nd failure → wait 4s  → retry
    3rd failure → wait 8s  → retry
    4th failure → give up, raise the original error

WHY EXPONENTIAL (not fixed wait):
    Fixed wait hammers the API repeatedly.
    Exponential wait gives the API time to recover.
    This is the industry standard pattern used by AWS, Google, etc.

USAGE:
    from src.utils.retry import retry

    @retry(max_attempts=3, initial_wait=2)
    def call_external_api():
        response = requests.get("https://api.example.com/data")
        return response

    # Only retry on network errors — not on ValueError or TypeError
    @retry(max_attempts=3, initial_wait=2, exceptions=(ConnectionError, TimeoutError))
    def call_with_specific_retry():
        response = requests.get("https://api.example.com/data")
        return response

    # Now call_external_api() auto-retries on any exception
    result = call_external_api()
"""

import time
import functools
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    initial_wait: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator factory that adds exponential backoff retry to any function.

    HOW DECORATOR FACTORY WORKS:
        retry(max_attempts=3)     ← this call returns a decorator
        @that_decorator           ← which is applied to the function
        def my_function(): ...

        So retry() is a function that RETURNS a decorator.
        That decorator WRAPS the original function with retry logic.

    Args:
        max_attempts: Total number of tries including the first attempt.
                      Default 3 means: 1 try + 2 retries.
        initial_wait: Seconds to wait after first failure.
                      Doubles after each failure (exponential backoff).
                      Default 2 means: wait 2s, then 4s, then 8s.
        exceptions:   Tuple of exception types to retry on.
                      Default is (Exception,) — retries on everything.
                      Pass specific types to avoid retrying on logic errors.

                      Example:
                        @retry(exceptions=(ConnectionError, TimeoutError))
                        def fetch():  ...
                        → retries on network errors only
                        → ValueError / KeyError raised immediately, no retry

    Returns:
        A decorator that wraps the target function with retry logic.
    """

    def decorator(func):
        """
        The actual decorator — receives the function being wrapped.
        Returns wrapper which replaces the original function.
        """

        @functools.wraps(func)
        # functools.wraps copies the original function's name and docstring
        # onto wrapper. Without it, all wrapped functions would appear
        # to be named "wrapper" in logs and stack traces.
        def wrapper(*args, **kwargs):
            """
            Wrapper replaces the original function.
            Calls the original function inside a retry loop.

            *args, **kwargs:
                Accepts any arguments so this wrapper works with
                any function regardless of its signature.
            """

            # Track the last exception to re-raise if all attempts fail
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    # ── Try calling the original function ─────────────────────
                    # If it succeeds, return immediately — no retry needed
                    result = func(*args, **kwargs)

                    # Log successful retry (not the first attempt)
                    if attempt > 1:
                        logger.info(
                            f"{func.__name__} succeeded on attempt "
                            f"{attempt}/{max_attempts}"
                        )
                    return result

                except Exception as e:
                    # ── Function failed ───────────────────────────────────────
                    last_exception = e

                    # ── Check if this exception type should be retried ────────
                    # If the exception is NOT in the allowed retry types,
                    # re-raise it immediately — no waiting, no retrying.
                    # Example: ValueError for invalid ticker should never retry.
                    if not isinstance(e, exceptions):
                        logger.error(
                            f"{func.__name__} raised {type(e).__name__} "
                            f"(not retryable) — failing immediately: {e}"
                        )
                        raise

                    if attempt == max_attempts:
                        # All attempts exhausted — give up
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} "
                            f"attempts. Final error: {e}"
                        )
                        # Re-raise the original exception
                        # so the caller knows what went wrong
                        raise

                    # ── Calculate wait time ───────────────────────────────────
                    # initial_wait doubles each attempt
                    # attempt=1 → wait = 2 * (2^0) = 2s
                    # attempt=2 → wait = 2 * (2^1) = 4s
                    # attempt=3 → wait = 2 * (2^2) = 8s
                    wait_seconds = initial_wait * (2 ** (attempt - 1))

                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {e}. "
                        f"Retrying in {wait_seconds:.0f}s..."
                    )

                    # ── Wait before retrying ──────────────────────────────────
                    time.sleep(wait_seconds)

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# Run directly to see retry in action.
#
# Usage:
#   python src/utils/retry.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n── Test 1: Function succeeds on first try ───────────────\n")

    @retry(max_attempts=3, initial_wait=1)
    def always_succeeds():
        logger.info("API call succeeded immediately")
        return {"data": "success"}

    result = always_succeeds()
    print(f"  Result: {result}\n")


    print("── Test 2: Function fails twice then succeeds ───────────\n")

    # Counter to track how many times the function was called
    call_count = {"n": 0}

    @retry(max_attempts=3, initial_wait=1)
    def fails_twice_then_succeeds():
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError(f"Simulated API timeout (attempt {call_count['n']})")
        logger.info("API call succeeded on attempt 3")
        return {"data": "success after retry"}

    result = fails_twice_then_succeeds()
    print(f"  Result: {result}\n")


    print("── Test 3: Function always fails — exhausts all attempts ─\n")

    @retry(max_attempts=3, initial_wait=1)
    def always_fails():
        raise TimeoutError("Simulated permanent API failure")

    try:
        always_fails()
    except TimeoutError as e:
        print(f"  ✅ Correctly raised after all attempts: {e}\n")


    print("── Test 4: Verify wait times are exponential ────────────\n")

    times = []
    attempt_count = {"n": 0}

    @retry(max_attempts=4, initial_wait=1)
    def track_timing():
        attempt_count["n"] += 1
        times.append(time.time())
        if attempt_count["n"] < 4:
            raise ValueError("Simulated failure")
        return "done"

    track_timing()

    print("  Wait times between attempts:")
    for i in range(1, len(times)):
        wait = times[i] - times[i - 1]
        expected = 1 * (2 ** (i - 1))
        print(f"    Attempt {i} → {i+1}: {wait:.1f}s (expected ~{expected}s)")
    print()