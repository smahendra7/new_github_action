
"""
Central NVIDIA API rate limiter for CodeGate.

Features:
- One NVIDIA request at a time per Python process.
- Minimum 2-second interval between request starts.
- Maximum 30 request starts in a rolling 60-second window.
- Individual retry counter for every API request.
- Exponential backoff: 5, 10, 20, 40, 80 seconds.
- Respect NVIDIA Retry-After header.
- Shared cooldown after HTTP 429.
- Normal and streaming request support.
"""

import threading
import time

from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import APIConnectionError

from ai.context_config import (
    NVIDIA_ACCOUNT_RPM,
    NVIDIA_TARGET_RPM,
    NVIDIA_MIN_REQUEST_INTERVAL,
    NVIDIA_MAX_RETRIES,
    NVIDIA_RETRY_BACKOFF,
)


# ============================================================
# GLOBAL STATE
# ============================================================

NVIDIA_API_LOCK = threading.Lock()

_last_request_started = 0.0

_cooldown_until = 0.0

_request_start_times = deque()


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# REQUEST PACING AND ROLLING RPM LIMIT
# ============================================================

def _wait_for_request_slot():
    """
    Must be called while NVIDIA_API_LOCK is held.

    Enforces:
    1. Minimum interval between request starts.
    2. Target RPM using a rolling 60-second window.
    3. Shared cooldown after HTTP 429.
    """

    global _last_request_started

    target_rpm = max(
        1,
        min(NVIDIA_TARGET_RPM, NVIDIA_ACCOUNT_RPM)
    )

    while True:

        now = time.monotonic()

        # Remove request timestamps outside the rolling
        # 60-second window.
        while (
            _request_start_times
            and now - _request_start_times[0] >= 60.0
        ):
            _request_start_times.popleft()

        # Minimum interval between request starts.
        interval_wait = (
            _last_request_started
            + NVIDIA_MIN_REQUEST_INTERVAL
            - now
        )

        # Shared cooldown after HTTP 429.
        cooldown_wait = _cooldown_until - now

        wait_time = max(
            0.0,
            interval_wait,
            cooldown_wait,
        )

        # Enforce maximum requests in a rolling 60-second window.
        if len(_request_start_times) >= target_rpm:

            rpm_wait = (
                _request_start_times[0]
                + 60.0
                - now
            )

            wait_time = max(
                wait_time,
                rpm_wait,
            )

        if wait_time <= 0:
            break

        print(
            f"[NVIDIA RATE LIMIT] Waiting "
            f"{wait_time:.2f} seconds before next request."
        )

        time.sleep(wait_time)

    # Record the actual request start time.
    now = time.monotonic()

    _last_request_started = now

    _request_start_times.append(now)


# ============================================================
# ERROR DETECTION
# ============================================================

def _get_status_code(exc):
    return getattr(exc, "status_code", None)


def _is_retryable_error(exc):
    """
    Retry only known transient provider errors.
    """

    if isinstance(exc, APIConnectionError):
        return True

    return (
        _get_status_code(exc)
        in RETRYABLE_STATUS_CODES
    )


# ============================================================
# RETRY-AFTER HEADER
# ============================================================

def _get_retry_after(exc, fallback_delay):
    """
    Read Retry-After from NVIDIA's HTTP response.

    Supports both:
    - Seconds
    - HTTP date
    """

    try:

        response = getattr(
            exc,
            "response",
            None
        )

        if response is None:
            return fallback_delay

        headers = response.headers

        retry_after = headers.get("retry-after")

        if not retry_after:
            return fallback_delay

        # Retry-After expressed in seconds.
        try:

            return max(
                0.0,
                float(retry_after)
            )

        except (ValueError, TypeError):
            pass

        # Retry-After expressed as an HTTP date.
        try:

            retry_time = parsedate_to_datetime(
                retry_after
            )

            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(
                    tzinfo=timezone.utc
                )

            return max(
                0.0,
                (
                    retry_time
                    - datetime.now(timezone.utc)
                ).total_seconds()
            )

        except (ValueError, TypeError, OverflowError):
            pass

    except Exception:
        pass

    return fallback_delay


# ============================================================
# SHARED COOLDOWN
# ============================================================

def _set_shared_cooldown(delay):
    """
    Must be called while NVIDIA_API_LOCK is held.
    """

    global _cooldown_until

    new_cooldown = (
        time.monotonic() + delay
    )

    _cooldown_until = max(
        _cooldown_until,
        new_cooldown,
    )


# ============================================================
# RETRY DELAY CALCULATION
# ============================================================

def _prepare_retry(exc, attempt):
    """
    Each individual API request has its own retry sequence.

    attempt 0 -> 5 seconds
    attempt 1 -> 10 seconds
    attempt 2 -> 20 seconds
    attempt 3 -> 40 seconds
    attempt 4 -> 80 seconds
    """

    fallback_delay = NVIDIA_RETRY_BACKOFF[
        min(attempt, len(NVIDIA_RETRY_BACKOFF) - 1)
    ]

    delay = _get_retry_after(
        exc,
        fallback_delay,
    )

    status_code = _get_status_code(exc)

    # Preserve cooldown across individual API requests.
    if status_code == 429:
        _set_shared_cooldown(delay)

    print(
        f"[NVIDIA RETRY] "
        f"HTTP {status_code or 'CONNECTION ERROR'} | "
        f"Attempt {attempt + 1}/{NVIDIA_MAX_RETRIES} | "
        f"Waiting {delay:.2f} seconds."
    )

    return delay


# ============================================================
# NORMAL NON-STREAMING REQUEST
# ============================================================

def execute_nvidia_request(request_function):
    """
    Execute one normal NVIDIA API request.

    Example:

        completion = execute_nvidia_request(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
        )

    The lock remains held until the API response is received.

    Every new API request starts its own retry counter.
    """

    with NVIDIA_API_LOCK:

        for attempt in range(NVIDIA_MAX_RETRIES + 1):

            _wait_for_request_slot()

            try:

                return request_function()

            except Exception as exc:

                if not _is_retryable_error(exc):
                    raise

                delay = _prepare_retry(
                    exc,
                    attempt,
                )

                if attempt >= NVIDIA_MAX_RETRIES:

                    print(
                        "[NVIDIA ERROR] Maximum retries exhausted."
                    )

                    raise

                time.sleep(delay)

    raise RuntimeError(
        "NVIDIA request failed unexpectedly."
    )


# ============================================================
# STREAMING REQUEST
# ============================================================

def execute_nvidia_stream(request_function):
    """
    Execute a streaming NVIDIA API request.

    The lock remains held until the complete stream finishes.

    If an error occurs before receiving the first chunk,
    transient errors can be retried.

    If an error occurs after receiving chunks, the stream
    is not restarted because that could duplicate output.
    """

    with NVIDIA_API_LOCK:

        for attempt in range(NVIDIA_MAX_RETRIES + 1):

            _wait_for_request_slot()

            stream = None

            chunks_received = 0

            try:

                stream = request_function()

                try:

                    for chunk in stream:

                        chunks_received += 1

                        yield chunk

                    return

                finally:

                    close_stream = getattr(
                        stream,
                        "close",
                        None
                    )

                    if callable(close_stream):

                        try:
                            close_stream()
                        except Exception:
                            pass

            except Exception as exc:

                # Do not restart a partially delivered stream.
                if chunks_received > 0:
                    raise

                if not _is_retryable_error(exc):
                    raise

                delay = _prepare_retry(
                    exc,
                    attempt,
                )

                if attempt >= NVIDIA_MAX_RETRIES:

                    print(
                        "[NVIDIA STREAM ERROR] "
                        "Maximum retries exhausted."
                    )

                    raise

                time.sleep(delay)

    raise RuntimeError(
        "NVIDIA streaming request failed unexpectedly."
    )
