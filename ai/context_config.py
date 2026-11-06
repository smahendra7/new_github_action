"""
ai/context_config.py

Configuration constants for AI code review context sizing, token limits, and chunking.
"""
import os

# Target input token budget.
# 120,000 input tokens + 4,096 max output = 124,096 tokens,
# leaving ~6,976 tokens safety margin below the observed 131,072-token endpoint limit.
SAFE_INPUT_TOKEN_LIMIT: int = 120_000

# Maximum number of concurrent AI requests to NVIDIA NIM.
MAX_CONCURRENCY: int = 1
# ============================================================
# NVIDIA API RATE LIMIT CONFIGURATION
# ============================================================

# NVIDIA account RPM limit
NVIDIA_ACCOUNT_RPM: int = 40

# CodeGate target RPM
NVIDIA_TARGET_RPM: int = 30

# Minimum interval between request starts
NVIDIA_MIN_REQUEST_INTERVAL: float = 2.0

# Maximum retries per individual API request
NVIDIA_MAX_RETRIES: int = 5

# Exponential backoff delays (seconds)
NVIDIA_RETRY_BACKOFF: tuple = (
    5, 10, 20, 40, 80
)


# ============================================================
# NVIDIA MODEL CONFIGURATION
# ============================================================

NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

NVIDIA_MODEL: str = os.getenv(
    "NVIDIA_MODEL",
    "poolside/laguna-xs-2.1"
)

# Number of lines of unchanged context to include above and below each diff hunk.
CONTEXT_WINDOW_LINES: int = 50

# Maximum number of lines permitted for an extracted enclosing function/context block.
# If an enclosing function/method exceeds this limit, only a limited window around the change is sent.
MAX_CONTEXT_LINES: int = 50

# Fallback context window: lines before and after changed lines when AST parsing fails or function is too large.
FALLBACK_CONTEXT_WINDOW_LINES: int = 15

# Maximum output tokens reserved per chunk request (matches reviewer.py max_tokens).
MAX_OUTPUT_TOKENS: int = 8192

# Overlap in lines when a large continuous hunk/region must be split into sub-chunks.
CHUNK_OVERLAP_LINES: int = 30

# Legacy character threshold kept for backward compatibility if needed.
SAFE_PROMPT_CHAR_LIMIT: int = 360_000
