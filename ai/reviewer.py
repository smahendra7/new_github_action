import sys
import os
import threading
from groq import Groq
from openai import OpenAI
from ai.context_config import (
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
)

from ai.nvidia_request_lock import execute_nvidia_stream

_stdout_lock = threading.Lock()


class ReviewResult(str):
    """String subclass that carries the review completion status."""

    status: str

    def __new__(cls, content: str, status: str = "completed"):
        obj = super().__new__(cls, content)
        obj.status = status
        return obj


def review_code_with_status(prompt: str) -> tuple[str, str]:
    """
    Execute the AI code review and return (status, review_text).

    Status values:
        completed  - The model returned a complete response.
        incomplete - The response was empty or truncated.
        failed     - The provider request or stream failed.
    """

    print("\n===== PROMPT LENGTH =====")
    print(len(prompt))
    print("=========================\n")

    # NVIDIA OpenAI-compatible client (active provider).
    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=os.environ["NVIDIA_API_KEY"],
    )

    # Groq client kept for reference (inactive).
    # groq_client = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY")
    # )

    print("\n===== AI REVIEW (STREAMING LIVE) =====")

    review_buffer = []
    reasoning_received = False
    finish_reason = None

    try:
        # Ensure the prompt can be encoded as UTF-8.
        safe_prompt = prompt.encode(
            "utf-8",
            errors="replace",
        ).decode("utf-8")

        completion = execute_nvidia_stream(
            lambda: client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": safe_prompt,
                    }
                ],
                temperature=0,
                max_tokens=4096,
                top_p=1,
                reasoning_effort="low",
                stream=True,
            )
        )

        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            # Track reasoning without printing or publishing it.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_received = True

            # Collect and stream only the actual review text.
            content = getattr(delta, "content", None)

            if content:
                review_buffer.append(content)

                with _stdout_lock:
                    sys.stdout.write(content)
                    sys.stdout.flush()

    except Exception as e:
        print(f"\nNVIDIA inference engine failure: {e}")

        partial_review = "".join(review_buffer).strip()

        if partial_review:
            partial_review += (
                "\n\n### ⚠️ Review Notices\n"
                "- `AI review` (failed): "
                "The provider stream failed before completion. "
                "The displayed findings may be incomplete."
            )

        return (
            "failed",
            partial_review
            or f"Error occurred during AI code review: {e}",
        )

    print("\n======================================")

    print(f"[DEBUG] Reasoning received: {reasoning_received}")
    print(f"[DEBUG] Final content received: {bool(review_buffer)}")
    print(f"[DEBUG] Finish reason: {finish_reason}")

    full_review = "".join(review_buffer).strip()

    if not full_review:
        print("[DEBUG] AI returned no usable review text.")
        return "incomplete", "No review generated."

    # A length finish reason means the model may have stopped before
    # completing its review. Preserve the response but mark it incomplete.
    if finish_reason == "length":
        print("[DEBUG] Review hit generation length limit.")
        return "incomplete", full_review

    # If the stream ended without an explicit finish reason, completion
    # cannot be confirmed. Preserve the response and flag it as incomplete.
    if finish_reason is None:
        print("[DEBUG] Stream ended without a finish reason.")
        return "incomplete", full_review

    if finish_reason != "stop":
        print(f"[DEBUG] Unexpected finish reason: {finish_reason}")
        return "incomplete", full_review

    return "completed", full_review


def review_code(prompt: str) -> ReviewResult:
    """Run the review and return its text with completion status attached."""
    status, review_text = review_code_with_status(prompt)
    return ReviewResult(review_text, status=status)
