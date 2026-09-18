import sys
import os
from groq import Groq


def review_code(prompt: str) -> str:

    print("\n===== PROMPT LENGTH =====")
    print(len(prompt))
    print("=========================\n")

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    print("\n===== AI REVIEW (STREAMING LIVE) =====")

    review_buffer = []

    try:
        # Replace characters that cannot be encoded as UTF-8 before sending the prompt to Groq.
        safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": safe_prompt
                }
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=True,
            stop=None,
        )

        for chunk in completion:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content

                # Append to our local aggregator buffer
                review_buffer.append(token)

                # Instantly write the token out to the console terminal
                sys.stdout.write(token)
                sys.stdout.flush()

    except Exception as e:
        print(f"\nGroq inference engine failure: {e}")
        return "Error occurred during AI code review."

    print("\n======================================")

    # Reassemble individual tokens into one complete string response body
    full_review = "".join(review_buffer).strip()
    return full_review if full_review else "No review generated."
