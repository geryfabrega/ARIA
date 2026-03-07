"""Feedback model: GPT model returning mutation hints for failed attacks."""

from .config import FEEDBACK_MAX_TOKENS, FEEDBACK_MODEL, FEEDBACK_SYSTEM_PROMPT, FEEDBACK_TEMPERATURE
from .models import call_model


def get_feedback(
    goal: str,
    attack_prompt: str,
    target_response: str,
    judge_reason: str,
    api_key: str,
) -> str:
    """Return 2–3 sentences of actionable feedback on why the attack failed."""
    user_content = (
        f"Original goal: {goal}\n\n"
        f"Test prompt used:\n{attack_prompt}\n\n"
        f"Target model response:\n{target_response}\n\n"
        f"Judge's reason for failure: {judge_reason}"
    )
    messages = [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return call_model(
        model=FEEDBACK_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=FEEDBACK_MAX_TOKENS,
        temperature=FEEDBACK_TEMPERATURE,
    )
