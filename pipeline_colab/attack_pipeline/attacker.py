"""Attack-prompt generator for local HF chat models."""

import re

from .config import (
    ATTACKER_MAX_TOKENS,
    ATTACKER_MODEL,
    ATTACKER_SYSTEM_PROMPT,
    ATTACKER_TEMPERATURE,
    MUTATION_SYSTEM_PROMPT,
)
from .models import call_model


_LABEL_PREFIX = re.compile(
    r"^\s*(?:attack prompt|revised test prompt|revised prompt|prompt)\s*:\s*",
    re.IGNORECASE,
)


def _clean_attack_prompt(raw: str, fallback: str) -> str:
    """Normalize model output and drop obvious scaffold echos."""
    text = (raw or "").strip()
    if not text:
        return fallback

    # Remove common markdown wrappers.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    text = _LABEL_PREFIX.sub("", text)

    # If the model echoed mutation scaffolding, keep prior prompt instead.
    lower = text.lower()
    echoed_scaffold = "original goal:" in lower and "previous test prompt:" in lower
    if echoed_scaffold and "revised test prompt:" not in lower and "revised prompt:" not in lower:
        return fallback

    return text.strip() or fallback


def generate_initial_attack(goal: str, api_key: str) -> str:
    """Generate a first-attempt attack prompt for the given JBB behavior goal."""
    messages = [
        {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Behavior goal: {goal}"},
    ]
    raw = call_model(
        model=ATTACKER_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=ATTACKER_MAX_TOKENS,
        temperature=ATTACKER_TEMPERATURE,
    )
    return _clean_attack_prompt(raw, fallback=goal)


def mutate_attack(goal: str, previous_prompt: str, feedback: str, api_key: str) -> str:
    """Produce a revised attack prompt incorporating the judge's feedback."""
    user_content = (
        f"Original goal: {goal}\n\n"
        f"Previous test prompt:\n{previous_prompt}\n\n"
        f"Feedback on why it failed:\n{feedback}"
    )
    messages = [
        {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = call_model(
        model=ATTACKER_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=ATTACKER_MAX_TOKENS,
        temperature=ATTACKER_TEMPERATURE,
    )
    return _clean_attack_prompt(raw, fallback=previous_prompt)
