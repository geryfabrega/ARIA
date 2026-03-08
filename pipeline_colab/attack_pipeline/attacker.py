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

    # Drop common leading labels like "Attack prompt:".
    text = _LABEL_PREFIX.sub("", text).strip()
    lower = text.lower()

    # If the model echoed our mutation scaffold ("Original goal", "Previous test prompt",
    # "Feedback …") then try to *extract* just the revised prompt section.
    if "original goal:" in lower and "previous test prompt:" in lower:
        # If the model never produced an explicit revised prompt section, discard the
        # whole scaffold and fall back to the previous prompt.
        has_revised_label = "revised test prompt:" in lower or "revised prompt:" in lower
        if not has_revised_label:
            return fallback

        # Otherwise, keep only the text after the last known "revised" label.
        for label in ("revised test prompt:", "revised prompt:"):
            idx = lower.rfind(label)
            if idx != -1:
                text = text[idx + len(label) :].strip()
                lower = text.lower()
                break

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
