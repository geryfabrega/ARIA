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
    """Normalize model output and drop obvious scaffold echos.

    The target model must receive ONLY the jailbreak prompt string—never the
    mutation context (goal, previous prompt, feedback). If the attacker echoes
    that scaffold, we extract the revised prompt or fall back to previous.
    """
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

    # Detect ANY echo of our mutation scaffold (we send "Original goal", "Previous test
    # prompt", "Feedback on why it failed"). The model may echo "Revised goal" or
    # "Previous test prompt" etc. If this output contains those markers, we must NOT
    # send it to the target—extract revised prompt or use fallback.
    scaffold_markers = (
        "previous test prompt",
        "feedback on why it failed",
        "original goal:",
        "revised goal:",
    )
    looks_like_scaffold = any(m in lower for m in scaffold_markers)

    if looks_like_scaffold:
        # Try to extract only the revised prompt section.
        revised_labels = ("revised test prompt:", "revised prompt:")
        for label in revised_labels:
            idx = lower.rfind(label)
            if idx != -1:
                after = text[idx + len(label) :].strip()
                # If what follows is still huge and contains scaffold, don't use it.
                if after and not any(m in after.lower() for m in scaffold_markers):
                    return after.strip() or fallback
                break
        # No clean revised section, or revised section still contained scaffold: do not
        # send scaffold to target.
        return fallback

    # Meta-commentary: model wrote *about* the prompt ("To respond to the feedback...",
    # "For example:", "This revised prompt encourages") instead of the prompt itself.
    # Don't send that to the target—try to extract a quoted prompt or use fallback.
    meta_markers = (
        "to respond to the feedback",
        "for example:",
        "this revised prompt",
        "you may modify the original",
        "you may modify the",
    )
    if any(m in lower for m in meta_markers):
        # Try to extract the first substantial quoted string (the actual prompt).
        match = re.search(r'"([^"]{20,})"', text)
        if match:
            return match.group(1).strip()
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
