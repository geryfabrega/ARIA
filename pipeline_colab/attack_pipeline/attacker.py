"""Attack-prompt generator for local HF chat models."""

from .config import (
    ATTACKER_MAX_TOKENS,
    ATTACKER_MODEL,
    ATTACKER_SYSTEM_PROMPT,
    ATTACKER_TEMPERATURE,
    MUTATION_SYSTEM_PROMPT,
)
from .models import call_model


def generate_initial_attack(goal: str, api_key: str) -> str:
    """Generate a first-attempt attack prompt for the given JBB behavior goal."""
    messages = [
        {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Behavior goal: {goal}"},
    ]
    return call_model(
        model=ATTACKER_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=ATTACKER_MAX_TOKENS,
        temperature=ATTACKER_TEMPERATURE,
    )


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
    return call_model(
        model=ATTACKER_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=ATTACKER_MAX_TOKENS,
        temperature=ATTACKER_TEMPERATURE,
    )
