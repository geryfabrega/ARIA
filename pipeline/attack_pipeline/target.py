"""Target model: Llama 3.1 8B Instruct via Together AI."""

from .config import TARGET_MAX_TOKENS, TARGET_MODEL, TARGET_TEMPERATURE
from .models import call_model


def query_target(attack_prompt: str, api_key: str) -> str:
    """Send an attack prompt to the target model and return its raw response.

    No system prompt is set so the model responds without safety framing.
    """
    messages = [{"role": "user", "content": attack_prompt}]
    return call_model(
        model=TARGET_MODEL,
        messages=messages,
        api_key=api_key,
        max_tokens=TARGET_MAX_TOKENS,
        temperature=TARGET_TEMPERATURE,
    )
