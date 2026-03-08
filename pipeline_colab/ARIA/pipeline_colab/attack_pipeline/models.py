"""Thin litellm wrapper with retry logic shared by all pipeline roles."""

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress litellm's verbose success logging
litellm.success_callback = []
litellm.set_verbose = False


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_model(
    model: str,
    messages: list[dict],
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    response_format: dict | None = None,
) -> str:
    """Call any litellm-supported model and return the response text.

    Retries up to 3 times with exponential backoff on any exception.
    """
    kwargs: dict = dict(
        model=model,
        messages=messages,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    return content.strip() if content else ""
