"""Model wrapper with retry logic for hosted APIs and local HF models."""

import os

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress litellm's verbose success logging
litellm.success_callback = []
litellm.set_verbose = False

_LOCAL_GENERATORS: dict[str, tuple[object, object]] = {}


def _load_local_generator(model_id: str) -> tuple[object, object]:
    """Lazy-load and cache a local text-generation pipeline + tokenizer."""
    if model_id in _LOCAL_GENERATORS:
        return _LOCAL_GENERATORS[model_id]

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        device_map="auto",
        torch_dtype="auto",
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    _LOCAL_GENERATORS[model_id] = (generator, tokenizer)
    return generator, tokenizer


def _call_local_hf(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """Call a local HF model via transformers when model starts with hf_local:."""
    model_id = model.split(":", 1)[1]
    generator, tokenizer = _load_local_generator(model_id)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_full_text=False,
    )
    return outputs[0]["generated_text"].strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_model(
    model: str,
    messages: list[dict],
    api_key: str = "",
    max_tokens: int = 512,
    temperature: float = 0.7,
    response_format: dict | None = None,
) -> str:
    """Call any litellm-supported model and return the response text.

    Retries up to 3 times with exponential backoff on any exception.
    """
    if model.startswith("hf_local:"):
        return _call_local_hf(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

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
