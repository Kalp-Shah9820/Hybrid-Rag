"""
Centralized Google GenAI integration for production use.

- Uses the current google-genai SDK
- Validates configured model names
- Selects from an ordered fallback list when needed
- Exposes one text-generation helper for the rest of the app
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from google import genai
from google.genai import types
from loguru import logger

from src.config import settings


def _normalize_model_name(name: str) -> str:
    """Convert API model paths into plain model ids."""
    return name.split("/", 1)[-1] if name.startswith("models/") else name


def _extract_response_text(response) -> str:
    """Return text content from a GenerateContent response."""
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    parts: List[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            value = getattr(part, "text", None)
            if value:
                parts.append(value)

    return "\n".join(parts).strip()


def _get_attr(obj, *names):
    """Return the first present attribute from a list of snake/camel variants."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _describe_empty_response(response) -> str:
    """Build a concrete diagnostic for successful-but-empty Gemini responses."""
    prompt_feedback = _get_attr(response, "prompt_feedback", "promptFeedback")
    if prompt_feedback:
        block_reason = _get_attr(prompt_feedback, "block_reason", "blockReason")
        block_message = _get_attr(
            prompt_feedback,
            "block_reason_message",
            "blockReasonMessage",
        )
        if block_reason:
            return (
                "Gemini returned no candidates because the prompt was blocked. "
                f"block_reason={block_reason}, message={block_message or 'n/a'}"
            )

    candidates = _get_attr(response, "candidates") or []
    if not candidates:
        usage = _get_attr(response, "usage_metadata", "usageMetadata")
        total_tokens = _get_attr(usage, "total_token_count", "totalTokenCount")
        return (
            "Gemini returned no candidates and no text output. "
            f"total_tokens={total_tokens if total_tokens is not None else 'unknown'}"
        )

    candidate = candidates[0]
    finish_reason = _get_attr(candidate, "finish_reason", "finishReason")
    finish_message = _get_attr(candidate, "finish_message", "finishMessage")
    safety_ratings = _get_attr(candidate, "safety_ratings", "safetyRatings")

    return (
        "Gemini returned an empty candidate. "
        f"finish_reason={finish_reason or 'unknown'}, "
        f"finish_message={finish_message or 'n/a'}, "
        f"safety_ratings={safety_ratings or 'n/a'}"
    )


def _extract_finish_reason(response) -> str:
    """Extract the first candidate finish reason as a normalized string."""
    candidates = _get_attr(response, "candidates") or []
    if not candidates:
        return ""
    finish_reason = _get_attr(candidates[0], "finish_reason", "finishReason")
    return str(finish_reason or "")


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Create a singleton Gemini client from the environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY in your environment or .env file."
        )
    return genai.Client(api_key=api_key)


def list_available_models() -> List[str]:
    """Return normalized model ids visible to the configured API key."""
    models: List[str] = []
    for model in get_gemini_client().models.list():
        name = _normalize_model_name(getattr(model, "name", ""))
        supported_actions = getattr(model, "supported_actions", None) or []

        if not name.startswith("gemini-"):
            continue

        if supported_actions and "generateContent" not in supported_actions:
            continue

        models.append(name)

    unique_models = sorted(set(models))
    logger.info(f"Gemini model discovery returned {len(unique_models)} candidate models")
    return unique_models


def get_candidate_models(preferred_model: Optional[str] = None) -> List[str]:
    """Return ordered unique model candidates from config."""
    candidates = [
        preferred_model or settings.generator.model,
        *settings.generator.fallback_models,
    ]

    ordered: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_model_name(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


@lru_cache(maxsize=8)
def resolve_generation_model(preferred_model: Optional[str] = None) -> str:
    """
    Resolve the best available Gemini model for text generation.

    Falls back across an ordered candidate list. Raises an actionable error if
    none of the configured candidates are available to this API key.
    """
    candidates = get_candidate_models(preferred_model)
    available_models = set(list_available_models())

    for candidate in candidates:
        if candidate in available_models:
            if candidate != settings.generator.model:
                logger.warning(
                    f"Configured Gemini model '{settings.generator.model}' is unavailable. "
                    f"Using fallback model '{candidate}'."
                )
            return candidate

    raise RuntimeError(
        "None of the configured Gemini models are available for this API key. "
        f"Tried: {candidates}. Visible models: {sorted(available_models)[:20]}"
    )


def validate_gemini_configuration() -> str:
    """Validate Gemini connectivity and return the selected generation model."""
    model = resolve_generation_model()
    logger.info(f"Gemini model validated successfully: {model}")
    return model


def generate_text(
    prompt: str,
    *,
    system_instruction: str,
    preferred_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """Generate text through the centralized Gemini client."""
    response = get_gemini_client().models.generate_content(
        model=resolve_generation_model(preferred_model),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=(
                settings.generator.temperature if temperature is None else temperature
            ),
            max_output_tokens=(
                settings.generator.max_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
            thinking_config=types.ThinkingConfig(
                thinking_budget=(
                    settings.generator.thinking_budget
                    if thinking_budget is None
                    else thinking_budget
                )
            ),
        ),
    )

    text = _extract_response_text(response)
    if not text:
        raise RuntimeError(_describe_empty_response(response))
    return text


def generate_text_with_retry(
    prompt: str,
    *,
    system_instruction: str,
    preferred_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    retry_max_output_tokens: Optional[int] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """
    Generate text and retry once with a larger output cap if Gemini stops on MAX_TOKENS.
    """
    first_cap = settings.generator.max_tokens if max_output_tokens is None else max_output_tokens
    retry_cap = (
        settings.generator.retry_max_tokens
        if retry_max_output_tokens is None
        else retry_max_output_tokens
    )

    try:
        return generate_text(
            prompt,
            system_instruction=system_instruction,
            preferred_model=preferred_model,
            temperature=temperature,
            max_output_tokens=first_cap,
            thinking_budget=thinking_budget,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "MAX_TOKENS" not in message or retry_cap <= first_cap:
            raise

        logger.warning(
            "Gemini stopped on MAX_TOKENS. Retrying with a larger output budget "
            f"({first_cap} -> {retry_cap})."
        )
        return generate_text(
            prompt,
            system_instruction=system_instruction,
            preferred_model=preferred_model,
            temperature=temperature,
            max_output_tokens=retry_cap,
            thinking_budget=thinking_budget,
        )


def print_available_models() -> None:
    """Utility for CLI scripts."""
    models = list_available_models()
    if not models:
        print("No Gemini generation models were visible for this API key.")
        return

    print("Available Gemini models:")
    for model in models:
        print(f" - {model}")
