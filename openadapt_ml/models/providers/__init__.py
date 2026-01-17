"""API Provider implementations for VLM backends.

This module provides a unified interface for different API providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)

Usage:
    from openadapt_ml.models.providers import get_provider

    provider = get_provider("anthropic")
    client = provider.create_client(api_key)
    response = provider.send_message(client, model, system, content)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openadapt_ml.models.providers.base import BaseAPIProvider
from openadapt_ml.models.providers.anthropic import AnthropicProvider
from openadapt_ml.models.providers.openai import OpenAIProvider
from openadapt_ml.models.providers.google import GoogleProvider

if TYPE_CHECKING:
    pass

__all__ = [
    "BaseAPIProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "get_provider",
    "PROVIDERS",
]

# Provider registry
PROVIDERS: dict[str, type[BaseAPIProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

# Model aliases for convenience
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude-opus-4.5": ("anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet-4.5": ("anthropic", "claude-sonnet-4-5-20250929"),
    # OpenAI
    "gpt-5.2": ("openai", "gpt-5.2"),
    "gpt-5.1": ("openai", "gpt-5.1"),
    "gpt-4o": ("openai", "gpt-4o"),
    # Google
    "gemini-3-pro": ("google", "gemini-3-pro"),
    "gemini-3-flash": ("google", "gemini-3-flash"),
    "gemini-2.5-pro": ("google", "gemini-2.5-pro"),
    "gemini-2.5-flash": ("google", "gemini-2.5-flash"),
}


def get_provider(provider_name: str) -> BaseAPIProvider:
    """Get a provider instance by name.

    Args:
        provider_name: Provider identifier ('anthropic', 'openai', 'google').

    Returns:
        Provider instance.

    Raises:
        ValueError: If provider_name is not recognized.

    Example:
        >>> provider = get_provider("anthropic")
        >>> provider.name
        'anthropic'
    """
    provider_class = PROVIDERS.get(provider_name.lower())
    if provider_class is None:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )
    return provider_class()


def resolve_model_alias(alias: str) -> tuple[str, str]:
    """Resolve a model alias to (provider, model_id).

    Args:
        alias: Model alias (e.g., 'claude-opus-4.5').

    Returns:
        Tuple of (provider_name, model_id).

    Raises:
        ValueError: If alias is not recognized.

    Example:
        >>> resolve_model_alias("claude-opus-4.5")
        ('anthropic', 'claude-opus-4-5-20251101')
    """
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]

    raise ValueError(
        f"Unknown model alias: {alias}. "
        f"Available: {', '.join(MODEL_ALIASES.keys())}"
    )
