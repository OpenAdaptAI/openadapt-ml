"""API Provider implementations for VLM backends.

This module provides a unified interface for different API providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)

The provider abstraction allows switching between different VLM backends
without changing the calling code. Each provider handles:
- Client creation with API key management
- Message sending with vision support
- Image encoding in provider-specific formats

Usage:
    from openadapt_ml.models.providers import get_provider

    # Get a provider and send a message
    provider = get_provider("anthropic")
    client = provider.create_client(api_key)
    response = provider.send_message(
        client,
        model="claude-opus-4-5-20251101",
        system="You are a GUI agent.",
        content=provider.build_content(
            text="Click the submit button",
            image=screenshot,
        ),
    )

    # Or use the quick_message helper
    response = provider.quick_message(
        api_key=key,
        model="claude-opus-4-5-20251101",
        prompt="What's in this image?",
        image=screenshot,
    )

Model Aliases:
    Common model aliases are provided for convenience:
    - "claude-opus-4.5" -> ("anthropic", "claude-opus-4-5-20251101")
    - "gpt-5.2" -> ("openai", "gpt-5.2")
    - "gemini-3-pro" -> ("google", "gemini-3-pro")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openadapt_ml.models.providers.base import (
    BaseAPIProvider,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
)
from openadapt_ml.models.providers.anthropic import AnthropicProvider
from openadapt_ml.models.providers.openai import OpenAIProvider
from openadapt_ml.models.providers.google import GoogleProvider

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    # Base classes and exceptions
    "BaseAPIProvider",
    "ProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    # Provider implementations
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    # Factory functions
    "get_provider",
    "get_provider_for_model",
    "resolve_model_alias",
    # Registries
    "PROVIDERS",
    "MODEL_ALIASES",
    # Convenience functions
    "quick_message",
    "list_providers",
    "list_models",
]

# Provider registry
PROVIDERS: dict[str, type[BaseAPIProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

# Model aliases for convenience
# Maps friendly names to (provider, model_id) tuples
MODEL_ALIASES: dict[str, tuple[str, str]] = {
    # Anthropic
    "claude-opus-4.5": ("anthropic", "claude-opus-4-5-20251101"),
    "claude-sonnet-4.5": ("anthropic", "claude-sonnet-4-5-20250929"),
    "claude-haiku-3.5": ("anthropic", "claude-haiku-3-5-20241022"),
    # OpenAI
    "gpt-5.2": ("openai", "gpt-5.2"),
    "gpt-5.1": ("openai", "gpt-5.1"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
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
        raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
    return provider_class()


def resolve_model_alias(alias: str) -> tuple[str, str]:
    """Resolve a model alias to (provider, model_id).

    Args:
        alias: Model alias (e.g., 'claude-opus-4.5') or full model ID.

    Returns:
        Tuple of (provider_name, model_id).

    Raises:
        ValueError: If alias is not recognized and can't be inferred.

    Example:
        >>> resolve_model_alias("claude-opus-4.5")
        ('anthropic', 'claude-opus-4-5-20251101')
        >>> resolve_model_alias("gemini-3-pro")
        ('google', 'gemini-3-pro')
    """
    # Check explicit aliases first
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]

    # Try to infer provider from model name patterns
    alias_lower = alias.lower()

    if alias_lower.startswith("claude"):
        return ("anthropic", alias)
    elif alias_lower.startswith("gpt"):
        return ("openai", alias)
    elif alias_lower.startswith("gemini"):
        return ("google", alias)

    raise ValueError(
        f"Unknown model alias: {alias}. "
        f"Available aliases: {', '.join(MODEL_ALIASES.keys())}. "
        f"Or use a full model ID with a known prefix (claude-*, gpt-*, gemini-*)."
    )


def get_provider_for_model(model: str) -> tuple[BaseAPIProvider, str]:
    """Get the appropriate provider for a model.

    Args:
        model: Model alias or full model ID.

    Returns:
        Tuple of (provider_instance, resolved_model_id).

    Example:
        >>> provider, model_id = get_provider_for_model("claude-opus-4.5")
        >>> provider.name
        'anthropic'
        >>> model_id
        'claude-opus-4-5-20251101'
    """
    provider_name, model_id = resolve_model_alias(model)
    provider = get_provider(provider_name)
    return provider, model_id


def quick_message(
    model: str,
    prompt: str,
    image: "Image | None" = None,
    system: str = "",
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Send a quick message to any model.

    Convenience function that resolves the provider, creates a client,
    and sends a message in one call. Useful for one-off requests.

    Args:
        model: Model alias or full model ID.
        prompt: User prompt text.
        image: Optional image to include.
        system: Optional system prompt.
        api_key: Optional API key (uses settings/env if not provided).
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.

    Returns:
        Model response text.

    Raises:
        AuthenticationError: If no API key is available.
        ProviderError: For API errors.

    Example:
        >>> response = quick_message(
        ...     model="claude-opus-4.5",
        ...     prompt="What's in this image?",
        ...     image=screenshot,
        ... )
    """
    provider, model_id = get_provider_for_model(model)
    resolved_key = provider.get_api_key(api_key)
    return provider.quick_message(
        api_key=resolved_key,
        model=model_id,
        prompt=prompt,
        image=image,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def list_providers() -> list[str]:
    """List available provider names.

    Returns:
        List of provider identifiers.

    Example:
        >>> list_providers()
        ['anthropic', 'openai', 'google']
    """
    return list(PROVIDERS.keys())


def list_models(provider: str | None = None) -> dict[str, dict]:
    """List available models, optionally filtered by provider.

    Args:
        provider: Optional provider name to filter by.

    Returns:
        Dict mapping model IDs to their properties.

    Example:
        >>> list_models("anthropic")
        {
            'claude-opus-4-5-20251101': {'context': 200000, 'description': 'SOTA computer use'},
            ...
        }
    """
    if provider:
        provider_instance = get_provider(provider)
        return provider_instance.supported_models

    # Combine models from all providers
    all_models = {}
    for provider_name in PROVIDERS:
        provider_instance = get_provider(provider_name)
        for model_id, props in provider_instance.supported_models.items():
            all_models[model_id] = {**props, "provider": provider_name}

    return all_models
