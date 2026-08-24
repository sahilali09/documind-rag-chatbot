from __future__ import annotations

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

import config
from src.exceptions import ProviderConfigurationError


class MissingAPIKeyError(ProviderConfigurationError):
    """Raised when a provider requires an API key that is unavailable."""


def _get_override(
    overrides: dict[str, Any],
    name: str,
    default: str,
) -> str:
    value = overrides.get(name, default)

    if value is None:
        value = default

    value = str(value).strip()

    if not value:
        raise ProviderConfigurationError(
            f"{name} must not be empty."
        )

    return value


def get_llm(
    provider: str | None = None,
    **overrides: Any,
) -> BaseChatModel:
    """Create a configured chat model without persisting runtime secrets."""
    selected_provider = (
        provider or config.LLM_PROVIDER
    ).strip().lower()

    if selected_provider not in config.VALID_LLM_PROVIDERS:
        raise ProviderConfigurationError(
            f"Unknown LLM provider '{selected_provider}'. "
            f"Use one of {sorted(config.VALID_LLM_PROVIDERS)}."
        )

    try:
        if selected_provider == "groq":
            from langchain_groq import ChatGroq

            api_key = (
                overrides.get("api_key")
                or _require_env("GROQ_API_KEY", selected_provider)
            )

            return ChatGroq(
                model=_get_override(
                    overrides,
                    "model",
                    config.GROQ_MODEL,
                ),
                temperature=config.LLM_TEMPERATURE,
                api_key=api_key,
            )

        if selected_provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = (
                overrides.get("api_key")
                or _require_env("OPENAI_API_KEY", selected_provider)
            )

            return ChatOpenAI(
                model=_get_override(
                    overrides,
                    "model",
                    config.OPENAI_MODEL,
                ),
                temperature=config.LLM_TEMPERATURE,
                api_key=api_key,
            )

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=_get_override(
                overrides,
                "model",
                config.OLLAMA_MODEL,
            ),
            base_url=_get_override(
                overrides,
                "base_url",
                config.OLLAMA_BASE_URL,
            ),
            temperature=config.LLM_TEMPERATURE,
        )

    except ProviderConfigurationError:
        raise
    except ImportError as exc:
        raise ProviderConfigurationError(
            f"Required package for LLM provider "
            f"'{selected_provider}' is not installed: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderConfigurationError(
            f"Failed to initialize LLM provider "
            f"'{selected_provider}': {exc}"
        ) from exc


def get_embeddings(
    provider: str | None = None,
    **overrides: Any,
) -> Embeddings:
    """Create a configured embedding model."""
    selected_provider = (
        provider or config.EMBEDDING_PROVIDER
    ).strip().lower()

    if selected_provider not in config.VALID_EMBEDDING_PROVIDERS:
        raise ProviderConfigurationError(
            f"Unknown embedding provider '{selected_provider}'. "
            f"Use one of {sorted(config.VALID_EMBEDDING_PROVIDERS)}."
        )

    try:
        if selected_provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=_get_override(
                    overrides,
                    "model",
                    config.HUGGINGFACE_EMBEDDING_MODEL,
                ),
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

        if selected_provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = (
                overrides.get("api_key")
                or _require_env(
                    "OPENAI_API_KEY",
                    selected_provider,
                )
            )

            return OpenAIEmbeddings(
                model=_get_override(
                    overrides,
                    "model",
                    config.OPENAI_EMBEDDING_MODEL,
                ),
                api_key=api_key,
            )

        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=_get_override(
                overrides,
                "model",
                config.OLLAMA_EMBEDDING_MODEL,
            ),
            base_url=_get_override(
                overrides,
                "base_url",
                config.OLLAMA_BASE_URL,
            ),
        )

    except ProviderConfigurationError:
        raise
    except ImportError as exc:
        raise ProviderConfigurationError(
            f"Required package for embedding provider "
            f"'{selected_provider}' is not installed: {exc}"
        ) from exc
    except Exception as exc:
        raise ProviderConfigurationError(
            f"Failed to initialize embedding provider "
            f"'{selected_provider}': {exc}"
        ) from exc


def _require_env(
    var_name: str,
    provider: str,
) -> str:
    value = os.getenv(var_name)

    if value is None or not value.strip():
        raise MissingAPIKeyError(
            f"{var_name} is not configured, but provider "
            f"'{provider}' requires it."
        )

    return value.strip()