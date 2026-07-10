"""Tests for the model registry, settings, and Router-kwargs compilation.

All offline: we build the kwargs dict but never construct a real Router or call a
provider. Settings are built with explicit values so ambient env can't leak in.
"""

from __future__ import annotations

import pytest

from agentic_core.core.config import (
    DEFAULT_REGISTRY,
    Deployment,
    Settings,
    build_router_kwargs,
    effective_params,
)


def _settings(**kw):
    # _env_file=None so the test never reads a real .env on the machine.
    base = dict(
        openrouter_api_key="sk-openrouter",
        openai_api_key="sk-openai",
        anthropic_api_key="sk-anthropic",
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_settings_have_offline_safe_defaults():
    s = Settings(_env_file=None)
    assert s.default_model == "fast"
    assert s.llm_live_tests is False
    assert s.openrouter_api_key is None  # nothing required to import/run offline


def test_registry_aliases_decouple_code_from_provider():
    assert "fast" in DEFAULT_REGISTRY
    # Default registry fronts everything through OpenRouter (one key, any model).
    assert DEFAULT_REGISTRY["fast"].model.startswith("openrouter/")


def test_build_router_kwargs_shapes_model_list_for_litellm():
    kwargs = build_router_kwargs(_settings())
    names = {d["model_name"] for d in kwargs["model_list"]}
    assert {"fast", "smart"} <= names

    fast = next(d for d in kwargs["model_list"] if d["model_name"] == "fast")
    assert fast["litellm_params"]["model"] == "openrouter/openai/gpt-4o-mini"
    # The key is injected from Settings, not read from ambient env by LiteLLM.
    assert fast["litellm_params"]["api_key"] == "sk-openrouter"
    # Registry default params flow into litellm_params.
    assert fast["litellm_params"]["temperature"] == 0.0


def test_build_router_kwargs_injects_key_by_provider():
    kwargs = build_router_kwargs(_settings())
    # Both default deployments front through OpenRouter, so both get that key.
    smart = next(d for d in kwargs["model_list"] if d["model_name"] == "smart")
    assert smart["litellm_params"]["api_key"] == "sk-openrouter"


def test_build_router_kwargs_carries_reliability_settings():
    kwargs = build_router_kwargs(_settings(num_retries=5, request_timeout_s=12.0))
    # These configure the Router's transient-retry domain — not the gateway's.
    assert kwargs["num_retries"] == 5
    assert kwargs["timeout"] == 12.0
    # Fallbacks compiled into LiteLLM's list-of-single-key-dicts shape.
    assert {"smart": ["fast"]} in kwargs["fallbacks"]


def test_missing_key_is_simply_omitted():
    # No openrouter key -> the deployments just have no api_key field.
    s = _settings(openrouter_api_key=None)
    kwargs = build_router_kwargs(s)
    smart = next(d for d in kwargs["model_list"] if d["model_name"] == "smart")
    assert "api_key" not in smart["litellm_params"]


def test_effective_params_overrides_win_over_defaults():
    merged = effective_params("fast", temperature=0.7, max_tokens=256)
    assert merged["temperature"] == 0.7  # override beats registry default of 0.0
    assert merged["max_tokens"] == 256


def test_effective_params_rejects_unknown_alias():
    with pytest.raises(KeyError):
        effective_params("does-not-exist")


def test_key_resolution_follows_provider_convention():
    # A consumer adds a new provider by subclassing Settings with `<provider>_api_key`.
    # The package resolves it by convention — no package change, no hardcoded map.
    class AppSettings(Settings):
        groq_api_key: str | None = None

    s = AppSettings(_env_file=None, groq_api_key="sk-groq")
    reg = {"g": Deployment(alias="g", model="groq/llama-3.1-8b")}
    kwargs = build_router_kwargs(s, registry=reg, fallbacks={})
    assert kwargs["model_list"][0]["litellm_params"]["api_key"] == "sk-groq"


def test_custom_registry_is_honoured():
    reg = {"tiny": Deployment(alias="tiny", model="openai/gpt-4o-mini")}
    kwargs = build_router_kwargs(_settings(), registry=reg, fallbacks={})
    assert [d["model_name"] for d in kwargs["model_list"]] == ["tiny"]
    assert kwargs["fallbacks"] == []
