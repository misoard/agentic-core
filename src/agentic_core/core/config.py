"""config — models and secrets as data, not code.

Why this module exists
----------------------
Swapping a cheap model for an expensive one should be a *config change, not a
rewrite*. So models live in a registry keyed by stable aliases your code uses
("fast", "smart") that map to concrete provider/model/params. Business logic
says ``model="fast"``; which provider that is, and what it costs, is decided
here. That's what makes a cost/accuracy sweep cheap: you change an alias, not code.

What it owns
------------
  * ``Settings`` — secrets + run-wide defaults, loaded from env / ``.env`` via
    pydantic-settings. Secrets flow through here, not scattered ``os.getenv`` calls.
  * ``Deployment`` — one concrete model behind an alias (a Router "deployment").
  * ``DEFAULT_REGISTRY`` / ``DEFAULT_FALLBACKS`` — the editable alias table.
  * ``build_router_kwargs`` — turns registry + settings into the exact dict you
    splat into ``litellm.Router(**kwargs)``. This is the one place config meets
    the gateway's underlying Router.
  * ``effective_params`` — documents call-time precedence: per-call overrides win
    over a deployment's registry defaults.

Interview line: "My cost/accuracy benchmark was cheap to run because models are
config, not code — I could sweep candidates by changing an alias."
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Run-wide secrets and defaults, read from environment / ``.env``.

    Field names map to UPPER_SNAKE env vars automatically (``openai_api_key`` ->
    ``OPENAI_API_KEY``). Keeping keys here — instead of letting LiteLLM read
    ambient env behind our back — means the secret's source of truth is one typed
    object we can inspect and test.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Provider secrets. Optional so the suite runs offline with none of them set.
    # OpenRouter is the default: one key fronts ~any model, so an alias can point
    # anywhere without adding a new provider credential.
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # The alias business logic asks for when it doesn't name a model explicitly.
    default_model: str = "fast"

    # Router-level reliability knobs. These configure the *Router's* transient
    # retry domain (point-4: transient retries are the Router's job, never the
    # gateway's). num_retries is how many times the Router re-attempts a transient
    # failure before moving to a fallback / giving up.
    request_timeout_s: float = 30.0
    num_retries: int = 2

    # Gate: real provider calls in tests only run when this is true, so
    # `uv run pytest` is fully offline by default.
    llm_live_tests: bool = False

    # Observability is opt-in. With neither set, OTel's default tracer is a
    # no-op — spans cost nothing and never touch the network (so tests stay
    # offline). Set a token to ship traces to Logfire; the console flag prints
    # spans locally for dev without any account.
    logfire_token: str | None = None
    otel_console_export: bool = False
    # OTLP/HTTP endpoint of a Jaeger collector, e.g.
    # "http://localhost:4318/v1/traces". Jaeger ingests OTLP natively, so no
    # Jaeger-specific exporter is needed — just point OTLP at it.
    jaeger_endpoint: str | None = None


@lru_cache
def get_settings() -> Settings:
    """One settings load per process (the config singleton)."""
    return Settings()


class Deployment(BaseModel):
    """One concrete model sitting behind an alias — a Router "deployment".

    Several deployments can share one alias to let the Router load-balance across
    them; here we keep it one-to-one for legibility and let *fallbacks* (a
    different alias) handle resilience.
    """

    alias: str  # the stable name code uses, e.g. "fast"
    model: str  # the LiteLLM model string, e.g. "openai/gpt-4o-mini"
    # Default sampling params (temperature, max_tokens, ...). Kept as a plain dict
    # rather than an over-typed model — these are provider-passthrough and vary.
    params: dict[str, Any] = Field(default_factory=dict)
    # Optional per-deployment rate caps the Router enforces.
    rpm: int | None = None
    tpm: int | None = None


# ILLUSTRATIVE SCAFFOLDING, not the real registry. It exists so the core is
# runnable standalone (tests, the demo, zero-config experiments). A real consumer
# defines its own registry in its own repo and passes it via `registry=` (see
# demo/ for the pattern) — this default is the fallback when they don't.
# Model strings are OpenRouter slugs: `openrouter/<vendor>/<model>`.
DEFAULT_REGISTRY: dict[str, Deployment] = {
    "fast": Deployment(
        alias="fast",
        model="openrouter/openai/gpt-4o-mini",
        params={"temperature": 0.0},
    ),
    "smart": Deployment(
        alias="smart",
        model="openrouter/anthropic/claude-sonnet-4.6",
        params={"temperature": 0.0},
    ),
}

#To check whether a model is supported by LiteLLM:
#import litellm
#"openrouter" in [p.value for p in litellm.provider_list]   # True


# Illustrative scaffolding too (see DEFAULT_REGISTRY note). alias -> ordered
# fallback aliases; the Router moves to the next only after exhausting
# `num_retries` transient attempts on the current model.
DEFAULT_FALLBACKS: dict[str, list[str]] = {
    "smart": ["fast"],
}


def _api_key_for(model: str, settings: Settings) -> str | None:
    """Resolve the API key for a ``provider/model`` string from Settings.

    Convention over configuration: provider ``X`` -> Settings field ``X_api_key``.
    This is pure mechanism — it hardcodes NO provider names. A consumer who adds a
    new provider just declares ``<provider>_api_key`` on a Settings subclass and
    routing to ``<provider>/...`` finds the key with zero package change. (Assumes
    the prefix is a valid identifier; via OpenRouter the prefix is always
    ``openrouter``, so the default path is unaffected regardless.)
    """
    provider = model.split("/", 1)[0] if "/" in model else ""
    return getattr(settings, f"{provider}_api_key", None) if provider else None


def build_router_kwargs(
    settings: Settings | None = None,
    *,
    registry: dict[str, Deployment] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Compile registry + settings into ``litellm.Router(**kwargs)`` input.

    This is the single seam between our config and the Router. It builds the
    ``model_list`` (injecting the right key per deployment) and attaches the
    Router-level reliability settings. The gateway calls this once at startup.
    """
    settings = settings or get_settings()
    registry = registry if registry is not None else DEFAULT_REGISTRY
    fallbacks = fallbacks if fallbacks is not None else DEFAULT_FALLBACKS

    model_list: list[dict[str, Any]] = []
    for dep in registry.values():
        litellm_params: dict[str, Any] = {"model": dep.model, **dep.params}
        key = _api_key_for(dep.model, settings)
        if key:
            litellm_params["api_key"] = key
        if dep.rpm is not None:
            litellm_params["rpm"] = dep.rpm
        if dep.tpm is not None:
            litellm_params["tpm"] = dep.tpm
        model_list.append({"model_name": dep.alias, "litellm_params": litellm_params})

    # LiteLLM wants fallbacks as a list of single-key dicts: [{"smart": ["fast"]}].
    router_fallbacks = [{alias: targets} for alias, targets in fallbacks.items()]

    return {
        "model_list": model_list,
        "num_retries": settings.num_retries,
        "timeout": settings.request_timeout_s,
        "fallbacks": router_fallbacks,
    }


def effective_params(
    alias: str,
    *,
    registry: dict[str, Deployment] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """The params a call actually runs with: registry defaults, then overrides.

    Documents precedence in one place — a per-call override always wins over the
    deployment's default. (At call time the gateway also passes overrides straight
    to ``router.acompletion``, where LiteLLM applies the same precedence; this
    helper makes the rule explicit and testable.)
    """
    registry = registry if registry is not None else DEFAULT_REGISTRY
    if alias not in registry:
        raise KeyError(f"Unknown model alias {alias!r}. Known: {sorted(registry)}")
    return {**registry[alias].params, **overrides}
