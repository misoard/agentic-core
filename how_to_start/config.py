"""demo/config.py — the CONSUMER's configuration (policy, not machinery).

This is the counterpart to the package's core/config.py. The package owns the
*shape* of config (``Settings``, ``Deployment``, ``build_router_kwargs``); this
file owns the *values* for this project:

  * ``AppSettings`` — extends the base Settings with project-specific fields
    (here: model-name overrides). It inherits the base's ``.env`` loading, so it
    reads the repo-root ``.env`` (cwd) — which is also what it'll do naturally
    when demo/ becomes its own repo and that .env sits at its root.
  * ``build_registry`` / ``FALLBACKS`` — the real alias table, built from settings
    so a model swap is an env change (FAST_MODEL=...), not a code edit.
  * ``build_gateway`` — wires it all into a Gateway. The workflow (added later)
    imports this and never touches the package's scaffolding defaults.

In a real project this file lives in your repo; agentic_core is an installed dep.
"""

from __future__ import annotations

from agentic_core import Deployment, Gateway, Settings


class AppSettings(Settings):
    """Project settings = base Settings + this project's extra knobs.

    Model names are env-driven (FAST_MODEL / SMART_MODEL) so swapping a model
    across dev/staging/prod is a pure env change. Params (temperature, rpm) stay
    in code (see build_registry) — env is for what varies per environment, code
    for what's structural. Inherits the base Settings' ``.env`` loading.
    """

    fast_model: str = "openrouter/openai/gpt-4o-mini"
    smart_model: str = "openrouter/anthropic/claude-sonnet-4.6"

    # To add a provider, just declare its key here — the package resolves it by
    # the `<provider>_api_key` convention, no core change:
    #   groq_api_key: str | None = None


def build_registry(settings: AppSettings) -> dict[str, Deployment]:
    """The real alias table for this project (slug from env, params from code)."""
    return {
        "fast": Deployment(
            alias="fast", model=settings.fast_model, params={"temperature": 0.0}
        ),
        "smart": Deployment(
            alias="smart", model=settings.smart_model, params={"temperature": 0.0}
        ),
    }


# This project's resilience graph: smart degrades to fast.
FALLBACKS: dict[str, list[str]] = {"smart": ["fast"]}


def build_gateway(settings: AppSettings | None = None) -> Gateway:
    """Construct the Gateway this project runs on — the one call path for the app."""
    settings = settings or AppSettings()
    return Gateway(
        settings=settings,
        registry=build_registry(settings),
        fallbacks=FALLBACKS,
    )
