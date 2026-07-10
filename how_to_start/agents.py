"""demo/agents.py — concrete agents for the demo (the CONSUMER side).

This is the "policy" half of the split: the package owns the ``Agent`` machinery;
here we define which agents exist, their typed contracts, and (loaded from
``prompts/``) what they say. In a real project this file would live in your repo,
importing agentic_core as an installed dependency — exactly as it does here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from agentic_core import Agent, Gateway

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptSpec(BaseModel):
    """Validated contract for a prompt file — fail fast on a malformed YAML.

    A prompt file is untrusted input like any other config, so it gets the same
    boundary validation as ``Settings``/``Deployment``: a missing key or wrong
    type raises a ``ValidationError`` at load (naming the field), not a ``KeyError``
    later during agent construction.
    """

    system: str
    user_template: str
    # SPEC frames prompts as versioned config; default keeps version-less files valid.
    version: int = 1


def load_prompt(name: str) -> PromptSpec:
    """Load a versioned prompt (system + user_template) from ``prompts/<name>.yaml``.

    A tiny loader on purpose: prompts-as-files is a consumer concern, so it lives
    here rather than in the package. Promote it to the core only if several
    projects end up wanting the exact same loader.
    """
    data = yaml.safe_load((_PROMPTS_DIR / f"{name}.yaml").read_text())
    return PromptSpec.model_validate(data)


class TextIn(BaseModel):
    """Input contract: just the text to classify."""

    text: str


class Sentiment(BaseModel):
    """Output contract: a label plus a calibrated confidence.

    ``confidence`` is bounded [0, 1] — an out-of-range value is exactly the kind
    of malformed output the gateway catches and re-prompts on.
    """

    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


def sentiment_agent(gateway: Gateway, *, model: str | None = None) -> Agent[TextIn, Sentiment]:
    """Build the sentiment agent: typed text in, typed classification out."""
    prompt = load_prompt("sentiment")
    return Agent(
        name="sentiment",
        gateway=gateway,
        input_model=TextIn,
        output_model=Sentiment,
        model=model,
        system_prompt=prompt.system,
        user_template=prompt.user_template,
    )
