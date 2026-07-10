"""schemas — the typed I/O contracts that cross every boundary.

Why this module exists
----------------------
If an LLM's output feeds *code* rather than a human, every unvalidated field is a
latent incident. So failures are caught at boundaries: each agent gets a typed
output contract the gateway enforces, and downstream code only ever receives
validated data.

This file holds the two *shared carrier* types that every layer passes around:

  * ``TokenUsage`` — prompt/completion/total token counts for one call.
  * ``Completion`` — the gateway's return value: the raw text, the (optionally)
    parsed-and-validated Pydantic model, plus the cost/latency/usage metadata
    that observability and config-sweeps need.

Concrete *agent* contracts (a given agent's input and output models) live next to
their agents, because they're project-specific; they're plain ``pydantic.BaseModel``
subclasses validated by the gateway. The carriers here are deliberately
dataclasses, not BaseModels: they wrap an already-validated payload and shouldn't
re-run validation (or try to coerce the arbitrary ``parsed`` model) on the hot path.

Interview line: "If an LLM's output feeds code rather than a human, every
unvalidated field is a latent incident — so each agent has a typed output
contract the gateway enforces."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

# The validated output model an agent expects back, when it asks for one.
T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single model call. Drives cost and budget math."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool the model asked to call. ``arguments`` is raw, model-proposed, and
    therefore UNTRUSTED — validate it (guardrails.validate_tool_args) before use.

    Carried on ``Completion`` so the single call path can *report* a tool request;
    executing it and looping is the agent layer's job, never the gateway's.
    """

    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Completion(Generic[T]):
    """Everything one trip through the gateway produced.

    Generic over ``T`` so a typed call (``response_model=Foo``) hands back a
    ``Completion[Foo]`` and ``.parsed`` is statically known to be ``Foo | None``.
    Frozen because a completed call is a fact, not mutable state.
    """

    # The raw assistant text, always present — the source of truth for everything else.
    text: str
    # Which concrete provider/model actually served the call (post-fallback).
    model: str
    # The validated structured output, or None when the caller asked for free text.
    parsed: T | None = None
    # Cost/latency/usage: the three things observability and config-sweeps exist to read.
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    # How many *re-prompts* the gateway needed to get schema-valid output (0 == first try).
    # Transient network retries are the Router's and are deliberately not counted here.
    reprompt_attempts: int = 0
    # Tools the model asked to call this turn. Empty for an ordinary answer; when
    # non-empty the agent-layer tool loop executes them and calls again.
    tool_calls: tuple[ToolCall, ...] = ()
    # Free-form extras (provider id, finish_reason, ...) without widening the contract.
    extra: dict = field(default_factory=dict)
