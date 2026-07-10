"""agents/base.py — the agent abstraction.

Why this module exists
----------------------
Without this, every LLM task is bespoke glue: ad-hoc prompt strings, ad-hoc
parsing, ad-hoc validation. An ``Agent`` makes the unit uniform: a prompt
template + a typed input schema + a typed output schema, run through the one
gateway. That uniformity is what lets orchestration compose agents without caring
how any of them is implemented, and it's what makes each agent independently
testable.

The shape
---------
An agent is deliberately thin — it owns only three things:

  1. *How to turn typed input into messages* (``render_messages`` / the templates).
  2. *Which typed output it expects* (``output_model``, enforced by the gateway).
  3. *Which model alias it runs on* (``model``; None -> the gateway's default).

Everything hard — retry, fallback, validation, re-prompt, cost, tracing — is the
gateway's, not the agent's. An agent never talks to a provider; it only ever
calls ``gateway.complete``. That's the layering the whole repo is built on:
orchestration -> agent -> gateway -> router -> provider.

Tools are intentionally *not* here yet. Tool-calling adds an attack surface
(untrusted args flowing into code), so it belongs with the guardrails layer;
we'll add it there rather than bolt a half-guarded version on now.

Interview line: "An agent is just a typed, validated, single-responsibility LLM
call. Narrow agents beat one giant prompt for reliability — splitting a problem
across focused agents is the core agentic move."
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..core.gateway import Gateway, Messages
from ..core.schemas import Completion

# Typed input and output contracts for this agent.
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Agent(Generic[InputT, OutputT]):
    """A single, typed, validated LLM call. Configure one; call ``run``.

    Configure with templates for the common case, or subclass and override
    ``render_messages`` when a prompt needs more than string formatting.
    """

    def __init__(
        self,
        *,
        name: str,
        gateway: Gateway,
        output_model: type[OutputT],
        system_prompt: str,
        user_template: str,
        input_model: type[InputT] | None = None,
        model: str | None = None,
    ) -> None:
        self.name = name
        self.gateway = gateway
        self.output_model = output_model
        # input_model lets run() validate a raw dict at the boundary. If callers
        # always pass an already-typed model instance, it's optional.
        self.input_model = input_model
        self.system_prompt = system_prompt
        self.user_template = user_template
        # Pinning a model per agent is a design choice: a cheap agent and an
        # expensive one can coexist in one workflow, each on the right model.
        self.model = model

    def render_messages(
        self, inp: InputT, *, history: Messages | None = None
    ) -> Messages:
        """Typed input -> OpenAI-format messages. Override for richer prompts.

        Receives an already-validated model instance (run() coerces first), so
        overrides can assume clean, typed fields.

        ``history`` is prior conversation turns (assistant/user message dicts),
        threaded between the fixed system prompt and this turn's user message.
        The agent stays *stateless*: it renders whatever history it's given but
        remembers nothing across calls. A caller (or an orchestration session)
        owns the history list and passes it back each turn — the standard
        "resend the whole conversation" pattern, since the model has no memory.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            *(history or []),  # prior turns, if any — empty for a single-shot call
            {"role": "user", "content": self.user_template.format(**inp.model_dump())},
        ]

    async def run(
        self,
        inp: InputT | dict[str, Any],
        *,
        history: Messages | None = None,
        **opts: Any,
    ) -> Completion[OutputT]:
        """Validate input, render the prompt, and execute through the gateway.

        Returns the full ``Completion`` — ``.parsed`` is the typed output, and the
        cost/latency/trace metadata rides along so orchestration keeps visibility.

        ``history`` (optional) carries prior conversation turns for a multi-turn
        exchange; omit it for a single-shot call. Ownership stays with the caller
        — the agent doesn't retain it — keeping ``run`` a stateless unit that a
        session layer can drive repeatedly.
        """
        typed = self._coerce_input(inp)
        return await self.gateway.complete(
            self.render_messages(typed, history=history),
            model=self.model,
            response_model=self.output_model,
            **opts,
        )

    def _coerce_input(self, inp: InputT | dict[str, Any]) -> InputT:
        # Validate at the boundary: a dict from outside is checked against the
        # input contract before it ever shapes a prompt.
        if isinstance(inp, BaseModel):
            return inp  # already typed & validated by construction
        if self.input_model is not None:
            return self.input_model.model_validate(inp)
        raise TypeError(
            f"Agent {self.name!r} got a dict but has no input_model to validate it."
        )
