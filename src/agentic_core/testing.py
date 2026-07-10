"""testing — a *shipped* test seam for consumers of agentic_core.

Why this is in the package (not just tests/)
--------------------------------------------
Any repo built on this core needs to test its agents/workflows without hitting a
provider. The gateway is designed for exactly that: it depends on a ``RouterLike``
Protocol, so a fake Router is the injection point. Shipping that fake (plus a
response factory) means every consumer tests the same way we do — inject
``FakeRouter``, assert on the recorded calls — instead of reinventing it.

Import it in your tests:  ``from agentic_core.testing import FakeRouter, make_response``
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeRouter:
    """Stands in for ``litellm.Router``. Plays back a scripted list of behaviours.

    Each behaviour is either a response object (returned) or an Exception (raised),
    so a test can model "malformed then valid", "rate-limited", etc. Every call is
    recorded in ``.calls`` so tests can assert what messages the gateway sent (e.g.
    that a re-prompt carried the corrective feedback, or an alias was forwarded).
    """

    def __init__(self, behaviours: list):
        self._behaviours = list(behaviours)
        self.calls: list[dict] = []

    async def acompletion(self, *, model: str, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self._behaviours:
            raise AssertionError("FakeRouter ran out of scripted behaviours")
        behaviour = self._behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


def make_response(
    content: str,
    *,
    model: str = "openrouter/openai/gpt-4o-mini",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    cost: float | None = 0.00012,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for a LiteLLM ModelResponse.

    Mimics the attribute access the gateway uses (``.choices[0].message.content``,
    ``.usage.*``, ``.model``, ``._hidden_params``), nothing more.
    """
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model,
        _hidden_params={"response_cost": cost},
    )
