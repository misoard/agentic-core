"""agents/tool_loop.py — the client-side tool-calling loop.

Why this module exists
----------------------
The gateway is a *single* call: it can now *report* that the model asked for a
tool (``Completion.tool_calls``), but it never executes anything. Client-side
tools need a *loop* — call the model, run the tool it asked for, feed the result
back, call again — until the model returns a final answer. That loop is more than
one call, so it lives here (the agent layer), above the gateway, never inside it.

The mechanism/policy split
--------------------------
This module is the generic *engine*: it knows *how* to run a tool cycle but
nothing about *which* tools exist. A consumer defines concrete ``Tool``\\s in its
own repo (name + typed arg schema + function) and passes them in — exactly the
dependency-injection shape of ``RouterLike`` and ``Step``.

Safety
------
Tool arguments are proposed by the model and therefore UNTRUSTED. Every call's
args are validated (``guardrails.validate_tool_args``) before the function runs;
a bad shape is fed back to the model as an error result so it can correct, rather
than crashing the loop — the same "honest correction" idea as the gateway's
re-prompt.

Interview line: "The model never executes anything — it emits a request. The
loop runs the tool client-side behind a strict typed boundary, then feeds the
result back. Tool args are untrusted input, so they're Pydantic-validated before
they ever reach code."
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..core.errors import AgenticError
from ..core.gateway import Gateway, Messages
from ..core.schemas import Completion
from ..guardrails.io_guards import GuardrailError, validate_tool_args


class ToolLoopError(AgenticError):
    """The tool loop ran its iteration budget without the model settling on an
    answer — a runaway/looping model, surfaced instead of spinning forever."""


@dataclass
class Tool:
    """A client-side tool: a typed arg contract + the function that runs it.

    ``fn`` receives an *already-validated* arg model instance (so its body can
    trust the fields) and may be sync or async. ``arg_model`` is both the
    validation contract and the schema advertised to the provider.
    """

    name: str
    description: str
    arg_model: type[BaseModel]
    fn: Callable[[BaseModel], Any | Awaitable[Any]]

    def spec(self) -> dict[str, Any]:
        """The provider-facing tool schema (OpenAI/LiteLLM 'function' shape)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arg_model.model_json_schema(),
            },
        }

    async def invoke(self, args: BaseModel) -> Any:
        result = self.fn(args)
        if inspect.isawaitable(result):  # support both sync and async tools
            result = await result
        return result


def _assistant_tool_message(completion: Completion) -> dict[str, Any]:
    """Reconstruct the assistant turn that requested the tools, so the next call
    shows the model its own request (providers require it before tool results)."""
    return {
        "role": "assistant",
        "content": completion.text or None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in completion.tool_calls
        ],
    }


def _tool_result_message(call_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "tool", "tool_call_id": call_id, "content": content}
    if is_error:
        msg["is_error"] = True
    return msg


async def run_with_tools(
    gateway: Gateway,
    messages: Messages,
    tools: list[Tool],
    *,
    model: str | None = None,
    max_iterations: int = 8,
    **opts: Any,
) -> Completion:
    """Drive the call -> execute -> feed-back loop until a final answer.

    Each turn: call the gateway with the tool specs; if the model returned no
    tool calls, that's the final answer -> return it. Otherwise validate + run
    each requested tool client-side, append the results, and call again. Bounded
    by ``max_iterations`` so a model that never stops calling tools fails loudly.
    """
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    convo: Messages = list(messages)  # our own copy; we grow it across turns

    for _ in range(max_iterations):
        completion = await gateway.complete(convo, model=model, tools=specs, **opts)

        if not completion.tool_calls:
            return completion  # no tool request -> the model's final answer

        # Show the model its own tool-call request before the results.
        convo.append(_assistant_tool_message(completion))

        for call in completion.tool_calls:
            tool = registry.get(call.name)
            if tool is None:
                # Model hallucinated a tool we don't have -> tell it, don't crash.
                convo.append(_tool_result_message(
                    call.id, f"error: unknown tool {call.name!r}", is_error=True))
                continue
            try:
                # UNTRUSTED args -> validate at the boundary before executing.
                args = validate_tool_args(tool.arg_model, call.arguments)
                result = await tool.invoke(args)
                convo.append(_tool_result_message(call.id, str(result)))
            except GuardrailError as exc:
                # Bad args: feed the validation error back so the model can fix it.
                convo.append(_tool_result_message(
                    call.id, f"invalid arguments: {exc}", is_error=True))

    raise ToolLoopError(
        f"tool loop did not converge within {max_iterations} iterations"
    )
