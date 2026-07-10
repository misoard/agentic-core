"""Tests for the client-side tool loop — fully offline via a scripted FakeRouter.

Proves the cycle: model asks for a tool -> loop validates args + runs it ->
feeds the result back -> model returns a final answer.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agentic_core.agents.tool_loop import Tool, ToolLoopError, run_with_tools
from agentic_core.core.errors import GuardrailError
from agentic_core.core.gateway import Gateway
from conftest import FakeRouter, make_response


class WeatherArgs(BaseModel):
    city: str


def get_weather(args: WeatherArgs) -> str:
    return f"18C in {args.city}"


def _weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Current weather for a city.",
        arg_model=WeatherArgs,
        fn=get_weather,
    )


def make_tool_response(name: str, arguments: dict, *, call_id: str = "call_1"):
    """A FakeRouter response whose message carries a tool call (OpenAI shape)."""
    fn = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    tc = SimpleNamespace(id=call_id, function=fn)
    message = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        model="openrouter/openai/gpt-4o-mini",
        _hidden_params={"response_cost": 0.0001},
    )


async def test_tool_call_then_final_answer(offline_settings):
    # turn 1: model asks for get_weather; turn 2: model gives the final answer.
    router = FakeRouter([
        make_tool_response("get_weather", {"city": "Paris"}),
        make_response("It's 18C in Paris."),
    ])
    gw = Gateway(router=router, settings=offline_settings)

    out = await run_with_tools(
        gw,
        [{"role": "user", "content": "weather in Paris?"}],
        [_weather_tool()],
    )

    assert out.text == "It's 18C in Paris."
    assert len(router.calls) == 2  # one tool round-trip, then the final answer

    # The 2nd call must carry the assistant tool-call request + the tool result.
    second_msgs = router.calls[1]["messages"]
    roles = [m["role"] for m in second_msgs]
    assert "assistant" in roles and "tool" in roles
    tool_msg = next(m for m in second_msgs if m["role"] == "tool")
    assert tool_msg["content"] == "18C in Paris"          # our function actually ran


async def test_bad_tool_args_are_fed_back_not_crash(offline_settings):
    # Model proposes args that fail validation (missing 'city'), then recovers.
    router = FakeRouter([
        make_tool_response("get_weather", {"wrong": "field"}),
        make_response("done"),
    ])
    gw = Gateway(router=router, settings=offline_settings)

    out = await run_with_tools(
        gw, [{"role": "user", "content": "?"}], [_weather_tool()]
    )

    assert out.text == "done"
    # The invalid-args error was fed back to the model, not raised.
    tool_msg = next(m for m in router.calls[1]["messages"] if m["role"] == "tool")
    assert "invalid arguments" in tool_msg["content"]
    assert tool_msg.get("is_error") is True


async def test_unknown_tool_is_reported_not_crash(offline_settings):
    router = FakeRouter([
        make_tool_response("nonexistent", {}),
        make_response("ok"),
    ])
    gw = Gateway(router=router, settings=offline_settings)

    out = await run_with_tools(gw, [{"role": "user", "content": "?"}], [_weather_tool()])
    assert out.text == "ok"
    tool_msg = next(m for m in router.calls[1]["messages"] if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


async def test_runaway_loop_raises(offline_settings):
    # Model keeps asking for tools forever -> bounded, fails loudly.
    router = FakeRouter([make_tool_response("get_weather", {"city": "X"}) for _ in range(10)])
    gw = Gateway(router=router, settings=offline_settings)

    with pytest.raises(ToolLoopError):
        await run_with_tools(
            gw, [{"role": "user", "content": "?"}], [_weather_tool()], max_iterations=3
        )
    assert len(router.calls) == 3  # stopped at the budget
