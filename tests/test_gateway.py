"""Tests for the keystone.

These pin the point-4 division of labour behaviourally:
  * malformed output -> the gateway re-prompts (its job), with honest feedback;
  * a provider/Router exception -> the gateway does NOT retry (the Router's job,
    already exhausted) and surfaces the right terminal error type.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentic_core.core.errors import AllModelsFailedError, MalformedOutputError, PermanentError
from agentic_core.core.gateway import Gateway
from conftest import FakeRouter, make_response


class Weather(BaseModel):
    temp: int
    summary: str


def _gateway(behaviours, offline_settings, **kw):
    return Gateway(router=FakeRouter(behaviours), settings=offline_settings, **kw)


async def test_free_text_call_returns_completion(offline_settings):
    gw = _gateway([make_response("hello there")], offline_settings)
    out = await gw.complete([{"role": "user", "content": "hi"}])
    assert out.text == "hello there"
    assert out.parsed is None
    assert out.reprompt_attempts == 0


async def test_structured_output_parsed_and_validated(offline_settings):
    gw = _gateway([make_response('{"temp": 21, "summary": "warm"}')], offline_settings)
    out = await gw.complete(
        [{"role": "user", "content": "weather?"}], response_model=Weather
    )
    assert isinstance(out.parsed, Weather)
    assert out.parsed.temp == 21
    assert out.reprompt_attempts == 0


async def test_captures_tokens_cost_and_latency(offline_settings):
    gw = _gateway([make_response("ok", prompt_tokens=10, completion_tokens=4, cost=0.002)], offline_settings)
    out = await gw.complete([{"role": "user", "content": "hi"}])
    assert out.usage.total_tokens == 14
    assert out.cost_usd == pytest.approx(0.002)
    assert out.latency_ms is not None and out.latency_ms >= 0.0


async def test_malformed_output_triggers_reprompt(offline_settings):
    # First reply is junk, second is valid -> gateway re-prompts once and succeeds.
    router = FakeRouter([make_response("not json at all"), make_response('{"temp": 9, "summary": "cold"}')])
    gw = Gateway(router=router, settings=offline_settings, max_reprompts=2)
    out = await gw.complete(
        [{"role": "user", "content": "weather?"}], response_model=Weather
    )
    assert out.parsed.temp == 9
    assert out.reprompt_attempts == 1
    assert len(router.calls) == 2

    # The re-prompt must be honest: the 2nd call carries the bad answer + a correction.
    second_msgs = router.calls[1]["messages"]
    roles_contents = [(m["role"], m["content"]) for m in second_msgs]
    assert ("assistant", "not json at all") in roles_contents
    assert any(role == "user" and "did not validate" in content for role, content in roles_contents)


async def test_reprompt_accumulates_cost_across_attempts(offline_settings):
    router = FakeRouter([make_response("junk", cost=0.001), make_response('{"temp": 1, "summary": "x"}', cost=0.001)])
    gw = Gateway(router=router, settings=offline_settings, max_reprompts=2)
    out = await gw.complete([{"role": "user", "content": "w?"}], response_model=Weather)
    # Both attempts cost money; honest accounting sums them.
    assert out.cost_usd == pytest.approx(0.002)


async def test_exhausted_reprompts_raise_malformed(offline_settings):
    router = FakeRouter([make_response("nope"), make_response("still nope")])
    gw = Gateway(router=router, settings=offline_settings, max_reprompts=1)
    with pytest.raises(MalformedOutputError) as ei:
        await gw.complete([{"role": "user", "content": "w?"}], response_model=Weather)
    # 1 initial + 1 re-prompt = 2 attempts, then terminal.
    assert len(router.calls) == 2
    assert ei.value.raw_output == "still nope"


async def test_permanent_provider_error_fails_fast_without_retry(offline_settings):
    from litellm import exceptions as le

    err = le.AuthenticationError("bad key", llm_provider="openrouter", model="fast")
    router = FakeRouter([err])
    gw = Gateway(router=router, settings=offline_settings)
    with pytest.raises(PermanentError):
        await gw.complete([{"role": "user", "content": "hi"}])
    # Fail fast: a permanent cause is never retried.
    assert len(router.calls) == 1


async def test_router_exhaustion_surfaces_as_all_models_failed(offline_settings):
    from litellm import exceptions as le

    # A transient cause reaching the gateway means the Router already gave up.
    err = le.RateLimitError("slow down", llm_provider="openrouter", model="fast")
    router = FakeRouter([err])
    gw = Gateway(router=router, settings=offline_settings)
    with pytest.raises(AllModelsFailedError):
        await gw.complete([{"role": "user", "content": "hi"}])
    # Crucial: the gateway does NOT re-try transient failures itself.
    assert len(router.calls) == 1
