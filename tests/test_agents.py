"""Tests for the agent abstraction, using the real gateway + a fake Router.

Demo-free: uses an inline test agent (conftest.make_text_agent) so the core suite
has no dependency on the consumer/demo. These exercise the agent boundary: typed
output back, input validated on the way in, prompt rendered, model alias forwarded.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_core.agents.base import Agent
from agentic_core.core.gateway import Gateway
from conftest import EchoIn, EchoOut, FakeRouter, make_response, make_text_agent


def _gateway_returning(json_text: str, offline_settings, **kw):
    return Gateway(router=FakeRouter([make_response(json_text)]), settings=offline_settings, **kw)


async def test_run_returns_typed_validated_output(offline_settings):
    gw = _gateway_returning('{"label": "positive"}', offline_settings)
    agent = make_text_agent(gw, model="fast")
    out = await agent.run(EchoIn(text="I love this"))
    assert isinstance(out.parsed, EchoOut)
    assert out.parsed.label == "positive"


async def test_run_accepts_raw_dict_and_validates_at_boundary(offline_settings):
    gw = _gateway_returning('{"label": "neutral"}', offline_settings)
    agent = make_text_agent(gw)
    out = await agent.run({"text": "it is fine"})  # dict coerced via input_model
    assert out.parsed.label == "neutral"


async def test_bad_input_dict_is_rejected_before_any_call(offline_settings):
    router = FakeRouter([make_response('{"label": "neutral"}')])
    gw = Gateway(router=router, settings=offline_settings)
    agent = make_text_agent(gw)
    with pytest.raises(ValidationError):
        await agent.run({"not_text": "oops"})
    # Boundary validation means we never spent a model call on bad input.
    assert router.calls == []


async def test_prompt_rendered_and_model_alias_forwarded(offline_settings):
    router = FakeRouter([make_response('{"label": "negative"}')])
    gw = Gateway(router=router, settings=offline_settings)
    agent = make_text_agent(gw, model="smart")
    await agent.run(EchoIn(text="this is terrible"))

    call = router.calls[0]
    assert call["model"] == "smart"  # the agent's pinned alias reached the gateway
    user_msg = next(m for m in call["messages"] if m["role"] == "user")
    assert "this is terrible" in user_msg["content"]  # template rendered the field
    assert any(m["role"] == "system" for m in call["messages"])  # instructions present


async def test_agent_without_input_model_rejects_dict(offline_settings):
    gw = _gateway_returning('{"label": "neutral"}', offline_settings)
    agent = Agent(
        name="no_input_model",
        gateway=gw,
        output_model=EchoOut,
        system_prompt="x",
        user_template="{text}",
    )
    with pytest.raises(TypeError):
        await agent.run({"text": "hi"})  # no input_model -> can't validate a raw dict
