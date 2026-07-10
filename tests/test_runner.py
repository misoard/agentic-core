"""Tests for the orchestration primitives.

Mostly pure async (no LLM) — the runner is agent-agnostic, so it's tested with
trivial async steps. One integration test drives real agents through the gateway
(with a fake Router) to prove the spine composes: runner -> agent -> gateway.
"""

from __future__ import annotations

import asyncio

import pytest

from agentic_core.orchestration.runner import (
    State,
    run_concurrent,
    run_conditional,
    run_sequential,
)
from conftest import FakeRouter, make_response


async def test_sequential_threads_state_and_returns_results_in_order():
    async def step1(s: State):
        s["x"] = 1
        return "s1"

    async def step2(s: State):
        s["x"] += 1  # sees what step1 wrote
        return s["x"]

    state = State()
    results = await run_sequential(state, [step1, step2])
    assert results == ["s1", 2]
    assert state["x"] == 2


async def test_concurrent_actually_overlaps():
    # Deterministic proof of concurrency: `b` can only finish if `a` runs at the
    # same time (a sets the event b waits on). Sequential execution would deadlock.
    started = asyncio.Event()

    async def a(_s: State):
        started.set()
        return "a"

    async def b(_s: State):
        await asyncio.wait_for(started.wait(), timeout=1.0)
        return "b"

    results = await run_concurrent(State(), [b, a])
    assert results == ["b", "a"]  # results preserve step order, not finish order


async def test_concurrent_fails_fast_by_default():
    async def ok(_s):
        return 1

    async def boom(_s):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await run_concurrent(State(), [ok, boom])


async def test_concurrent_can_collect_exceptions():
    async def ok(_s):
        return 1

    async def boom(_s):
        raise ValueError("nope")

    results = await run_concurrent(State(), [ok, boom], return_exceptions=True)
    assert results[0] == 1
    assert isinstance(results[1], ValueError)


async def test_conditional_selects_branch():
    async def yes(_s):
        return "yes"

    async def no(_s):
        return "no"

    assert await run_conditional(State({"flag": True}), lambda s: s.get("flag"), yes, no) == "yes"
    assert await run_conditional(State({"flag": False}), lambda s: s.get("flag"), yes, no) == "no"
    # Missing false branch -> None (early-exit / no-op path).
    assert await run_conditional(State({"flag": False}), lambda s: s.get("flag"), yes) is None


async def test_agents_compose_concurrently_through_the_gateway(offline_settings):
    # The previewed spine: two stateless agents run at once under run_concurrent,
    # each through its own gateway. Demo-free — uses the inline test agent.
    from agentic_core.core.gateway import Gateway
    from conftest import EchoIn, make_text_agent

    gw_a = Gateway(
        router=FakeRouter([make_response('{"label": "a"}')]),
        settings=offline_settings,
    )
    gw_b = Gateway(
        router=FakeRouter([make_response('{"label": "b"}')]),
        settings=offline_settings,
    )
    agent_a = make_text_agent(gw_a)
    agent_b = make_text_agent(gw_b)

    state = State({"t1": "great", "t2": "awful"})
    r_a, r_b = await run_concurrent(
        state,
        [
            lambda s: agent_a.run(EchoIn(text=s["t1"])),
            lambda s: agent_b.run(EchoIn(text=s["t2"])),
        ],
    )
    assert r_a.parsed.label == "a"
    assert r_b.parsed.label == "b"
