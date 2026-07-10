"""Tests for the shared carrier contracts (Completion / TokenUsage)."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel

from agentic_core.core.schemas import Completion, TokenUsage


class _Parsed(BaseModel):
    value: int


def test_completion_defaults_to_free_text():
    c = Completion(text="hello", model="gpt-4o-mini")
    assert c.parsed is None
    assert c.reprompt_attempts == 0
    assert c.extra == {}


def test_completion_carries_parsed_model_and_metadata():
    c = Completion(
        text='{"value": 7}',
        model="gpt-4o-mini",
        parsed=_Parsed(value=7),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        cost_usd=0.00012,
        latency_ms=420.0,
        reprompt_attempts=1,
    )
    assert c.parsed.value == 7
    assert c.usage.total_tokens == 13
    assert c.cost_usd == pytest.approx(0.00012)
    assert c.reprompt_attempts == 1


def test_completion_is_frozen():
    # A completed call is a fact, not mutable state.
    c = Completion(text="x", model="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "y"  # type: ignore[misc]
