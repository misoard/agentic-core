"""Tests for gateway tracing.

We install a real OTel SDK provider with an in-memory exporter (no network), run
calls through the gateway, and assert the emitted span carries the right metrics
and error status. This proves the instrumentation without a Logfire account.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

from agentic_core.core.errors import PermanentError
from agentic_core.core.gateway import Gateway
from conftest import FakeRouter, make_response


class Weather(BaseModel):
    temp: int
    summary: str


# The global tracer provider can only be set once per process, so do it at module
# scope; each test clears the exporter for isolation.
@pytest.fixture(scope="module")
def _exporter() -> InMemorySpanExporter:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def spans(_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    _exporter.clear()
    return _exporter


def _attrs(exporter: InMemorySpanExporter) -> dict:
    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    return dict(finished[0].attributes), finished[0]


async def test_span_captures_model_tokens_and_cost(offline_settings, spans):
    gw = Gateway(
        router=FakeRouter([make_response("hi", prompt_tokens=10, completion_tokens=4, cost=0.003)]),
        settings=offline_settings,
    )
    await gw.complete([{"role": "user", "content": "hi"}], model="fast")

    attrs, span = _attrs(spans)
    assert span.name == "llm.completion"
    assert attrs["gen_ai.request.model"] == "fast"           # the alias asked for
    assert attrs["gen_ai.response.model"].startswith("openrouter/")  # concrete server
    assert attrs["gen_ai.usage.total_tokens"] == 14
    assert attrs["agentic.cost_usd"] == pytest.approx(0.003)
    assert attrs["agentic.reprompt_attempts"] == 0
    assert "agentic.latency_ms" in attrs


async def test_span_records_reprompt_count(offline_settings, spans):
    gw = Gateway(
        router=FakeRouter([make_response("junk"), make_response('{"temp": 3, "summary": "x"}')]),
        settings=offline_settings,
        max_reprompts=2,
    )
    await gw.complete([{"role": "user", "content": "w?"}], response_model=Weather)

    attrs, _ = _attrs(spans)
    assert attrs["agentic.reprompt_attempts"] == 1


async def test_span_records_error_status(offline_settings, spans):
    from litellm import exceptions as le

    err = le.AuthenticationError("bad key", llm_provider="openrouter", model="fast")
    gw = Gateway(router=FakeRouter([err]), settings=offline_settings)

    with pytest.raises(PermanentError):
        await gw.complete([{"role": "user", "content": "hi"}])

    _, span = _attrs(spans)
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR
    # The surfaced (classified) error type is what's recorded, not the raw one.
    assert span.attributes["agentic.error.type"] == "PermanentError"
    assert len(span.events) >= 1  # record_exception adds an event
