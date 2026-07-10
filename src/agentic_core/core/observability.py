"""observability — one span around every model call.

Why this module exists
----------------------
You can't optimize latency or cost you don't measure. Because every call already
flows through the gateway (the keystone), wrapping *that one place* in a span
gives complete, uniform coverage: latency, tokens, cost, which model actually
served, re-prompt count, and errors — for every call in the system, captured the
same way, for free.

Design: no-op by default
------------------------
This module talks to the **OpenTelemetry API**, not a concrete backend. Until an
app calls ``configure`` (or wires its own OTel provider), OTel's global tracer is
a non-recording no-op: ``completion_span`` still runs, but the spans do nothing
and reach no network. That's what keeps the test suite fully offline while the
gateway is instrumented unconditionally. ``configure`` is the opt-in seam that
points those spans at Logfire (or a local console) — and it lives at the app's
startup, never in library code paths.

Attribute naming follows OTel's ``gen_ai.*`` semantic conventions where they
exist (so standard tooling understands them), with ``agentic.*`` for the few
things unique to this core (cost, re-prompt count).

Interview line: "The '9-second threshold' only means something because I measured
the latency distribution first — observability comes before optimization."
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

if TYPE_CHECKING:  # import only for types; no runtime coupling to those modules
    from .config import Settings
    from .schemas import Completion

# Captured once. It's a ProxyTracer that forwards to whatever global provider is
# set later, so `configure` (or a test harness) can install a provider after import.
_TRACER = trace.get_tracer("agentic_core")

# The single span name every model call is recorded under.
_SPAN_NAME = "llm.completion"


def configure(
    settings: "Settings | None" = None,
    *,
    service_name: str = "agentic-core",
) -> bool:
    """Opt-in: point the gateway's spans at a backend. Returns True if wired.

    Call this once at app startup. Priority: Logfire token, then a Jaeger (OTLP)
    endpoint, then local console printing; with none set this does nothing and
    spans stay no-op. Library/test code never calls it, so the default path is
    always offline. The first matching branch wins — pick one backend per process,
    since OTel allows the global tracer provider to be set only once.
    """
    from .config import get_settings

    settings = settings or get_settings()

    if settings.logfire_token:
        import logfire

        logfire.configure(token=settings.logfire_token, service_name=service_name)
        return True

    if settings.jaeger_endpoint:
        # Jaeger speaks OTLP natively (since v1.35), so we export plain OTLP/HTTP
        # to its collector rather than using the deprecated Jaeger exporter. This
        # is also how you'd target any OTLP backend (Tempo, Honeycomb, ...).
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        # Batched, not simple: real export shouldn't add latency to the call path.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.jaeger_endpoint))
        )
        trace.set_tracer_provider(provider)
        return True

    if settings.otel_console_export:
        import logfire

        # send_to_logfire=False -> local console span printing, no account.
        logfire.configure(send_to_logfire=False, service_name=service_name)
        return True

    return False


@contextmanager
def completion_span(
    *, model: str, response_model: str | None = None
) -> Iterator[Span]:
    """Wrap one gateway call. Records the request up front; on exception records
    the error and marks the span failed, then re-raises (never swallows)."""
    with _TRACER.start_as_current_span(_SPAN_NAME) as span:
        span.set_attribute("gen_ai.request.model", model)  # the alias asked for
        if response_model is not None:
            span.set_attribute("agentic.response_model", response_model)
        try:
            yield span
        except BaseException as exc:
            # Observability observes failures too — the whole point of measuring.
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("agentic.error.type", type(exc).__name__)
            raise


def record_completion(span: Span, completion: "Completion", *, alias: str) -> None:
    """Stamp the finished call's metrics onto its span (tokens, cost, latency...)."""
    span.set_attribute("gen_ai.request.alias", alias)
    if completion.model:
        # The concrete model that actually served, post-fallback — may differ
        # from the alias, which is exactly what you want to see in a trace.
        span.set_attribute("gen_ai.response.model", completion.model)

    usage = completion.usage
    if usage is not None:
        span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)

    if completion.cost_usd is not None:
        span.set_attribute("agentic.cost_usd", completion.cost_usd)
    if completion.latency_ms is not None:
        span.set_attribute("agentic.latency_ms", completion.latency_ms)
    # Re-prompt count is a reliability signal unique to this core: a call that
    # needed re-prompts cost more and ran slower for a reason worth surfacing.
    span.set_attribute("agentic.reprompt_attempts", completion.reprompt_attempts)
