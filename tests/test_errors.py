"""Tests for the error taxonomy and the provider-error classifier.

These pin down the point-4 division of labour: the classifier sorts a *terminal*
provider exception into "permanent (fail fast)" vs "transient (Router gave up)",
and never into anything the gateway would retry itself.
"""

from __future__ import annotations

import pytest

from agentic_core.core.errors import (
    AgenticError,
    AllModelsFailedError,
    MalformedOutputError,
    PermanentError,
    TransientError,
    classify_provider_error,
)


def test_all_categories_share_one_base():
    # Catching AgenticError must catch everything this core raises.
    for exc in (
        TransientError("t"),
        PermanentError("p"),
        MalformedOutputError("m", raw_output="{}"),
        AllModelsFailedError("a"),
    ):
        assert isinstance(exc, AgenticError)


def test_malformed_carries_bad_answer_and_reason():
    # The re-prompt is only honest if it can quote the bad answer + why it failed.
    cause = ValueError("field required")
    err = MalformedOutputError("bad", raw_output="{not json}", validation_error=cause)
    assert err.raw_output == "{not json}"
    assert err.validation_error is cause


def test_all_models_failed_records_what_was_attempted():
    err = AllModelsFailedError("dead", attempted=["gpt-4o-mini", "claude"])
    assert err.attempted == ["gpt-4o-mini", "claude"]


def test_classify_passes_our_own_errors_through_unchanged():
    ours = PermanentError("already classified")
    assert classify_provider_error(ours) is ours


def test_classify_permanent_provider_errors():
    from litellm import exceptions as le

    auth = le.AuthenticationError("nope", llm_provider="openai", model="gpt-4o")
    out = classify_provider_error(auth)
    assert isinstance(out, PermanentError)


def test_classify_transient_provider_errors():
    from litellm import exceptions as le

    rate = le.RateLimitError("slow down", llm_provider="openai", model="gpt-4o")
    out = classify_provider_error(rate)
    # Transient cause -> TransientError. The gateway turns this into
    # AllModelsFailedError; the classifier's job is only to name the cause.
    assert isinstance(out, TransientError)


def test_classify_unknown_defaults_to_permanent():
    # An error we can't recognise is surfaced loudly, not silently retried.
    out = classify_provider_error(RuntimeError("???"))
    assert isinstance(out, PermanentError)
