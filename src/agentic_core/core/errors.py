"""errors — the failure taxonomy that the gateway's control flow is built on.

Why this module exists
----------------------
Reliability is mostly *error classification*: knowing which failures to retry,
which to re-prompt, and which to fail on. The gateway can only be as good as the
categories it sorts failures into, so those categories live here, in one place,
named explicitly.

The division of labour (read this before touching the gateway)
--------------------------------------------------------------
Two retry-shaped jobs exist, and each belongs to exactly ONE layer so they
compose instead of multiplying:

  * Transient network failure (timeout / rate-limit / 5xx) — "try again and it
    might work" — is owned by the **LiteLLM Router**, only. The gateway has no
    transient-retry loop. When the Router has exhausted its own retries *and*
    every fallback, that is terminal: the gateway surfaces it as
    ``AllModelsFailedError`` and does NOT try again.
    
  * Malformed output (the model replied, but the text didn't satisfy our Pydantic
    schema) is owned by the **gateway**, only. To the Router that call *succeeded*
    — it got a valid HTTP response with text. Only the gateway knows the content
    was wrong against our schema, so only the gateway can re-prompt. The Router
    has no concept of our schema, so it cannot own this.

So: Router owns "retry the network failure", gateway owns "re-prompt the
wrong-shaped answer", and the gateway never wraps the Router in its own
transient-retry loop.

The categories
--------------
``TransientError``  network-level, retryable — but it's the Router's job, not ours.
``PermanentError``  will fail again identically (auth, bad request, content policy) -> fail fast.
``MalformedOutputError``  call succeeded, content didn't match the schema -> gateway re-prompts.
``AllModelsFailedError``  Router exhausted retries + fallbacks -> surface, terminal.

Interview line: "Reliability is mostly error classification — knowing which
failures to retry, which to re-prompt, and which to fall back on."
"""

from __future__ import annotations


class AgenticError(Exception):
    """Base for every error this core raises. Catch this to catch all of ours."""


class TransientError(AgenticError):
    """A network-level failure that *might* succeed on a retry.

    Owned by the Router, not the gateway. We define the category so the gateway
    can *recognise* a transient cause when classifying a terminal Router failure
    (for observability and for the right error type), never so the gateway can
    retry it itself.
    """


class PermanentError(AgenticError):
    """A failure that will recur identically no matter how many times we try.

    Auth failure, malformed request, content-policy violation, unknown model.
    There is nothing to retry or re-prompt: fail fast and surface it.
    """


class MalformedOutputError(AgenticError):
    """The provider answered, but the content did not satisfy our Pydantic schema.

    This is the ONE retry-shaped job the gateway still owns: feed the model its
    own bad answer plus the validation error and ask again (a re-prompt). It is
    not a network failure — to the provider and the Router, the call succeeded.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: str,
        validation_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        # Carry the bad answer + the reason it was bad so the gateway can build
        # an honest re-prompt ("here's what you said, here's why it didn't fit").
        self.raw_output = raw_output
        self.validation_error = validation_error


class AllModelsFailedError(AgenticError):
    """The Router exhausted its own retries AND every fallback model.

    Terminal by definition: the gateway surfaces this and does not retry. The
    originating provider exception is preserved via ``raise ... from exc`` and
    summarised in ``attempted``.
    """

    def __init__(self, message: str, *, attempted: list[str] | None = None) -> None:
        super().__init__(message)
        self.attempted = attempted or []


class GuardrailError(AgenticError):
    """A guardrail refused the input or output. Fail-closed and surface.

    Its own category on purpose: it is not transient (retrying won't help), not
    malformed (the schema was fine), and not a provider failure. A guard is a
    policy decision — an injection attempt, invalid tool args, PII in output — so
    the right behaviour is to stop and surface *why*, never to retry or re-prompt.
    """

    def __init__(self, message: str, *, guard: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.guard = guard  # which guard tripped, for logging/observability
        self.detail = detail


def classify_provider_error(exc: Exception) -> AgenticError:
    """Map a raw provider/Router exception onto our taxonomy.

    Used by the gateway at exactly one point: when a ``Router.acompletion`` call
    raises. By that moment the Router has already given up, so the only question
    left is *why*, which decides the error type we surface:

      * a *permanent* cause (auth, bad request, policy, unknown model) -> it would
        never have worked; surface ``PermanentError``.
      * a *transient* cause (timeout, rate-limit, 5xx, connection) -> the Router
        tried and exhausted itself; surface it as a transient-rooted terminal
        failure. The gateway wraps that into ``AllModelsFailedError``.

    LiteLLM is imported lazily so the taxonomy stays importable (and unit-testable)
    without the heavy provider dependency loaded, and so this file's coupling to
    LiteLLM lives in this one function.
    """
    # Already one of ours (e.g. re-raised through layers): pass it straight back.
    if isinstance(exc, AgenticError):
        return exc

    try:
        from litellm import exceptions as le
    except Exception:  # pragma: no cover - litellm always present in practice
        return PermanentError(f"Unclassifiable provider error: {exc!r}")

    # Permanent: identical retry would fail identically. Fail fast, no re-prompt.
    permanent = (
        le.AuthenticationError,
        le.PermissionDeniedError,
        le.NotFoundError,
        le.BadRequestError,
        le.UnprocessableEntityError,
        le.ContentPolicyViolationError,
    )
    # Transient: the Router's retry domain. We only ever see these here *after*
    # the Router has exhausted them, so reaching this branch means "gave up".
    transient = (
        le.Timeout,
        le.RateLimitError,
        le.ServiceUnavailableError,
        le.InternalServerError,
        le.APIConnectionError,
    )

    if isinstance(exc, permanent):
        return PermanentError(f"{type(exc).__name__}: {exc}")
    if isinstance(exc, transient):
        return TransientError(f"{type(exc).__name__}: {exc}")

    # Unknown shape: treat as permanent. Surfacing an unexpected error loudly is
    # safer than silently retrying something we don't understand.
    return PermanentError(f"Unclassified provider error {type(exc).__name__}: {exc}")
