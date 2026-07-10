"""gateway — the keystone: the one path every model call takes.

Why this module exists
----------------------
Every model call in the whole system flows through ``Gateway.complete``. That
single chokepoint is what lets reliability, latency, and cost be guaranteed
*centrally* instead of scattered across the codebase. Nothing else ever calls a
provider SDK (or LiteLLM, or the Router) directly — if it did, we'd lose the one
guarantee this architecture is built to make.

The control flow, mapped explicitly (this is the part to read slowly)
--------------------------------------------------------------------
Two retry-shaped jobs exist and each lives in exactly one layer (see errors.py):

  1. Transient network failure (timeout / rate-limit / 5xx) -> owned by the
     **Router**, underneath us. The gateway has NO transient-retry loop. When the
     Router raises, it has already exhausted its own retries AND every fallback,
     so we treat that as terminal:
        * a *permanent* cause (auth, bad request, unknown model) -> PermanentError
        * a *transient* cause the Router gave up on          -> AllModelsFailedError
     Either way we do not try again.

  2. Malformed output (a reply that doesn't satisfy the caller's Pydantic schema)
     -> owned by the **gateway**, only. The Router can't see our schema, so only
     we can re-prompt: feed the model its own bad answer plus the validation
     error and ask again. This is the ONLY retry-like loop in this file, and it is
     driven by tenacity (a hand-rolled loop would be noisier).

So: tenacity here governs the *re-prompt*, never transient retries. If you ever
find yourself adding a retry for a network error in this file, stop — that job
already belongs to the Router.

Interview line: "Every model call flows through one client, so retry, fallback,
timeout, output validation, and cost tracking live in exactly one place."
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from .config import Deployment, Settings, build_router_kwargs, get_settings
from .errors import (
    AllModelsFailedError,
    MalformedOutputError,
    PermanentError,
    classify_provider_error,
)
from .observability import completion_span, record_completion
from .schemas import Completion, TokenUsage, ToolCall

# OpenAI-format messages: [{"role": "user", "content": "..."}, ...].
Messages = list[dict[str, Any]]


class RouterLike(Protocol):
    """The slice of ``litellm.Router`` the gateway depends on.

    Declaring it as a Protocol keeps the gateway testable: tests inject a fake
    with one async method instead of standing up a real Router + provider.
    """

    async def acompletion(
        self, *, model: str, messages: Messages, **kwargs: Any
    ) -> Any: ...


class Gateway:
    """The single LLM client. Build one, share it, route every call through it."""

    def __init__(
        self,
        *,
        router: RouterLike | None = None,
        settings: Settings | None = None,
        max_reprompts: int = 2,
        registry: dict[str, Deployment] | None = None,
        fallbacks: dict[str, list[str]] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # How many times we'll re-prompt on malformed output before giving up.
        # This bounds the ONLY loop the gateway owns; transient retries are the
        # Router's and are configured in build_router_kwargs, not here.
        self._max_reprompts = max_reprompts
        self._router = router or self._build_router(registry, fallbacks)

    def _build_router(
        self,
        registry: dict[str, Deployment] | None,
        fallbacks: dict[str, list[str]] | None,
    ) -> RouterLike:
        # Imported lazily so importing the gateway (e.g. for type hints or tests
        # with an injected router) doesn't drag in the heavy Router at import time.
        from litellm import Router

        kwargs = build_router_kwargs(
            self._settings, registry=registry, fallbacks=fallbacks
        )
        return Router(**kwargs)

    async def complete(
        self,
        messages: Messages,
        *,
        model: str | None = None,
        response_model: type[BaseModel] | None = None,
        **opts: Any,
    ) -> Completion:
        """Run one call end-to-end and return a validated ``Completion``.

        ``model`` is a registry alias ("fast"/"smart"), not a provider string —
        the Router maps it. ``response_model`` opts into structured output: the
        gateway instructs the model to emit matching JSON, validates it, and
        re-prompts on failure. Omit it for free text.
        """
        model = model or self._settings.default_model
        # Observability wraps the keystone: one span per call, capturing model,
        # tokens, cost, latency, reprompts, and errors. No-op unless configured.
        with completion_span(
            model=model,
            response_model=response_model.__name__ if response_model else None,
        ) as span:
            completion = await self._run(
                messages, model=model, response_model=response_model, **opts
            )
            record_completion(span, completion, alias=model)
            return completion

    async def _run(
        self,
        messages: Messages,
        *,
        model: str,
        response_model: type[BaseModel] | None,
        **opts: Any,
    ) -> Completion:
        """The reliability core: the malformed-output re-prompt loop plus terminal
        error classification. Split out from complete() so the span there reads as
        one clean wrapper around the whole call (and errors from here propagate up
        into that span, where they're recorded)."""

        # When structured output is asked for, append a schema instruction. Kept
        # provider-agnostic on purpose (works even on models without native
        # structured-output support) so the re-prompt loop is the real guarantee.
        base = list(messages)
        if response_model is not None:
            base = [*base, self._schema_instruction(response_model)]

        # Mutated between re-prompts: the model's bad answer + why it was rejected.
        feedback: Messages = []
        # Cost/tokens accumulate across re-prompts — a re-prompt isn't free, and
        # honest cost accounting must include the wasted attempts.
        totals = {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0}
        cost_seen = False
        started = time.perf_counter()

        try:
            # tenacity retries ONLY on MalformedOutputError. A provider exception
            # (the Router having given up) does not match the predicate, so with
            # reraise=True it propagates straight out to the except blocks below —
            # never retried here.
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(1 + self._max_reprompts),
                retry=retry_if_exception_type(MalformedOutputError),
                reraise=True,
            ):
                with attempt:
                    reprompts = attempt.retry_state.attempt_number - 1
                    response = await self._router.acompletion(
                        model=model, messages=base + feedback, **opts
                    )

                    p, c, t, cost = self._extract_usage_and_cost(response)
                    totals["prompt"] += p
                    totals["completion"] += c
                    totals["total"] += t
                    if cost is not None:
                        totals["cost"] += cost
                        cost_seen = True

                    text = self._extract_text(response)

                    parsed: BaseModel | None = None
                    if response_model is not None:
                        try:
                            parsed = response_model.model_validate_json(
                                self._strip_fences(text)
                            )
                        except (ValidationError, ValueError) as exc:
                            # The Router thinks this call succeeded; only we know
                            # the content is wrong. Own the re-prompt: stash the
                            # correction and raise so tenacity loops.
                            feedback = self._correction(text, exc)
                            raise MalformedOutputError(
                                f"Output failed {response_model.__name__} validation",
                                raw_output=text,
                                validation_error=exc,
                            ) from exc

                    return self._build_completion(
                        text=text,
                        response=response,
                        parsed=parsed,
                        totals=totals,
                        cost_seen=cost_seen,
                        started=started,
                        reprompts=reprompts,
                    )

        except MalformedOutputError:
            # Re-prompts exhausted: the model never produced schema-valid output.
            # Terminal — surface it (carries the last bad answer + validation error).
            raise
        except PermanentError:
            raise
        except Exception as exc:
            # We only reach here if the Router itself raised — i.e. it exhausted
            # its transient retries and fallbacks. Classify why, then surface. The
            # gateway does not retry either branch.
            classified = classify_provider_error(exc)
            if isinstance(classified, PermanentError):
                raise classified from exc
            raise AllModelsFailedError(
                f"Router exhausted retries and fallbacks for model={model!r}: {classified}",
                attempted=[model],
            ) from exc

        # Unreachable: the loop above always returns or raises.
        raise AssertionError("gateway._run exited its retry loop without result")

    # --- helpers: small, single-purpose, so the flow above stays legible -------

    @staticmethod
    def _schema_instruction(response_model: type[BaseModel]) -> dict[str, Any]:
        schema = json.dumps(response_model.model_json_schema())
        return {
            "role": "system",
            "content": (
                "You must respond with a single JSON object that validates "
                "against this JSON Schema. Output only the JSON — no prose, no "
                f"markdown fences.\n\nJSON Schema:\n{schema}"
            ),
        }

    @staticmethod
    def _correction(bad_output: str, error: Exception) -> Messages:
        # An honest re-prompt: quote what the model said and exactly why it failed.
        return [
            {"role": "assistant", "content": bad_output},
            {
                "role": "user",
                "content": (
                    f"Your previous response did not validate:\n{error}\n\n"
                    "Return a corrected JSON object that satisfies the schema. "
                    "Output only JSON."
                ),
            },
        ]

    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```(?:json)?\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        return t.strip()

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError):
            content = None
        return content or ""

    @staticmethod
    def _extract_tool_calls(response: Any) -> tuple[ToolCall, ...]:
        """Pull any tool-call requests off the response (OpenAI/LiteLLM shape).

        The gateway only *reports* them — it never executes; the agent-layer tool
        loop does. Defensive like _extract_text: a response without tool_calls
        (the common case) yields an empty tuple, never an error.
        """
        try:
            raw = response.choices[0].message.tool_calls
        except (AttributeError, IndexError, KeyError, TypeError):
            return ()
        if not raw:
            return ()
        calls: list[ToolCall] = []
        for tc in raw:
            try:
                args = tc.function.arguments
                # Providers send arguments as a JSON string; tolerate a dict too.
                parsed = json.loads(args) if isinstance(args, str) else (args or {})
                calls.append(
                    ToolCall(
                        id=getattr(tc, "id", "") or "",
                        name=tc.function.name,
                        arguments=parsed if isinstance(parsed, dict) else {},
                    )
                )
            except (AttributeError, ValueError, TypeError):
                continue  # skip a malformed tool-call entry rather than crash
        return tuple(calls)

    @staticmethod
    def _extract_usage_and_cost(response: Any) -> tuple[int, int, int, float | None]:
        usage = getattr(response, "usage", None)
        p = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        c = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        t = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0

        # LiteLLM stashes its computed cost here; fall back to recomputing.
        cost: float | None = None
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
        if cost is None:
            try:
                import litellm

                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = None
        return p, c, t, cost

    @staticmethod
    def _build_completion(
        *,
        text: str,
        response: Any,
        parsed: BaseModel | None,
        totals: dict[str, Any],
        cost_seen: bool,
        started: float,
        reprompts: int,
    ) -> Completion:
        latency_ms = (time.perf_counter() - started) * 1000.0
        has_tokens = totals["total"] or totals["prompt"] or totals["completion"]
        usage = (
            TokenUsage(
                prompt_tokens=totals["prompt"],
                completion_tokens=totals["completion"],
                total_tokens=totals["total"],
            )
            if has_tokens
            else None
        )
        return Completion(
            text=text,
            model=getattr(response, "model", None) or "",
            parsed=parsed,
            usage=usage,
            cost_usd=totals["cost"] if cost_seen else None,
            latency_ms=latency_ms,
            reprompt_attempts=reprompts,
            tool_calls=Gateway._extract_tool_calls(response),
        )
