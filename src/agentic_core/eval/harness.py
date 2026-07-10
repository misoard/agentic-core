"""eval/harness.py — datasets -> run -> score -> report -> compare.

Why this module exists
----------------------
Prompt and model changes are guesses until you measure them. A fixed dataset, run
through the same task and scored the same way, turns "I think this is better" into
a number you can diff. It's the layer most people skip and the one senior roles
ask about — so it's a first-class loop here, not an afterthought.

The pieces (all small, all composable)
--------------------------------------
  * ``EvalCase``   — one input (+ optional reference output) to run and score.
  * ``Task``       — ``async (input) -> output``; usually ``agent.run(...).parsed``.
  * ``Scorer``     — ``(output, case) -> Score``; sync or async. Deterministic ones
    (``exact_match``, ``numeric_close``) and an LLM-as-judge (``llm_judge``) are
    provided; a project adds its own.
  * ``run_eval``   — runs every case (concurrently), applies scorers, captures
    per-case errors, and returns an ``EvalReport``.
  * ``compare``    — per-scorer deltas between two reports: the regression check.

The mechanism/policy split holds: the harness + generic scorers are the package's;
the dataset, the rubric text, and project-specific metrics live in your repo.

Interview line: "This is where prompt or model changes are measured, not guessed —
a fixed dataset, deterministic metrics, error analysis, and LLM-as-judge."
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Sequence

from pydantic import BaseModel

# A task turns a case's input into an output to be scored (typically an agent).
Task = Callable[[Any], Awaitable[Any]]


@dataclass
class EvalCase:
    """One evaluation example: an input, an optional reference, some metadata."""

    id: str
    input: Any
    # Reference/expected output for reference-based metrics; None is fine for
    # reference-free scoring (e.g. an LLM-judge rubric with no gold answer).
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    """One scorer's verdict on one case. ``value`` is normalized to [0, 1]."""

    name: str
    value: float
    passed: bool
    detail: str | None = None


# A scorer grades an output against its case. Sync or async (async lets a scorer
# call the gateway, e.g. the LLM-judge).
Scorer = Callable[[Any, EvalCase], "Score | Awaitable[Score]"]


@dataclass
class CaseResult:
    """What happened for one case: its output, its scores, or the error it hit."""

    case_id: str
    output: Any = None
    scores: list[Score] = field(default_factory=list)
    error: str | None = None  # set if the task/scorer raised; the run continues


@dataclass
class EvalReport:
    """The whole run. ``summary`` aggregates; ``compare`` diffs two of these."""

    results: list[CaseResult]

    def summary(self) -> dict[str, Any]:
        """Per-scorer mean value + pass-rate, plus how many cases errored.

        Errored cases are excluded from scorer means (they produced no score) but
        surfaced via ``errors`` — silently averaging over failures would hide them.
        """
        by_scorer: dict[str, list[Score]] = {}
        for res in self.results:
            for score in res.scores:
                by_scorer.setdefault(score.name, []).append(score)

        scorers = {
            name: {
                "mean": sum(s.value for s in scores) / len(scores),
                "pass_rate": sum(s.passed for s in scores) / len(scores),
                "n": len(scores),
            }
            for name, scores in by_scorer.items()
        }
        return {
            "n_cases": len(self.results),
            "errors": sum(1 for r in self.results if r.error is not None),
            "scorers": scorers,
        }


async def run_eval(
    dataset: Sequence[EvalCase],
    task: Task,
    scorers: Sequence[Scorer],
    *,
    concurrency: int = 8,
) -> EvalReport:
    """Run every case through ``task``, score it, and collect an ``EvalReport``.

    Cases run concurrently (bounded by ``concurrency`` so a big dataset doesn't
    open thousands of calls at once). A case that raises is captured as a
    ``CaseResult`` with ``error`` set — one bad case never sinks the whole run,
    which is exactly what you want when eyeballing failures.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with semaphore:
            try:
                output = await task(case.input)
            except Exception as exc:  # capture, don't abort the batch
                return CaseResult(case_id=case.id, error=f"task: {exc}")
            scores: list[Score] = []
            for scorer in scorers:
                try:
                    result = scorer(output, case)
                    score = await result if isawaitable(result) else result
                    scores.append(score)
                except Exception as exc:
                    return CaseResult(
                        case_id=case.id, output=output, error=f"scorer: {exc}"
                    )
            return CaseResult(case_id=case.id, output=output, scores=scores)

    results = await asyncio.gather(*(run_one(c) for c in dataset))
    return EvalReport(results=list(results))


def compare(baseline: EvalReport, candidate: EvalReport) -> dict[str, Any]:
    """Per-scorer mean deltas (candidate - baseline): the regression check.

    Positive delta = improvement. This is the "did my change make it better?"
    answer made numeric — wire it into CI to fail a PR that regresses a metric.
    """
    base = baseline.summary()["scorers"]
    cand = candidate.summary()["scorers"]
    deltas: dict[str, Any] = {}
    for name in sorted(set(base) | set(cand)):
        b = base.get(name, {}).get("mean")
        c = cand.get(name, {}).get("mean")
        deltas[name] = {
            "baseline": b,
            "candidate": c,
            "delta": None if b is None or c is None else c - b,
        }
    return deltas


# --- deterministic scorers -------------------------------------------------

def exact_match(output: Any, case: EvalCase) -> Score:
    """1.0 iff output equals case.expected. The simplest reference metric."""
    ok = output == case.expected
    return Score("exact_match", 1.0 if ok else 0.0, ok)


def numeric_close(tolerance: float = 1e-6) -> Scorer:
    """Scorer factory: pass if |output - expected| <= tolerance (numeric outputs)."""

    def scorer(output: Any, case: EvalCase) -> Score:
        try:
            ok = abs(float(output) - float(case.expected)) <= tolerance
        except (TypeError, ValueError):
            return Score("numeric_close", 0.0, False, "non-numeric")
        return Score("numeric_close", 1.0 if ok else 0.0, ok)

    return scorer


# --- LLM-as-judge scorer (routes through the gateway) ----------------------

class JudgeVerdict(BaseModel):
    """Typed contract for the judge's grade."""

    passed: bool
    score: float  # 0..1
    reasoning: str


def llm_judge(gateway: Any, *, rubric: str, model: str | None = None) -> Scorer:
    """Scorer factory: grade the output with an LLM against a rubric.

    Reference-free by default (uses ``case.expected`` only if you put it in the
    rubric prompt). Like every call, it goes through the gateway, so the judge
    inherits retry/validation/cost tracking. ``gateway`` is loosely typed to avoid
    an import cycle; pass a real ``Gateway``.
    """

    async def scorer(output: Any, case: EvalCase) -> Score:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. Grade the candidate output "
                    "against the rubric. Return the required JSON with a 0..1 score.\n\n"
                    f"Rubric:\n{rubric}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Input:\n{case.input}\n\n"
                    f"Reference (may be empty):\n{case.expected}\n\n"
                    f"Candidate output:\n{output}"
                ),
            },
        ]
        completion = await gateway.complete(
            messages, model=model, response_model=JudgeVerdict
        )
        v = completion.parsed
        return Score("llm_judge", v.score, v.passed, v.reasoning)

    return scorer
