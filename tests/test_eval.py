"""Tests for the eval harness. Offline — the task and LLM-judge use fakes."""

from __future__ import annotations

import pytest

from agentic_core.core.gateway import Gateway
from agentic_core.eval.harness import (
    EvalCase,
    EvalReport,
    Score,
    compare,
    exact_match,
    llm_judge,
    numeric_close,
    run_eval,
)
from conftest import FakeRouter, make_response


def _dataset():
    return [
        EvalCase(id="a", input="hello", expected="HELLO"),
        EvalCase(id="b", input="world", expected="WORLD"),
    ]


async def test_run_eval_scores_a_dataset():
    async def task(inp: str) -> str:
        return inp.upper()

    report = await run_eval(_dataset(), task, [exact_match])
    s = report.summary()
    assert s["n_cases"] == 2
    assert s["errors"] == 0
    assert s["scorers"]["exact_match"]["pass_rate"] == 1.0


async def test_failing_case_lowers_pass_rate():
    async def task(inp: str) -> str:
        return inp.upper() if inp == "hello" else "WRONG"

    report = await run_eval(_dataset(), task, [exact_match])
    assert report.summary()["scorers"]["exact_match"]["pass_rate"] == 0.5


async def test_task_error_is_captured_not_raised():
    async def task(inp: str) -> str:
        if inp == "world":
            raise RuntimeError("boom")
        return inp.upper()

    report = await run_eval(_dataset(), task, [exact_match])
    s = report.summary()
    assert s["errors"] == 1
    # The erroring case is excluded from the mean, not silently counted as a pass.
    assert s["scorers"]["exact_match"]["n"] == 1
    assert s["scorers"]["exact_match"]["pass_rate"] == 1.0


async def test_numeric_close_scorer():
    async def task(inp):
        return 3.14159

    ds = [EvalCase(id="pi", input=None, expected=3.14160)]
    report = await run_eval(ds, task, [numeric_close(tolerance=1e-3)])
    assert report.summary()["scorers"]["numeric_close"]["pass_rate"] == 1.0


async def test_llm_judge_scores_through_gateway(offline_settings):
    gw = Gateway(
        router=FakeRouter([make_response('{"passed": true, "score": 0.9, "reasoning": "accurate"}')]),
        settings=offline_settings,
    )
    judge = llm_judge(gw, rubric="Is the answer correct?")

    async def task(inp):
        return "an answer"

    ds = [EvalCase(id="x", input="q", expected="a")]
    report = await run_eval(ds, task, [judge])
    score = report.results[0].scores[0]
    assert score.name == "llm_judge"
    assert score.value == pytest.approx(0.9)
    assert score.passed is True


def test_compare_reports_computes_deltas():
    baseline = EvalReport(
        results=[type("R", (), {"scores": [Score("m", 0.5, False)], "error": None})()]
    )
    candidate = EvalReport(
        results=[type("R", (), {"scores": [Score("m", 0.8, True)], "error": None})()]
    )
    deltas = compare(baseline, candidate)
    assert deltas["m"]["delta"] == pytest.approx(0.3)
