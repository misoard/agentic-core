"""orchestration/runner.py — composition primitives.

Why this module exists
----------------------
Agents are single, narrow units. The value of a multi-agent system comes from
*how* they combine — run in order, run at once, branch on a result. That
combination logic is this layer, and keeping it a thin thing you own (rather than
a framework) is a deliberate SPEC choice: orchestration stays legible and swappable.

What it is (and isn't)
----------------------
Pure, agent-agnostic combinators over async *steps*. A step is just
``async (state) -> result``. This module imports no ``Agent`` and knows nothing
about the gateway: a workflow adapts its agents into steps (``lambda s: agent.run(...)``)
and hands them here. That's the mechanism/policy split — the primitives are the
package's, the specific wiring is the consumer's (demo/workflow.py).

Steps share a small mutable ``State`` bag, but the combinators are generic over
the state type: a workflow can pass a Pydantic model, a dataclass, or the ``State``
here — whatever it likes. The runner only threads it through.

On memory: agents are stateless (they take ``history`` per call and retain
nothing). If a workflow needs multi-turn memory, it keeps the history list in its
state and passes it into each agent step — the runner stays agnostic to that.

Interview line: "My interpreter and parser ran concurrently here. If a workflow
needed durable state or human-in-the-loop pause/resume, this is the layer I'd swap
for LangGraph rather than hand-rolling checkpointing."
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence, TypeVar

StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")

# A unit of orchestrated work: given the shared state, do something async.
Step = Callable[[StateT], Awaitable[ResultT]]


@dataclass
class State:
    """A tiny mutable bag threaded between steps. The workflow owns its contents.

    Convenience only — the combinators accept ANY state object (they just hand it
    to each step), so a workflow may pass its own typed model instead. Kept dict-
    backed and unopinionated so the primitive imposes no schema.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


async def run_sequential(
    state: StateT, steps: Sequence[Step[StateT, Any]]
) -> list[Any]:
    """Run steps in order over the same state; return each step's result.

    Order is the guarantee: step N sees whatever step N-1 wrote to ``state``. Use
    this when a later step depends on an earlier one's output.
    """
    results: list[Any] = []
    for step in steps:
        results.append(await step(state))
    return results


async def run_concurrent(
    state: StateT,
    steps: Sequence[Step[StateT, Any]],
    *,
    return_exceptions: bool = False,
) -> list[Any]:
    """Run steps concurrently (``asyncio.gather``); return results in step order.

    This is where concurrency pays off: independent agents run at once, so
    wall-clock is the slowest step, not the sum. Treat ``state`` as read-only in
    concurrent steps and merge their *returned* results afterward — having several
    steps write the shared bag at once would race.

    ``return_exceptions=False`` (default) fails fast: the first step error
    propagates. Pass True to collect exceptions alongside results instead.
    """
    return await asyncio.gather(
        *(step(state) for step in steps), return_exceptions=return_exceptions
    )


async def run_conditional(
    state: StateT,
    predicate: Callable[[StateT], bool],
    if_true: Step[StateT, ResultT],
    if_false: Step[StateT, ResultT] | None = None,
) -> ResultT | None:
    """Branch on the state: run ``if_true`` when ``predicate(state)`` is truthy,
    else ``if_false``. Returns the chosen branch's result, or None when the
    unchosen branch is absent. This is the seam for routing / early-exit logic.
    """
    branch = if_true if predicate(state) else if_false
    return await branch(state) if branch is not None else None
