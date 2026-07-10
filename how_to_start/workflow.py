"""how_to_start/workflow.py — the minimal end-to-end: sentiment through the gateway.

The smallest thing that shows the whole spine wired up: import agentic_core, build
this example's gateway (its own registry + key), run a typed agent, get a validated
result back. One agent, one call — just the shape. A richer two-agent concurrent
workflow would slot in here using agentic_core's ``run_concurrent``.

Run it live (needs OPENROUTER_API_KEY in your env), from the repo root:
    uv run python -m how_to_start.workflow
"""

from __future__ import annotations

import asyncio

from agentic_core import Gateway

from .agents import Sentiment, TextIn, sentiment_agent
from .config import build_gateway


async def classify(text: str, *, gateway: Gateway | None = None) -> Sentiment:
    """Classify one text's sentiment through this project's gateway.

    ``gateway`` is injectable so tests can pass one backed by a fake Router
    (offline); in production it defaults to ``build_gateway()`` — this project's
    real one call path.
    """
    gateway = gateway or build_gateway()
    agent = sentiment_agent(gateway)
    result = await agent.run(TextIn(text=text))
    return result.parsed


async def main() -> None:
    text = "I absolutely love this new scheduling assistant!"
    sentiment = await classify(text)
    print(f"{text!r}\n  -> {sentiment.label} (confidence {sentiment.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
