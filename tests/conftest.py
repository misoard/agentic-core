"""Shared pytest fixtures for the core suite.

``FakeRouter``/``make_response`` are re-exported from the package's public testing
module (agentic_core.testing) so the core tests use the same seam a consumer
would. The suite is fully offline — nothing here touches a provider.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from agentic_core.agents.base import Agent
from agentic_core.core.config import Settings
from agentic_core.testing import FakeRouter, make_response  # re-exported for tests

__all__ = [
    "FakeRouter",
    "make_response",
    "offline_settings",
    "make_text_agent",
    "EchoIn",
    "EchoOut",
]


class EchoIn(BaseModel):
    """Minimal input contract for a demo-free test agent."""

    text: str


class EchoOut(BaseModel):
    """Minimal output contract for a demo-free test agent."""

    label: str


def make_text_agent(gateway, *, model: str | None = None) -> Agent:
    """A tiny inline agent for testing the Agent base class without the demo."""
    return Agent(
        name="echo",
        gateway=gateway,
        input_model=EchoIn,
        output_model=EchoOut,
        model=model,
        system_prompt="Return a JSON label for the text.",
        user_template="{text}",
    )


@pytest.fixture
def offline_settings() -> Settings:
    # _env_file=None so no real .env / ambient key leaks into a test run.
    return Settings(_env_file=None, openrouter_api_key="sk-test")
