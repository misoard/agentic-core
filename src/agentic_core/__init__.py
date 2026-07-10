"""agentic_core — a project-agnostic, framework-agnostic agentic LLM skeleton.

The whole design hangs off one idea: *every model call flows through a single
place* (the gateway). That chokepoint is where reliability, latency, and cost are
guaranteed centrally instead of being scattered across the codebase.

Layer stack (spine):  orchestration -> agent -> gateway -> router -> provider(s)
Cross-cutting:        schemas (contracts) · guardrails · observability
Offline:              eval harness

Public API
----------
This module re-exports the stable, supported surface. Import from here
(``from agentic_core import Gateway, Agent``) rather than reaching into
submodules — deep paths are internal and may move. ``eval`` is intentionally not
re-exported at top level (it would shadow the builtin); use ``agentic_core.eval``.
"""

from .agents.base import Agent
from .core.config import (
    Deployment,
    Settings,
    build_router_kwargs,
    effective_params,
    get_settings,
)
from .core.errors import (
    AgenticError,
    AllModelsFailedError,
    GuardrailError,
    MalformedOutputError,
    PermanentError,
    TransientError,
    classify_provider_error,
)
from .core.gateway import Gateway, Messages
from .core.observability import completion_span, configure, record_completion
from .core.schemas import Completion, TokenUsage, ToolCall
from .guardrails.io_guards import (
    GuardResult,
    check_injection,
    check_injection_llm,
    enforce_policy,
    scan_pii,
    validate_tool_args,
)
from .orchestration.runner import (
    State,
    run_concurrent,
    run_conditional,
    run_sequential,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # gateway + contracts
    "Gateway",
    "Messages",
    "Completion",
    "TokenUsage",
    "ToolCall",
    # config
    "Settings",
    "Deployment",
    "get_settings",
    "build_router_kwargs",
    "effective_params",
    # errors
    "AgenticError",
    "TransientError",
    "PermanentError",
    "MalformedOutputError",
    "AllModelsFailedError",
    "GuardrailError",
    "classify_provider_error",
    # observability
    "configure",
    "completion_span",
    "record_completion",
    # agents + orchestration
    "Agent",
    "State",
    "run_sequential",
    "run_concurrent",
    "run_conditional",
    # guardrails
    "GuardResult",
    "check_injection",
    "check_injection_llm",
    "validate_tool_args",
    "scan_pii",
    "enforce_policy",
]
