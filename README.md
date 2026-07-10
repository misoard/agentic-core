# agentic-core

A minimal, opinionated, **framework-agnostic** skeleton for production LLM / agent
systems. A thin core *you own*, sitting on stable industry standards (Pydantic,
LiteLLM, OpenTelemetry, tenacity).

**The one idea everything hangs off:** *every model call flows through a single
place* (the gateway). That one chokepoint is what lets reliability, latency, and
cost be guaranteed **centrally** instead of scattered across the codebase.

## The layer stack

```
   request ─▶  Orchestration ─▶  Agent(s) ─▶  Gateway ─▶  Router ─▶  Provider(s)
                                                 │
                                                 ├─ retry / timeout / fallback
                                                 ├─ structured-output validation + re-prompt
                                                 └─ cost / latency / trace capture

   cross-cutting:  Schemas (contracts) · Guardrails (in/out) · Observability
   offline:        Eval harness (dataset → run → score → compare)
```

### What each layer owns — and what it deliberately doesn't know

| Layer | Owns | Deliberately does NOT know |
|---|---|---|
| **Orchestration** (`orchestration/runner.py`) | composing agents: sequential / concurrent / conditional, over a small `State` | what any agent *does*; how a model is called |
| **Agent** (`agents/base.py`) | a typed, validated, single-responsibility call: prompt template + input/output schema, run through the gateway | retry, fallback, cost, tracing — all the gateway's; which provider serves it |
| **Gateway** (`core/gateway.py`) — *the keystone* | the one call path: output validation + **re-prompt** on malformed, cost/latency/token capture, terminal error classification | business logic; *transient* retries (those are the Router's) |
| **Router** (LiteLLM, configured in `core/config.py`) | transient retry (timeout/rate-limit/5xx), fallback across models, rate limits | your Pydantic schema; your prompts |
| **Provider(s)** | inference | everything above |

Cross-cutting: **Schemas** (`core/schemas.py`) typed I/O contracts · **Guardrails**
(`guardrails/io_guards.py`) input/output safety · **Observability**
(`core/observability.py`) one OTel span per call. Offline: **Eval**
(`eval/harness.py`) the regression loop.

### The reliability rule (read this once)

Two retry-shaped jobs, each owned by exactly one layer so they compose instead of
multiply:

- **Transient network failure** → the **Router**'s job. The gateway has *no*
  transient-retry loop. When the Router gives up (retries + fallbacks exhausted),
  that's terminal → `AllModelsFailedError`, surfaced, never retried.
- **Malformed output** (reply doesn't satisfy your Pydantic schema) → the
  **gateway**'s job, only. It re-prompts with the bad answer + validation error.
  This is the single retry-like loop in the gateway (driven by tenacity).

## Install

```bash
uv sync --extra dev      # dev + runtime deps into .venv
uv run pytest -q         # fully offline, no API key needed
```

## Use it in your project

The core is machinery + types; your repo supplies the *values* (models, keys,
prompts, workflow). Nothing here holds a secret or hardcodes a provider.

```python
from agentic_core import Agent, Gateway, Settings, Deployment

# your registry (models are config, not code) — slugs are LiteLLM strings
REGISTRY = {"fast": Deployment(alias="fast", model="openrouter/openai/gpt-4o-mini")}
gateway = Gateway(registry=REGISTRY, fallbacks={})   # reads OPENROUTER_API_KEY from your .env

class In(BaseModel): text: str
class Out(BaseModel): summary: str

agent = Agent(name="summarize", gateway=gateway, input_model=In, output_model=Out,
              model="fast", system_prompt="Summarize the text.", user_template="{text}")

result = await agent.run({"text": "..."})   # result.parsed is a validated Out
```

- **Keys / models via env**: subclass `Settings` with your own `<provider>_api_key`
  and model-name fields; the package resolves keys by the `<provider>_api_key`
  convention (no core change to add a provider).
- **Observability**: `from agentic_core import configure; configure()` at startup —
  wires Logfire, Jaeger (OTLP), or local console per your `.env`; no-op otherwise.
- **Testing offline**: `from agentic_core.testing import FakeRouter, make_response`
  and inject the fake Router — the same seam the core's own tests use.

## What's here

```
src/agentic_core/
  core/         errors · schemas · config · gateway · observability
  agents/       base.py          (the Agent abstraction)
  orchestration/runner.py        (composition primitives)
  guardrails/   io_guards.py      (injection · tool-arg · PII · policy)
  eval/         harness.py        (dataset → run → score → compare)
  testing.py    FakeRouter + make_response (shipped test seam)
```

Frameworks deliberately **not** adopted (but swappable in at a clean seam):
LangGraph (durable state), Pydantic AI (typed-agent ergonomics), CrewAI, provider
Agents SDKs. Protocols to know by name: **MCP** (agent↔tools), **A2A** (agent↔agent).

## License

[MIT](LICENSE) © 2026 Mathieu Isoard.
