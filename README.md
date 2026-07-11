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

## Add it to your project

`agentic-core` is a dependency you add to *your* repo — you don't clone it. From
scratch (until it's on PyPI, install from Git):

```bash
mkdir my-agent-app && cd my-agent-app
uv init                                  # creates pyproject.toml + .venv
uv add "agentic-core @ git+https://github.com/<your-username>/agentic-core.git"
#   pin a release:  uv add "agentic-core @ git+https://github.com/<your-username>/agentic-core.git@v0.1.0"
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

`uv add` edits your `pyproject.toml`, writes `uv.lock`, and installs into `.venv`
— you never hand-edit them. Then `import agentic_core` (see below) and run with
`uv run python -m app.workflow`. Prefer pip? `pip install "agentic-core @ git+https://github.com/<your-username>/agentic-core.git"`.

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

**Start from a working example.** [`how_to_start/`](how_to_start/) is a runnable
toy showing the recommended file split you'd copy into your own project:
`config.py` (your models + `build_gateway`) · `agents.py` (typed agents) ·
`prompts/` (versioned prompts) · `workflow.py` (the entry point). It lives in this
repo but is **not** shipped in the package. Run it from the repo root:
`uv run python -m how_to_start.workflow` (needs `OPENROUTER_API_KEY`).

## What's here

```
src/agentic_core/                the PACKAGE (shipped in the wheel)
  core/         errors · schemas · config · gateway · observability
  agents/       base.py          (the Agent abstraction)
  orchestration/runner.py        (composition primitives)
  guardrails/   io_guards.py      (injection · tool-arg · PII · policy)
  eval/         harness.py        (dataset → run → score → compare)
  testing.py    FakeRouter + make_response (shipped test seam)

how_to_start/                    toy example (in the repo, NOT in the wheel):
  config.py · agents.py · prompts/ · workflow.py
```

### Module guide — the one thing to know about each file

| File | What it is / the one thing to know |
|---|---|
| `core/errors.py` | The failure taxonomy. **Transient → Router's job; malformed → gateway re-prompts; guard/permanent → fail-closed.** This classification *is* the reliability. |
| `core/schemas.py` | The typed carriers crossing every layer: `Completion` (generic over its parsed type), `TokenUsage`, `ToolCall`. |
| `core/config.py` | Models & secrets as data: `Settings`, `Deployment`, the registry, and `build_router_kwargs` (the one seam to the Router). Keys resolve by the `<provider>_api_key` convention. |
| `core/gateway.py` | **The keystone.** Every call goes through `complete()`: output validation + re-prompt, cost/latency/token capture, terminal error classification. No transient-retry loop (that's the Router's). |
| `core/observability.py` | One OTel span per call (latency, tokens, cost, model, errors). No-op until you `configure()` it — so tests stay offline. |
| `agents/base.py` | The `Agent`: prompt template + typed input/output, run through the gateway. Stateless; the caller owns any `history`. |
| `orchestration/runner.py` | Composition primitives over async steps: `run_sequential`, `run_concurrent`, `run_conditional`, and a small `State`. Agent-agnostic. |
| `guardrails/io_guards.py` | Input/output safety: injection check, tool-arg validation, PII scan, policy. Heuristic first line, `GuardResult` you fail-closed on. |
| `eval/harness.py` | The offline loop: dataset → run → score (deterministic + LLM-judge) → `compare` two runs for regressions. |
| `testing.py` | Shipped test seam: `FakeRouter` + `make_response` to test agents/workflows without a provider. |

**Deeper written notes:** [`orchestration_notes.pdf`](orchestration_notes.pdf) —
the longer-form design notes behind this pipeline (not part of the package; repo
reference only).

Frameworks deliberately **not** adopted (but swappable in at a clean seam):
LangGraph (durable state), Pydantic AI (typed-agent ergonomics), CrewAI, provider
Agents SDKs. Protocols to know by name: **MCP** (agent↔tools), **A2A** (agent↔agent).

## Develop this repo

Only if you're working *on* the package itself (contributing), not just using it:

```bash
uv sync --extra dev      # dev + runtime deps into .venv
uv run pytest -q         # fully offline, no API key needed
```

## License

[MIT](LICENSE) © 2026 Mathieu Isoard.
