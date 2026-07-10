"""agents — the composable unit of LLM work.

An agent is a typed, validated, single-responsibility call: prompt template +
input schema + output schema, executed through the gateway. Narrow agents beat
one giant prompt for reliability, and splitting a problem across focused agents
is the core agentic move. Orchestration (next layer up) composes them.
"""
