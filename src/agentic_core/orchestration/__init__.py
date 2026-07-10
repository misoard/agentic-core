"""orchestration — where "agentic workflow" actually lives.

Composition primitives (sequential / concurrent / conditional) over async steps,
plus a small state bag threaded between them. This layer owns *how agents combine*;
it deliberately knows nothing about what any agent does. The concrete workflow for
a project (which agents, wired which way) is the consumer's — see demo/workflow.py.
"""
