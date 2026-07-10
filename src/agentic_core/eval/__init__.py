"""eval — the offline measurement loop most people skip.

"Did my change make it better?" must be measurable, not vibes. This layer runs a
fixed dataset through a task (usually an agent), scores each result with
deterministic metrics and/or an LLM-as-judge, and produces a report you can diff
across changes to catch regressions. Datasets, metrics, and rubrics are the
consumer's; the harness and scorer primitives are the package's.
"""
