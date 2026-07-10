"""guardrails — treat the model like an untrusted intern with shell access.

Input guards run before a call (or before untrusted data reaches a tool); output
guards run after the gateway returns, before data flows downstream. The package
ships reusable *primitives* with sensible defaults; which patterns/PII/policy to
enforce is the consumer's to tune (pass your own into each guard).
"""
