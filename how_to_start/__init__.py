"""how_to_start — a runnable toy example of consuming agentic_core.

NOT part of the installed package (it's excluded from the wheel — the package is
only ``src/agentic_core``). It lives in the repo purely as a starting point: it
shows the recommended file split a real project would use —

    config.py    your models (env-driven) + build_gateway()   (policy)
    agents.py    your typed agents + prompts loader
    prompts/     versioned prompt files
    workflow.py  the end-to-end entry point

Copy this folder into your own project and adapt it. See the repo README's
"Add it to your project" section for the from-scratch setup.
"""
