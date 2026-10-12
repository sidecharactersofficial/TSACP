"""Shared guard for experiment modules that are declared but not implemented.

These placeholders exist so the experiment-ID space (exp1-exp24) is complete and
discoverable. They are listed in ``doc/OPEN_SCIENCE.md`` as *not independently
reproducible*: the repository ships no source or archived output for them.

Running one raises ``NotImplementedError`` (exit code 2) rather than silently
doing nothing, so orchestration cannot mistake a no-op for a successful run.
"""
import sys


def not_implemented(experiment_id, claim, message):
    print(
        f"[{experiment_id}] NOT IMPLEMENTED — claim {claim}.\n"
        f"  {message}\n"
        f"  This placeholder has no implementation and no archived result in this\n"
        f"  artifact. See doc/OPEN_SCIENCE.md for the full experiment inventory.\n",
        file=sys.stderr,
    )
    raise SystemExit(2)
