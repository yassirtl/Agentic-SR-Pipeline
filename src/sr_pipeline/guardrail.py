"""Code guardrail that enforces PRISMA prerequisites on the Coordinator's moves.

The Coordinator (an LLM) proposes the next stage; this module has the last word.

Freshness rule: a stage is fresh if
  1. it has run, and
  2. each prerequisite is itself fresh, and
  3. its last run came after the last run of each prerequisite.
The "clock" is the position of a stage's last occurrence in ``completed_stages``.
So re-running retrieval makes screening (and everything after it) stale.

A move is legal only if all its prerequisites are fresh. Otherwise the guardrail
substitutes the earliest non-fresh stage in PRISMA order (never re-asks the LLM).
"""

from typing import Literal, get_args

StageName = Literal[
    "planner",
    "search_strategy",
    "retrieval",
    "ta_screening",
    "ft_screening",
    "extraction",
    "synthesis",
    "critic",
    "report",
]

STAGES: tuple[str, ...] = get_args(StageName)  # PRISMA order

PREREQUISITES: dict[str, list[str]] = {
    "planner": [],
    "search_strategy": ["planner"],
    "retrieval": ["search_strategy"],
    "ta_screening": ["retrieval"],
    "ft_screening": ["ta_screening"],
    "extraction": ["ft_screening"],
    "synthesis": ["extraction"],
    "critic": ["synthesis"],
    "report": ["critic"],
}

MAX_STEPS = 40


def _last_run(stage: str, completed: list[str]) -> int:
    """Index of the last run of ``stage`` in ``completed``, or -1 if it never ran."""
    for i in range(len(completed) - 1, -1, -1):
        if completed[i] == stage:
            return i
    return -1


def is_fresh(stage: str, completed: list[str]) -> bool:
    last = _last_run(stage, completed)
    if last < 0:
        return False
    return all(
        is_fresh(p, completed) and last > _last_run(p, completed)
        for p in PREREQUISITES[stage]
    )


def fresh_stages(completed: list[str]) -> list[str]:
    return [s for s in STAGES if is_fresh(s, completed)]


def earliest_stale(completed: list[str]) -> str | None:
    """First stage in PRISMA order that is not fresh (None if all are fresh)."""
    for s in STAGES:
        if not is_fresh(s, completed):
            return s
    return None


def check_move(proposed: str, completed: list[str]) -> tuple[str, str | None]:
    """Validate a proposed move.

    Returns ``(stage_to_run, refusal_reason)``. ``refusal_reason`` is None when the
    proposal is accepted as is.
    """
    if proposed not in PREREQUISITES:
        reason = f"unknown stage '{proposed}'"
    else:
        missing = [p for p in PREREQUISITES[proposed] if not is_fresh(p, completed)]
        if not missing:
            return proposed, None
        reason = f"prerequisites not fresh: {', '.join(missing)}"

    substitute = earliest_stale(completed)
    if substitute is None:
        # Everything is fresh, only possible after a full run; the graph ends after
        # report, so this is a safety net: finishing the review is always legal.
        substitute = "report"
    return substitute, reason
