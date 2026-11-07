"""Shared state of a review run.

Every node receives the whole state and returns only the fields it changes.
LangGraph merges that partial update into the state:
- plain fields are overwritten;
- fields annotated with ``operator.add`` are append-only lists: the node returns
  the new items and LangGraph concatenates them (old + new). Lists that several
  nodes write (decisions, audit log, completed stages) must never be overwritten.
"""

import operator
from typing import Annotated, Any, TypedDict


class ReviewState(TypedDict, total=False):
    question: str
    protocol: dict
    search_strategy: dict

    records: Annotated[list, operator.add]   # every downloaded record, tagged with its round
    unique_records: list                     # deduplicated papers of the latest retrieval
    retrieval: dict                          # merge log of the latest retrieval (duplicates, conflicts)

    ta_decisions: Annotated[list, operator.add]
    ft_decisions: Annotated[list, operator.add]
    included: list

    extractions: Annotated[list, operator.add]
    risk_of_bias: Annotated[list, operator.add]

    synthesis: dict
    qa_report: dict
    prisma_counts: dict

    # Order matters: the guardrail uses positions in this list as a clock.
    completed_stages: Annotated[list, operator.add]
    audit_log: Annotated[list, operator.add]

    next_stage: str
    steps_taken: int


# --- Rounds: no duplicate decisions after a loop-back -----------------------
# Decision and extraction lists are append-only, so re-running a stage adds a
# second item for the same paper. Every item carries ``paper_id`` and ``round``;
# the full history stays in the state (audit trail), but stages that *use*
# decisions (screening, extraction, synthesis) must read them through latest().

def stage_round(stage: str, completed: list[str]) -> int:
    """Round number of the run about to happen (1 for the first run of ``stage``)."""
    return completed.count(stage) + 1


def latest(items: list[dict], key: str = "paper_id") -> list[dict]:
    """Keep only the most recent round per ``key``.

    Ties within the same round keep the item written last. Output order follows
    each key's first appearance, so results are deterministic.
    """
    best: dict[Any, dict] = {}
    for item in items:
        k = item[key]
        if k not in best or item.get("round", 0) >= best[k].get("round", 0):
            best[k] = item
    return list(best.values())
