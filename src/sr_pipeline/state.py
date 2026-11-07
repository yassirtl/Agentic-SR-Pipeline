"""Shared state of a review run.

Every node receives the whole state and returns only the fields it changes.
LangGraph merges that partial update into the state:
- plain fields are overwritten;
- fields annotated with ``operator.add`` are append-only lists: the node returns
  the new items and LangGraph concatenates them (old + new). Lists that several
  nodes write (decisions, audit log, completed stages) must never be overwritten.
"""

import operator
from typing import Annotated, TypedDict


class ReviewState(TypedDict, total=False):
    question: str
    protocol: dict
    search_strategy: dict

    records: Annotated[list, operator.add]
    unique_records: list

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
