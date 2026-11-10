"""LangGraph wiring: a supervisor loop around the Coordinator.

START -> coordinator -> (stage chosen by coordinator + guardrail) -> coordinator -> ...
report -> END. The coordinator can also route to END when the step budget is spent.

Fixed edges that are not stages (the guardrail never sees them, the coordinator
cannot skip them):
  planner -> protocol_approval -> coordinator          (if approve_protocol)
  ta_screening -> ta_conflicts -> coordinator          (always)
  ft_screening -> ft_conflicts -> coordinator          (always)
The conflict nodes pause for a human when ``human_screening`` is True, and
include unresolved records otherwise (``--auto``); with nothing unresolved they
do nothing.

Part 11b: every node is wrapped by ``timed``: the audit entries it returns get
``started_at`` / ``ended_at`` (UTC) and ``duration_s``, and ``embed_cost_usd`` when the node
paid for embeddings, so the UI can show time and spend per stage while the run is going.
A budget cap (``budget_usd`` in ``state['run']``) pauses the run at a stage boundary:
coordinator -> budget_pause (interrupt) -> coordinator.
"""

import time
from datetime import datetime, timezone
from functools import partial, wraps

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents.planner import protocol_approval
from .agents.screening import make_conflict_review
from .coordinator import BUDGET_SIGNAL, END_SIGNAL, budget_pause, coordinator_step
from .guardrail import MAX_STEPS, STAGES
from .llm import LLM
from .state import ReviewState

RECURSION_LIMIT = 100  # each step = 2 supersteps (coordinator + stage); default 25 is too low


def placeholder_stage(name: str):
    """Stand-in for a real agent: only records that the stage ran."""

    def node(state: ReviewState) -> dict:
        return {
            "completed_stages": [name],
            "audit_log": [{"node": name, "step": state.get("steps_taken", 0),
                           "event": "placeholder run"}],
        }

    node.__name__ = name
    return node


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _embed_cost(embedder) -> float:
    try:
        return (embedder.usage().get("cost_usd") or 0.0) if embedder is not None else 0.0
    except Exception:  # pragma: no cover - a meter must never break a run
        return 0.0


def timed(fn, embedder=None):
    """Stamp the audit entries a node returns with its start / end time (Part 11b).

    Exceptions pass through untouched: ``interrupt()`` works by raising, and a resumed node
    runs again from the top, so its ``started_at`` is the resume time (time spent waiting for
    a human is not counted). Fields a node already wrote are kept (``setdefault``)."""

    @wraps(fn)
    def node(state):
        started, t0, e0 = _now(), time.monotonic(), _embed_cost(embedder)
        update = fn(state)
        entries = (update or {}).get("audit_log") if isinstance(update, dict) else None
        if entries:
            stamp = {"started_at": started, "ended_at": _now(),
                     "duration_s": round(time.monotonic() - t0, 3)}
            spent = _embed_cost(embedder) - e0
            if spent > 0:
                stamp["embed_cost_usd"] = round(spent, 8)
            update = {**update, "audit_log": [{**stamp, **e} if isinstance(e, dict) else e
                                              for e in entries]}
        return update

    return node


def route(state: ReviewState) -> str:
    return state["next_stage"]


AFTER = {"ta_screening": "ta_conflicts", "ft_screening": "ft_conflicts"}


def build_graph(llm: LLM, *, max_steps: int = MAX_STEPS, stages: dict | None = None,
                approve_protocol: bool = False, human_screening: bool = False,
                checkpointer=None, embedder=None):
    """Compile the review graph.

    ``stages`` maps a stage name to its node function; missing stages get a
    placeholder (later parts plug the real agents in here).

    ``approve_protocol=True`` inserts the human approval node after the planner
    (planner -> protocol_approval -> coordinator). It is not a stage: the
    guardrail never sees it, and the coordinator cannot skip it. The CLI's
    ``--auto`` flag maps to ``approve_protocol=False`` and ``human_screening=False``.

    ``embedder``: only read, to stamp each node's embedding spend in its audit entries.
    """
    stages = stages or {}
    g = StateGraph(ReviewState)

    def add(name, fn):
        g.add_node(name, timed(fn, embedder))

    add("coordinator", partial(coordinator_step, llm=llm, max_steps=max_steps))
    add("budget_pause", budget_pause)
    for name in STAGES:
        add(name, stages.get(name, placeholder_stage(name)))

    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", route, {**{s: s for s in STAGES}, END_SIGNAL: END,
                                                   BUDGET_SIGNAL: "budget_pause"})
    g.add_edge("budget_pause", "coordinator")
    for name in STAGES:
        if name == "planner" and approve_protocol:
            continue
        if name in AFTER:
            add(AFTER[name], make_conflict_review(name[:2], human=human_screening))
            g.add_edge(name, AFTER[name])
            g.add_edge(AFTER[name], "coordinator")
            continue
        g.add_edge(name, END if name == "report" else "coordinator")

    if approve_protocol:
        add("protocol_approval", protocol_approval)
        g.add_edge("planner", "protocol_approval")
        g.add_edge("protocol_approval", "coordinator")

    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_config(thread_id: str) -> dict:
    """Invoke config: the thread id names the run in the checkpointer."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
