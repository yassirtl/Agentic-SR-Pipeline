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
"""

from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents.planner import protocol_approval
from .agents.screening import make_conflict_review
from .coordinator import END_SIGNAL, coordinator_step
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


def route(state: ReviewState) -> str:
    return state["next_stage"]


AFTER = {"ta_screening": "ta_conflicts", "ft_screening": "ft_conflicts"}


def build_graph(llm: LLM, *, max_steps: int = MAX_STEPS, stages: dict | None = None,
                approve_protocol: bool = False, human_screening: bool = False,
                checkpointer=None):
    """Compile the review graph.

    ``stages`` maps a stage name to its node function; missing stages get a
    placeholder (later parts plug the real agents in here).

    ``approve_protocol=True`` inserts the human approval node after the planner
    (planner -> protocol_approval -> coordinator). It is not a stage: the
    guardrail never sees it, and the coordinator cannot skip it. The CLI's
    ``--auto`` flag maps to ``approve_protocol=False`` and ``human_screening=False``.
    """
    stages = stages or {}
    g = StateGraph(ReviewState)
    g.add_node("coordinator", partial(coordinator_step, llm=llm, max_steps=max_steps))
    for name in STAGES:
        g.add_node(name, stages.get(name, placeholder_stage(name)))

    g.add_edge(START, "coordinator")
    g.add_conditional_edges("coordinator", route, {**{s: s for s in STAGES}, END_SIGNAL: END})
    for name in STAGES:
        if name == "planner" and approve_protocol:
            continue
        if name in AFTER:
            g.add_node(AFTER[name], make_conflict_review(name[:2], human=human_screening))
            g.add_edge(name, AFTER[name])
            g.add_edge(AFTER[name], "coordinator")
            continue
        g.add_edge(name, END if name == "report" else "coordinator")

    if approve_protocol:
        g.add_node("protocol_approval", protocol_approval)
        g.add_edge("planner", "protocol_approval")
        g.add_edge("protocol_approval", "coordinator")

    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_config(thread_id: str) -> dict:
    """Invoke config: the thread id names the run in the checkpointer."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
