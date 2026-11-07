"""Coordinator (supervisor): an LLM picks the next stage, the guardrail checks it."""

from pydantic import BaseModel, Field

from .guardrail import MAX_STEPS, PREREQUISITES, STAGES, StageName, check_move, fresh_stages
from .llm import LLM
from .state import ReviewState

END_SIGNAL = "__end__"  # next_stage value that routes the graph to END

LIST_FIELDS = ("records", "unique_records", "ta_decisions", "ft_decisions",
               "included", "extractions", "risk_of_bias")


class Decision(BaseModel):
    next: StageName = Field(description="The stage to run next.")
    rationale: str = Field(description="One or two sentences explaining the choice.")


def summarize_state(state: ReviewState) -> str:
    """Compact view of the state for the Coordinator: counts and flags, no raw records."""
    completed = state.get("completed_stages", [])
    lines = [
        f"Question: {state.get('question', '')}",
        f"Steps taken: {state.get('steps_taken', 0)}",
        f"Stages run (in order): {', '.join(completed) or 'none'}",
        f"Fresh stages: {', '.join(fresh_stages(completed)) or 'none'}",
        "Counts: " + ", ".join(f"{f}={len(state.get(f, []))}" for f in LIST_FIELDS),
    ]
    issues = state.get("qa_report", {}).get("issues", [])
    if issues:
        lines.append("Last critic issues: " + "; ".join(map(str, issues)))
    return "\n".join(lines)


def build_prompt(state: ReviewState) -> str:
    table = "\n".join(f"- {s}: needs {', '.join(PREREQUISITES[s]) or 'nothing'}" for s in STAGES)
    return (
        "You coordinate a PRISMA 2020 systematic review. Choose the next stage to run.\n"
        f"Stages and prerequisites:\n{table}\n"
        "A stage is fresh if it ran after its prerequisites. Re-running a stage makes "
        "later stages stale. The review ends after 'report'.\n\n"
        f"Current state:\n{summarize_state(state)}"
    )


def coordinator_step(state: ReviewState, llm: LLM, max_steps: int = MAX_STEPS) -> dict:
    steps = state.get("steps_taken", 0)
    if steps >= max_steps:
        return {
            "next_stage": END_SIGNAL,
            "audit_log": [{"node": "coordinator", "step": steps, "event": "budget exhausted",
                           "max_steps": max_steps}],
        }

    decision = llm.structured_call(Decision, build_prompt(state), agent="coordinator")
    stage, reason = check_move(decision.next, state.get("completed_stages", []))

    entry = {"node": "coordinator", "step": steps + 1, "proposed": decision.next,
             "rationale": decision.rationale}
    if reason is None:
        entry["outcome"] = "accepted"
    else:
        entry.update(outcome="refused", reason=reason, substituted=stage)

    return {"next_stage": stage, "steps_taken": steps + 1, "audit_log": [entry]}
