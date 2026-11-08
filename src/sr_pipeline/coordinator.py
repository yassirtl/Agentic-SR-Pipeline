"""Coordinator (supervisor): an LLM picks the next stage, the guardrail checks it."""

from pydantic import BaseModel, Field

from .agents.critic import describe_qa
from .agents.extraction import describe_extraction
from .agents.retrieval import describe_search
from .agents.screening import describe_screening
from .agents.synthesis import describe_synthesis
from .guardrail import (MAX_STEPS, PREREQUISITES, STAGES, StageName, check_move,
                        earliest_stale, fresh_stages, is_fresh)
from .llm import LLM, is_fatal
from .state import ReviewState

END_SIGNAL = "__end__"  # next_stage value that routes the graph to END
MAX_RESEARCHES = 2      # re-runs of search_strategy allowed after the first one

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
        # Only fields already filled: a bare 0 for a stage that has not run was read by
        # the LLM as "that stage found nothing" (first real runs, 2026-11-08).
        "Counts (filled so far): " + (", ".join(
            f"{f}={len(state.get(f, []))}" for f in LIST_FIELDS if state.get(f)) or "none"),
    ]
    search = describe_search(state)
    if search:
        lines.append(search)
    elif "search_strategy" in completed:
        lines.append("Retrieval: not run yet on the current search strategy "
                     "(record counts above are 0 because nothing was downloaded yet).")
    lines += describe_screening(state)
    lines += describe_extraction(state)
    lines += describe_synthesis(state)
    lines += describe_qa(state)
    return "\n".join(lines)


def loop_rule(proposed: str, state: ReviewState) -> str | None:
    """Code rule on the critic loop (Part 8): once the critic has judged the current
    synthesis, going back to synthesis is allowed only if its report says ``loop_back``
    (blockers found and loops left). Returns the refusal reason, or None. Without a QA
    report (placeholder critic) there is no rule.

    Part 10d: a synthesis the critic has not judged yet is not re-run either (same inputs,
    same result). Seen live (T3, --limit 500): after one critic round the coordinator ran
    synthesis four more times in a row, the critic never in between."""
    completed = state.get("completed_stages", [])
    qa = state.get("qa_report")
    if proposed != "synthesis" or not qa:
        return None
    if is_fresh("synthesis", completed) and not is_fresh("critic", completed):
        return "the critic has not judged the current synthesis yet"
    if not is_fresh("critic", completed):
        return None
    if qa.get("loop_back"):
        return None
    why = ("no blocker issues" if qa.get("passed") else
           f"loop budget spent ({qa.get('loops_used')}/{qa.get('max_loops')})")
    return f"critic loop not allowed: {why}"


def research_rule(proposed: str, state: ReviewState) -> str | None:
    """Code rule on re-searching (first real run, 2026-11-08: the coordinator read
    'unique_records=0' before retrieval as 'retrieval found nothing' and re-built the
    search 39 times). A new search is allowed only once retrieval has run on the current
    one, and at most MAX_RESEARCHES times. Returns the refusal reason, or None."""
    if proposed != "search_strategy":
        return None
    completed = state.get("completed_stages", [])
    runs = completed.count("search_strategy")
    if runs == 0:
        return None
    if not is_fresh("retrieval", completed):
        return "search strategy already built: run retrieval on it before searching again"
    ta = state.get("prisma_counts", {}).get("title_abstract") or {}
    if is_fresh("ta_screening", completed) and ta.get("included", 0) > 0:
        return (f"title/abstract screening included {ta['included']} records: "
                "no reason to search again")
    if runs > MAX_RESEARCHES:
        return f"re-search budget spent ({runs - 1}/{MAX_RESEARCHES})"
    return None


def repeated_failure(state: ReviewState) -> dict | None:
    """Part 10d: the stage that ran last failed (its audit entry has ``failed``, an error
    signature) and its previous run failed the same way: re-running it would only fail
    again. Returns the audit entry that stops the run, or None. Seen live (T3, --limit 500):
    synthesis hit the output cap five times in a row until the run was stopped by hand."""
    runs = [e for e in state.get("audit_log", []) if e.get("node") in STAGES and e.get("event")]
    if not runs or not runs[-1].get("failed"):
        return None
    last = runs[-1]
    prev = next((e for e in reversed(runs[:-1]) if e["node"] == last["node"]), None)
    if not prev or prev.get("failed") != last["failed"]:
        return None
    return {"node": "coordinator", "step": state.get("steps_taken", 0),
            "event": "stopped: repeated failure", "stage": last["node"],
            "error": last["failed"], "rounds": [prev.get("round"), last.get("round")],
            "detail": (f"{last['node']} failed twice in a row with the same error; running it "
                       "again would fail again. The run is stopped: fix the cause, then start "
                       "a new run (steps already done are read from the LLM cache).")}


RERUN_ALLOWED = ("search_strategy", "synthesis")  # own rules: research_rule, loop_rule


def rerun_rule(proposed: str, state: ReviewState) -> str | None:
    """Code rule against no-op re-runs (first real run, 2026-11-08: after full text the
    coordinator re-ran title/abstract screening "for the records not screened", which the
    screening limit leaves out by design). A stage that is still fresh has unchanged
    inputs, so re-running it gives the same result: refused. Re-searching and the critic
    loop have their own rules."""
    if proposed in RERUN_ALLOWED:
        return None
    if is_fresh(proposed, state.get("completed_stages", [])):
        return f"{proposed} is already up to date: its inputs have not changed"
    return None


def build_prompt(state: ReviewState) -> str:
    table = "\n".join(f"- {s}: needs {', '.join(PREREQUISITES[s]) or 'nothing'}" for s in STAGES)
    return (
        "You coordinate a PRISMA 2020 systematic review. Choose the next stage to run.\n"
        f"Stages and prerequisites:\n{table}\n"
        "A stage is fresh if it ran after its prerequisites. Re-running a stage makes "
        "later stages stale. The review ends after 'report'.\n"
        "If retrieval returned too few unique records to screen (fewer than the search "
        "target's minimum) or a database failed, you may run 'search_strategy' again.\n"
        "If screening included no study at all, the search may have missed the topic: you "
        "may run 'search_strategy' again (at most once for this reason).\n"
        "After 'critic': if its report says loop_back=True, run 'synthesis' again (the "
        "critic's blocker issues are given to it); otherwise run 'report'.\n\n"
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

    if (stop := repeated_failure(state)):
        return {"next_stage": END_SIGNAL, "audit_log": [stop]}

    completed = state.get("completed_stages", [])
    try:
        decision, usage = llm.call_with_usage(Decision, build_prompt(state), agent="coordinator")
    except Exception as exc:  # bad JSON, validation error, API error after retries
        if is_fatal(exc):
            raise
        # Never crash the run: take the guardrail's deterministic choice instead.
        # The step still counts, so a dead API ends the run at the budget.
        stage = earliest_stale(completed) or "report"
        entry = {"node": "coordinator", "step": steps + 1, "outcome": "llm_error",
                 "error": f"{type(exc).__name__}: {exc}", "substituted": stage}
        return {"next_stage": stage, "steps_taken": steps + 1, "audit_log": [entry]}

    stage, reason = check_move(decision.next, completed)
    if reason is None and (rule := loop_rule(stage, state) or research_rule(stage, state)
                                          or rerun_rule(stage, state)):
        stage, reason = earliest_stale(completed) or "report", rule

    entry = {"node": "coordinator", "step": steps + 1, "proposed": decision.next,
             "rationale": decision.rationale, **usage.as_dict()}
    if reason is None:
        entry["outcome"] = "accepted"
    else:
        entry.update(outcome="refused", reason=reason, substituted=stage)

    return {"next_stage": stage, "steps_taken": steps + 1, "audit_log": [entry]}
