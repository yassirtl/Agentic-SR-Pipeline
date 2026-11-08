"""Part 2 skeleton: supervisor loop + guardrail, with a scripted fake LLM."""

from sr_pipeline.coordinator import summarize_state
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import MAX_STEPS, STAGES, check_move, earliest_stale, is_fresh
from sr_pipeline.llm import ScriptedLLM


def coord(*stages: str) -> ScriptedLLM:
    return ScriptedLLM({"coordinator": [{"next": s, "rationale": f"go {s}"} for s in stages]})


def run(llm, thread_id="t1", **kw):
    graph = build_graph(llm, **kw)
    final = graph.invoke({"question": "Does X improve Y?"}, run_config(thread_id))
    return graph, final


def coordinator_entries(final):
    return [e for e in final["audit_log"] if e["node"] == "coordinator"]


# --- guardrail unit checks -------------------------------------------------

def test_freshness_rule():
    done = list(STAGES[:4])  # planner .. ta_screening
    assert all(is_fresh(s, done) for s in done)
    # Re-running retrieval makes screening stale, but not the stages before it.
    rerun = done + ["retrieval"]
    assert is_fresh("retrieval", rerun)
    assert not is_fresh("ta_screening", rerun)
    assert earliest_stale(rerun) == "ta_screening"


def test_unknown_stage_is_refused():
    stage, reason = check_move("teleport", ["planner"])
    assert stage == "search_strategy"
    assert "unknown" in reason


# --- (a) happy path --------------------------------------------------------

def test_happy_path_runs_prisma_order_and_ends_after_report():
    llm = coord(*STAGES)
    graph, final = run(llm)

    assert final["completed_stages"] == list(STAGES)
    assert final["steps_taken"] == len(STAGES)
    assert all(e["outcome"] == "accepted" for e in coordinator_entries(final))
    assert llm.script["coordinator"] == []  # every scripted answer was used, no extra call

    # Checkpointer: the final state is stored under the thread id and nothing is pending.
    snapshot = graph.get_state(run_config("t1"))
    assert snapshot.values["completed_stages"] == list(STAGES)
    assert snapshot.next == ()


# --- (b) cheating coordinator ----------------------------------------------

def test_cheating_coordinator_is_overruled():
    # Proposes extraction right after planner, then behaves.
    llm = coord("planner", "extraction", *STAGES[2:])
    _, final = run(llm)

    assert final["completed_stages"] == list(STAGES)
    refusal = coordinator_entries(final)[1]
    assert refusal["proposed"] == "extraction"
    assert refusal["outcome"] == "refused"
    assert refusal["substituted"] == "search_strategy"
    assert "ft_screening" in refusal["reason"]


# --- (c) critic loop -------------------------------------------------------

def test_critic_loop_reruns_critic_before_report():
    # Full chain up to critic, back to synthesis, then report too early, then report.
    llm = coord(*STAGES[:-1], "synthesis", "report", "report")
    _, final = run(llm)

    assert final["completed_stages"] == list(STAGES[:-1]) + ["synthesis", "critic", "report"]
    assert final["completed_stages"].count("critic") == 2

    entries = coordinator_entries(final)
    early = entries[-2]
    assert early["proposed"] == "report"
    assert early["outcome"] == "refused"
    assert early["substituted"] == "critic"
    assert entries[-1]["outcome"] == "accepted"


# --- (d) step budget ---------------------------------------------------------

def test_step_budget_ends_the_run():
    # A coordinator stuck in a loop: always proposes planner (always legal).
    llm = coord(*["planner"] * (MAX_STEPS + 5))
    _, final = run(llm)  # default MAX_STEPS and recursion_limit must fit together

    assert final["steps_taken"] == MAX_STEPS
    assert final["completed_stages"] == ["planner"] * MAX_STEPS
    assert "report" not in final["completed_stages"]
    assert final["audit_log"][-1]["event"] == "budget exhausted"
    assert len(llm.calls) == MAX_STEPS  # no LLM call once the budget is spent


# --- coordinator input -----------------------------------------------------

def test_coordinator_sees_summary_not_raw_records():
    state = {"question": "Q", "completed_stages": ["planner"],
             "records": [{"title": "SECRET TITLE"}] * 3}
    text = summarize_state(state)
    assert "records=3" in text
    assert "SECRET TITLE" not in text


# --- re-search rule (first real run, 2026-11-08) ------------------------------

def test_search_strategy_cannot_repeat_before_retrieval():
    # The real coordinator re-built the search 39 times without ever retrieving.
    llm = coord("planner", "search_strategy", "search_strategy", *STAGES[3:])
    _, final = run(llm)

    assert final["completed_stages"] == list(STAGES)
    refusal = coordinator_entries(final)[2]
    assert refusal["proposed"] == "search_strategy"
    assert refusal["outcome"] == "refused"
    assert refusal["substituted"] == "retrieval"
    assert "run retrieval" in refusal["reason"]


def test_research_budget_is_capped():
    from sr_pipeline.coordinator import MAX_RESEARCHES, research_rule
    done = ["planner"] + ["search_strategy", "retrieval"] * (MAX_RESEARCHES + 1)
    assert "budget spent" in research_rule("search_strategy", {"completed_stages": done})
    allowed = ["planner"] + ["search_strategy", "retrieval"] * MAX_RESEARCHES
    assert research_rule("search_strategy", {"completed_stages": allowed}) is None


def test_summary_says_retrieval_not_run_yet():
    text = summarize_state({"question": "Q", "completed_stages": ["planner", "search_strategy"]})
    assert "Retrieval: not run yet" in text
