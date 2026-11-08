"""Part 10d: synthesis output cap, stop on a repeated stage failure, D quote check without
a design list (from the --limit 500 runs T1 183527 and T3 183528)."""

from sr_pipeline.agents.screening import check_verdicts, screen_record, ta_evidence
from sr_pipeline.agents.synthesis import MANY_STUDIES, many_studies_rule, narrative_prompt
from sr_pipeline.coordinator import coordinator_step, loop_rule, repeated_failure
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import (DEFAULT_MAX_TOKENS, OpenAILLM, ScriptedLLM, error_signature,
                             max_tokens_for)

from test_fixes import Answer, FakeChat

LENGTH = ("LengthFinishReasonError: Could not parse response content as the length limit "
          "was reached - CompletionUsage(completion_tokens={}, prompt_tokens={})")


# --- (1) synthesis output cap ----------------------------------------------------------------

def test_synthesis_gets_a_larger_output_cap(tmp_path):
    made = []

    def factory(model, max_tokens=DEFAULT_MAX_TOKENS):
        made.append((model, max_tokens))
        return FakeChat(model)

    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=factory)
    llm.structured_call(Answer, "p1", agent="screener_a")
    llm.structured_call(Answer, "p2", agent="synthesis")
    llm.structured_call(Answer, "p3", agent="synthesis")  # same client reused
    assert made == [("gpt-4o-mini", DEFAULT_MAX_TOKENS), ("gpt-4o-mini", 16384)]
    assert max_tokens_for("critic") == DEFAULT_MAX_TOKENS


def test_one_argument_factories_still_work(tmp_path):
    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m: FakeChat(m))
    assert llm.structured_call(Answer, "p", agent="screener_a").label == "answer to: p"


def test_many_studies_rule_only_above_the_threshold():
    assert many_studies_rule(MANY_STUDIES) == ""
    assert "231 studies" in many_studies_rule(231)
    protocol = {"question": "q", "pico": {"outcomes": ["accuracy"]}}
    # Small reviews keep the exact prompt they had (their cached answers stay valid).
    assert narrative_prompt(protocol, "ev", [], ["x"], 13) == narrative_prompt(
        protocol, "ev", [], ["x"])
    assert "do not write one sentence per study" in narrative_prompt(protocol, "ev", [], [],
                                                                     231)


# --- (2) a stage that fails the same way twice -----------------------------------------------

def test_error_signature_ignores_token_counts():
    a = error_signature(ValueError(LENGTH.format(4096, 46661)))
    b = error_signature(ValueError(LENGTH.format(4096, 54719)))
    assert a == b and a.startswith("ValueError: LengthFinishReasonError")
    assert error_signature(ValueError("x")) != error_signature(TypeError("x"))


def _entry(node, rnd, failed=None):
    return {"node": node, "round": rnd, "event": node, **({"failed": failed} if failed else {})}


def test_repeated_failure_needs_the_same_error_twice_in_a_row():
    err, other = "ValueError: length", "TimeoutError: slow"
    coord = {"node": "coordinator", "proposed": "synthesis"}
    assert repeated_failure({"audit_log": [_entry("synthesis", 1, err)]}) is None
    assert repeated_failure({"audit_log": [_entry("synthesis", 1, err), coord,
                                           _entry("synthesis", 2, other)]}) is None
    # A success in between: the last run did not fail after a failure.
    assert repeated_failure({"audit_log": [_entry("synthesis", 1, err),
                                           _entry("synthesis", 2)]}) is None
    # A critic run in between does not break the pair (it is another stage).
    stop = repeated_failure({"steps_taken": 9, "audit_log": [
        _entry("synthesis", 1, err), coord, _entry("critic", 1), coord,
        _entry("synthesis", 2, err)]})
    assert stop["event"] == "stopped: repeated failure"
    assert stop["stage"] == "synthesis" and stop["rounds"] == [1, 2] and stop["error"] == err


def test_run_stops_when_synthesis_fails_twice():
    err = "ValueError: LengthFinishReasonError"

    def failing_synthesis(state):
        rnd = state["completed_stages"].count("synthesis") + 1
        return {"completed_stages": ["synthesis"],
                "audit_log": [{"node": "synthesis", "round": rnd, "event": "synthesis",
                               "failed": err}]}

    # The coordinator would keep proposing synthesis forever (as seen live).
    script = [{"next": s, "rationale": "r"} for s in STAGES[:STAGES.index("synthesis")]]
    script += [{"next": "synthesis", "rationale": "again"}] * 10
    graph = build_graph(ScriptedLLM({"coordinator": script}),
                        stages={"synthesis": failing_synthesis})
    final = graph.invoke({"question": "q"}, run_config("t"))
    assert final["completed_stages"].count("synthesis") == 2
    assert "report" not in final["completed_stages"]
    last = final["audit_log"][-1]
    assert last["event"] == "stopped: repeated failure" and last["stage"] == "synthesis"


def test_synthesis_not_rerun_before_the_critic_judged_it():
    done = list(STAGES[:STAGES.index("synthesis")])
    state = {"completed_stages": done + ["synthesis", "critic", "synthesis"],
             "qa_report": {"round": 1, "synthesis_round": 1, "passed": False, "blockers": 1,
                           "warnings": 0, "loop_back": True, "loops_used": 0, "max_loops": 2,
                           "issues": []}, "steps_taken": 9,
             "audit_log": []}
    assert loop_rule("synthesis", state) == "the critic has not judged the current synthesis yet"
    out = coordinator_step(state, ScriptedLLM({"coordinator": [
        {"next": "synthesis", "rationale": "the critic asked for it"}]}))
    assert out["next_stage"] == "critic"
    assert out["audit_log"][0]["outcome"] == "refused"
    # Right after the critic asked for a loop, synthesis is still allowed.
    state["completed_stages"] = done + ["synthesis", "critic"]
    assert loop_rule("synthesis", state) is None


# --- (3) D 'no' without a non-primary word, no design list -----------------------------------

T1_PROTOCOL = {"question": "q", "pico": {}, "study_designs": [],
               "inclusion_criteria": ["Studies reporting detection of cardiac arrhythmias using "
                                      "smartwatches"], "exclusion_criteria": []}
CAILLOL = ("Accuracy of a Smartwatch-Derived ECG for Diagnosing Bradyarrhythmias, "
           "Tachyarrhythmias, and Cardiac Ischemia.")


def v(cid, met, quote=""):
    return {"criterion": cid, "met": met, "quote": quote, "chunk_id": ""}


def answer(*verdicts):
    return {"comparison": "smartwatch vs ECG", "verdicts": list(verdicts), "reason": "r",
            "unresolved": False}


def test_d_no_quoting_a_bare_title_is_unclear_without_designs():
    text = f"Title: {CAILLOL}\nAbstract: (no abstract available)"
    out, flags = check_verdicts([v("D", "no", CAILLOL)], ["D"], text, "a")
    assert out[0]["met"] == "unclear" and out[0]["met_claimed"] == "no"
    assert flags == ["no_nonprimary_word_a_D"]
    # A quote that names a non-primary record still fails D.
    text = "Title: Smartwatches for arrhythmia detection: a systematic review"
    out, flags = check_verdicts([v("D", "no", "a systematic review")], ["D"], text, "a")
    assert out[0]["met"] == "no" and flags == []


def test_design_topics_keep_their_own_rule():
    import re
    text = "Title: x\nAbstract: A prospective cohort study of 300 adults."
    out, flags = check_verdicts([v("D", "no", "A prospective cohort study of 300 adults.")],
                                ["D"], text, "a", re.compile(r"randomi[sz]|\brcts?\b", re.I))
    assert out[0]["met"] == "no" and flags == []  # not the listed design: D fails


def test_caillol_case_is_no_longer_an_agreed_exclusion():
    """Live (T1 183527): A and B both excluded Caillol (no abstract) on D, quoting the title.
    With the check, neither exclusion has evidence: A (strict) excludes on missing
    information, B includes, and the adjudicator decides."""
    paper = {"paper_id": "caillol", "title": CAILLOL, "abstract": ""}
    excl = answer(v("D", "no", CAILLOL), v("I1", "unclear"))
    adj = answer(v("D", "unclear"), v("I1", "yes", "Smartwatch-Derived ECG"))
    llm = ScriptedLLM({"screener_a": [excl], "screener_b": [excl], "adjudicator": [adj]})
    item, _ = screen_record(llm, T1_PROTOCOL, paper, ta_evidence(paper), stage="ta", rnd=1)
    assert {"no_nonprimary_word_a_D", "no_nonprimary_word_b_D"} <= set(item["flags"])
    assert item["a"]["decision"] == "exclude" and item["a"]["missing_info"]
    assert item["b"]["decision"] == "include"
    assert item["final"] == "include" and item["resolved_by"] == "adjudicator"
