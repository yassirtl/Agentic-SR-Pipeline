"""Part 4d: design suggestion (the human decides) and narrowing that keeps recall.

1. The Planner suggests designs; the suggestion is not applied by default, is
   applied when the human accepts it, and never in --auto.
2. A refinement returns removed / changed terms; code records the real term
   difference and stops early when the counts do not move (no_progress).
"""

import pytest
from langgraph.types import Command

from sr_pipeline.agents.planner import apply_answer, build_prompt as planner_prompt
from sr_pipeline.agents.planner import make_planner
from sr_pipeline.agents.search_strategy import (RefinedPlan, SearchPlan, build_prompt,
                                                build_refine_prompt, choose_attempt,
                                                no_progress, term_diff)
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import Protocol, Template

from test_search_refine import INT, OUT, POP, PROTOCOL, FakeCounts, plan, run_stage

QUESTION = "Does exercise reduce depressive symptoms in older adults?"
RCT = ["randomized controlled trial"]


def draft(suggested=RCT, designs=()):
    return {"title": "t", "rationale": "r", "inclusion_criteria": [], "exclusion_criteria": [],
            "pico": {"population": "older adults", "intervention": "exercise",
                     "comparator": "any control", "outcomes": ["depressive symptoms"]},
            "study_designs": list(designs), "suggested_designs": list(suggested),
            "design_reason": "effect question: randomized trials give the least biased answer",
            "databases": ["pubmed"], "database_reason": "health"}


def scripted(d):
    return ScriptedLLM({"planner": [d],
                        "coordinator": [{"next": s, "rationale": "r"} for s in STAGES]})


def run_graph(d, approve, *answers):
    llm = scripted(d)
    graph = build_graph(llm, stages={"planner": make_planner(llm, Template())},
                        approve_protocol=approve)
    cfg = run_config(f"4d-{approve}-{answers!r}")
    out = graph.invoke({"question": QUESTION}, cfg)
    for a in answers:
        out = graph.invoke(Command(resume=a), cfg)
    return out


def entry(out, node):
    return next(e for e in out["audit_log"] if e["node"] == node)


# --- Fix 1: Planner suggests, the human decides ---------------------------------------

def test_planner_prompt_asks_for_a_suggestion_separate_from_study_designs():
    prompt = planner_prompt(QUESTION, Template())
    assert "Separately, suggest designs (suggested_designs + design_reason)" in prompt
    assert "it does not change study_designs" in prompt


def test_suggestion_is_not_applied_by_default():
    out = make_planner(ScriptedLLM({"planner": [draft(designs=RCT)]}), Template())(
        {"question": QUESTION, "completed_stages": []})
    p = out["protocol"]
    assert p["study_designs"] == []                   # 4c guard: question names no design
    assert p["suggested_designs"] == RCT and "effect question" in p["design_reason"]
    e = out["audit_log"][0]
    assert e["design_suggestion"] == "suggested, not applied"
    assert e["suggested_designs"] == RCT


def test_suggestion_applied_when_the_human_accepts_it():
    paused = run_graph(draft(), True)
    shown = paused["__interrupt__"][0].value
    assert shown["suggested_designs"] == RCT and "accept_suggested_designs" in shown["help"]
    final = run_graph(draft(), True, {"accept_suggested_designs": True})
    assert final["protocol"]["study_designs"] == RCT
    e = entry(final, "protocol_approval")
    assert e["design_suggestion"] == "accepted by human" and e["study_designs"] == RCT
    assert final["completed_stages"] == list(STAGES)


def test_approving_without_accepting_keeps_any_design():
    final = run_graph(draft(), True, True)
    assert final["protocol"]["study_designs"] == []
    assert entry(final, "protocol_approval")["design_suggestion"] == "suggested, not applied"


def test_suggestion_never_applied_in_auto():
    final = run_graph(draft(), False)
    assert "__interrupt__" not in final
    assert final["protocol"]["study_designs"] == []
    assert entry(final, "planner")["design_suggestion"] == "suggested, not applied"


def test_accepting_an_empty_suggestion_asks_again():
    again = run_graph(draft(suggested=[]), True, {"accept_suggested_designs": True})
    assert "suggested no design" in again["__interrupt__"][0].value["error"]
    assert entry(run_graph(draft(suggested=[]), True, True),
                 "planner")["design_suggestion"] == "none suggested"


def test_explicit_design_edit_wins_over_the_suggestion():
    protocol = Protocol.model_validate({**draft(), "question": QUESTION,
                                        "databases": ["pubmed"]}).model_dump(mode="json")
    approved, _, accepted = apply_answer(
        protocol, {"accept_suggested_designs": True, "edits": {"study_designs": ["cohort"]}})
    assert approved.study_designs == ["cohort"] and accepted
    with pytest.raises(ValueError, match="unrecognised"):
        apply_answer(protocol, {"accept_suggested_designs": "yes"})


def test_search_prompt_never_sees_the_suggestion():
    protocol = {**PROTOCOL, "suggested_designs": RCT, "design_reason": "effect question"}
    prompt = build_prompt(protocol)
    assert "Study designs: any\n" in prompt
    assert "randomized controlled trial" not in prompt and "effect question" not in prompt


# --- Fix 2: narrowing that keeps recall ------------------------------------------------

NOISY = {**POP, "keywords": ["older adults", "elderly", "aged"]}
NOISY_INT = {**INT, "keywords": ["exercise", "physical activity", "training", "fitness"]}
CLEAN = {**POP, "keywords": ["older adults", "elderly", "aged 65"]}
CLEAN_INT = {**INT, "keywords": ["exercise", "physical activity"]}


def refined(*blocks, removed=(), changed=()):
    return {**plan(*blocks, why="narrowed"),
            "removed_terms": [{"term": t, "reason": "generic"} for t in removed],
            "changed_terms": [{"old": o, "new": n, "reason": "ambiguous"} for o, n in changed]}


def noisy_hits(q):
    """Fake database: the generic words pull in many unrelated records."""
    q = q.lower()
    return 800 + 20_000 * ("training" in q) + 8_000 * ("aged 65" not in q)


def test_refine_prompt_allows_only_narrowing_moves():
    prompt = build_refine_prompt(PROTOCOL, [{"attempt": 1, "concepts": [NOISY, NOISY_INT, OUT],
                                             "status": {"pubmed": "too broad"},
                                             "counts": {"pubmed": 28_800}}],
                                 max_records=3000, max_hits=3000, min_hits=20)
    assert "ONLY allowed moves" in prompt
    assert "'aged' -> 'aged 65'" in prompt
    assert "Not allowed: adding keywords; removing a specific synonym" in prompt
    assert "MeSH headings are not keywords: keep them" in prompt
    assert "removed_terms" in prompt and "changed_terms" in prompt


def test_refinement_records_removed_and_changed_terms():
    out, _, _ = run_stage(
        [plan(NOISY, NOISY_INT, OUT),
         refined(CLEAN, CLEAN_INT, OUT, removed=["training", "fitness"],
                 changed=[("aged", "aged 65")])],
        fake=FakeCounts(noisy_hits), max_records=3000)
    s = out["search_strategy"]
    first, second = s["attempts"]
    assert first["counts"]["pubmed"] == 28_800 and second["counts"]["pubmed"] == 800
    assert "removed_terms" not in first                       # first call: plain SearchPlan
    assert [t["term"] for t in second["removed_terms"]] == ["training", "fitness"]
    assert second["changed_terms"] == [{"old": "aged", "new": "aged 65", "reason": "ambiguous"}]
    # what code saw really changed (not the LLM's own account)
    assert second["term_diff"] == {"removed": ["aged", "fitness", "training"],
                                   "added": ["aged 65"]}
    assert s["chosen_attempt"] == 2 and '"aged 65"' in s["queries"]["openalex"]
    e = out["audit_log"][0]
    assert e["removed_terms"] == ["training", "fitness"]
    assert e["changed_terms"] == ["aged -> aged 65"] and not e["stopped_no_progress"]


def test_refinement_calls_use_the_refined_schema():
    seen = []

    class Spy(ScriptedLLM):
        def call_with_usage(self, schema, prompt, *, agent):
            seen.append(schema)
            return super().call_with_usage(schema, prompt, agent=agent)

    from test_search_refine import http_for
    from sr_pipeline.agents.search_strategy import make_search_strategy
    llm = Spy({"search_strategy": [plan(POP, INT), refined(POP, INT, OUT)]})
    make_search_strategy(llm, http_for(FakeCounts()), max_records=1000, env={})(
        {"protocol": PROTOCOL, "completed_stages": ["planner"]})
    assert seen == [SearchPlan, RefinedPlan]


def test_no_progress_stops_the_loop_early():
    # the refinement adds a synonym: counts move by 0.1%, as in the 4c run
    def hits(q):
        return 29_587 + 23 * ("depressive episodes" in q.lower())
    more = {**OUT, "keywords": OUT["keywords"] + ["depressive episodes"]}
    out, llm, _ = run_stage([plan(POP, INT, OUT), refined(POP, INT, more), plan(POP, INT, OUT)],
                            fake=FakeCounts(hits), max_records=3000)
    s = out["search_strategy"]
    assert len(llm.calls) == 2 and len(s["attempts"]) == 2   # third call never made
    second = s["attempts"][1]
    assert second["no_progress"] and second["term_diff"]["added"] == ["depressive episodes"]
    assert "no progress (counts changed by less than 5%)" in second["reason"]
    assert "download capped at 3000" in second["reason"]
    assert s["chosen_attempt"] == 1          # the no-progress attempt does not win the tie
    assert s["not_retrieved"]["pubmed"] > 0                  # safety net unchanged (Fix 3)
    assert out["audit_log"][0]["stopped_no_progress"]


def test_no_progress_attempt_never_wins_a_tie():
    a = lambda n, flag=False: {"attempt": n, "status": {"x": "too broad"}, "no_progress": flag}
    assert choose_attempt([a(1), a(2), a(3, True)])["attempt"] == 2
    assert choose_attempt([a(1), a(2)])["attempt"] == 2


def test_no_progress_rule():
    a = lambda **c: {"counts": c}
    assert no_progress(a(x=1000, y=500), a(x=1040, y=510))          # < 5% everywhere
    assert not no_progress(a(x=1000, y=500), a(x=1040, y=400))      # y moved 20%
    assert not no_progress(a(x=0), a(x=10))                          # from zero
    assert not no_progress(a(x=1000), a())                           # nothing to compare


def test_term_diff_ignores_the_design_block():
    design = {"role": "design", "keywords": ["randomized"]}
    assert term_diff([NOISY, design], [CLEAN]) == {"removed": ["aged"], "added": ["aged 65"]}


# --- found in the live run: "capped" only when the ceiling stopped the download -------

def test_records_a_database_does_not_return_are_not_called_capped():
    import httpx
    from sr_pipeline.sources import HttpClient
    from sr_pipeline.sources.openalex import OpenAlexSource as OpenAlex

    def answer(request):  # 3 hits counted, one work has no title (skipped by the parser)
        works = [{"id": f"https://openalex.org/W{i}", "display_name": t}
                 for i, t in enumerate(["A", "B", None])]
        return httpx.Response(200, json={"meta": {"count": 3}, "results": works})

    http = HttpClient(cache_dir=None, transport=httpx.MockTransport(answer), sleep=lambda s: 0)
    r = OpenAlex(http).search("q", max_records=3000)
    assert len(r.records) == 2 and r.not_retrieved == 1     # still reported
    assert not r.capped                                     # but not blamed on the cap
    assert OpenAlex(http).search("q", max_records=1).capped
