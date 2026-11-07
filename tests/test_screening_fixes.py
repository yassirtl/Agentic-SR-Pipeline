"""Part 6b: topic-seeded protocol, per-criterion verdicts with a code-chosen deciding
criterion, primary-study (D) guard on the adjudicator."""

from datetime import date
from pathlib import Path

import pytest
from langgraph.types import Command

from sr_pipeline.agents.planner import apply_answer, build_prompt, make_planner
from sr_pipeline.agents.screening import (criteria, decide, fails, make_ta_screening,
                                          screen_record, ta_evidence)
from sr_pipeline.evaluation.topics import load_topic, seed_protocol, topic_template
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import load_template

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / "protocols" / "template.yaml")

DRAFT = {
    "title": "Exercise and depression in older adults",
    "pico": {"population": "older adults", "intervention": "exercise",
             "comparator": "no exercise", "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": ["Participants aged 65 years or older"],   # the Part 6 mistake
    "exclusion_criteria": ["Severe cognitive impairment"],
    "study_designs": [],
    "rationale": "r",
    "databases": ["openalex", "pubmed"],
    "database_reason": "health",
}


# --- Fix 1: protocol seeded from topics.yaml ------------------------------------------------

def test_topic_seeds_question_criteria_dates_and_designs():
    t3 = load_topic("T3")
    llm = ScriptedLLM({"planner": [DRAFT]})
    update = make_planner(llm, TEMPLATE, topic=t3)({"question": "ignored",
                                                    "completed_stages": []})
    p = update["protocol"]
    assert p["question"] == t3["question"]
    assert p["inclusion_criteria"] == t3["inclusion_criteria"]       # exact, not the draft
    assert "Healthy older adults aged over 60 years" in p["inclusion_criteria"]
    assert p["exclusion_criteria"] == t3["exclusion_criteria"]
    assert p["study_designs"] == ["randomized controlled trial"]
    assert p["years"] == {"start": None, "end": 2024, "end_date": "2024-01-07",
                          "recent": None}  # the topic replaces the template's last-5-years
    assert p["languages"] == ["English", "Chinese"]
    assert p["topic"] == "T3" and p["criteria_source"].startswith("topics.yaml T3 (review Li")
    entry = update["audit_log"][0]
    assert entry["criteria_source"] == p["criteria_source"]
    assert "inclusion_criteria" in entry["seeded"]["seeded_fields"]
    # The Planner saw the topic's scope and criteria as hints.
    prompt = llm.calls[0][1]
    assert "years any to 2024" in prompt and "aged over 60" in prompt


def test_topic_without_criteria_keeps_the_planners_and_says_so():
    t1 = load_topic("T1")
    p, audit = seed_protocol({"question": "q", **DRAFT}, t1)
    assert p["inclusion_criteria"] == DRAFT["inclusion_criteria"]
    assert p["criteria_source"] == "Planner (LLM draft); topics.yaml T1 has no criteria"
    assert p["years"]["end_date"] == "2021-02-28" and p["study_designs"] == []
    assert "inclusion_criteria" not in audit["seeded_fields"]
    with pytest.raises(KeyError):
        load_topic("T9")


def test_topic_template_scope():
    tpl = topic_template(TEMPLATE, load_topic("T3"))
    assert tpl.years.start is None and tpl.years.end_date == date(2024, 1, 7)
    assert tpl.study_designs == ["randomized controlled trial"]
    assert "Inclusion criteria required by the user" in build_prompt("q", tpl)
    assert TEMPLATE.years.recent == 5  # the base template is not modified


def test_approval_shows_seeded_protocol_and_edits_are_traced():
    t3 = load_topic("T3")
    llm = ScriptedLLM({"planner": [DRAFT],
                       "coordinator": [{"next": s, "rationale": "r"} for s in STAGES]})
    graph = build_graph(llm, stages={"planner": make_planner(llm, TEMPLATE, topic=t3)},
                        approve_protocol=True)
    cfg = run_config("topic-approval")
    first = graph.invoke({"question": t3["question"]}, cfg)
    (intr,) = first["__interrupt__"]
    assert intr.value["kind"] == "protocol_approval"
    assert intr.value["criteria_source"].startswith("topics.yaml T3")
    assert intr.value["protocol"]["inclusion_criteria"] == t3["inclusion_criteria"]
    # The human edits a criterion: the source says so.
    edited = t3["inclusion_criteria"][:-1]
    final = graph.invoke(Command(resume={"edits": {"inclusion_criteria": edited}}), cfg)
    assert final["protocol"]["inclusion_criteria"] == edited
    assert final["protocol"]["criteria_source"].endswith("edited by human at approval")
    approval = next(e for e in final["audit_log"] if e["node"] == "protocol_approval")
    assert approval["criteria_source"] == final["protocol"]["criteria_source"]
    # Approving as is keeps the source.
    approved, _, _ = apply_answer(intr.value["protocol"], True)
    assert approved.criteria_source == intr.value["criteria_source"]


# --- Fix 2: per-criterion verdicts, deciding criterion chosen by code -------------------------

PROTOCOL = {"question": "q", "pico": {}, "study_designs": ["randomized controlled trial"],
            "inclusion_criteria": ["Aged over 60", "Exercise intervention"],
            "exclusion_criteria": ["Comorbidities"]}
# Neutral title: the record type is only in the abstract, so the LLM path is tested
# (a title stating "meta-analysis" is decided by the Part 6d title rule).
PAPER = {"paper_id": "p1", "title": "Exercise in older adults",
         "abstract": "This network meta-analysis: we pooled 40 randomized trials of adults "
                     "aged 50 to 90 with diabetes; "
                     "this study protocol describes the planned analysis."}
EV = ta_evidence(PAPER)


def v(cid, met, quote=""):
    return {"criterion": cid, "met": met, "quote": quote, "chunk_id": ""}


def answer(*verdicts, unresolved=False):
    return {"verdicts": list(verdicts), "reason": "r", "unresolved": unresolved}


ALL_OK = [v("D", "yes", "randomized trials"), v("I1", "yes", "older adults"),
          v("I2", "yes", "exercise"), v("E1", "no")]


def test_d_is_first_and_always_present():
    assert [c for c, _ in criteria(PROTOCOL)] == ["D", "I1", "I2", "E1"]
    d_restricted = criteria(PROTOCOL)[0][1]
    assert "randomized controlled trial" in d_restricted and "network meta-analyses" in \
        d_restricted
    d_any = criteria({**PROTOCOL, "study_designs": []})[0]
    assert d_any[0] == "D" and "any design" in d_any[1] and "protocols" in d_any[1]


def test_fail_semantics_and_decision_rules():
    assert fails("I1", "no") and fails("D", "no") and fails("E1", "yes")
    assert not fails("E1", "no") and not fails("I1", "unclear")
    checked = [{**x, "verified": True} for x in
               [v("D", "yes"), v("I1", "unclear"), v("I2", "yes"), v("E1", "no")]]
    assert decide(checked, "strict")["decision"] == "exclude"        # missing information
    assert decide(checked, "strict")["missing_info"] is True
    assert decide(checked, "lenient")["decision"] == "include"


def test_deciding_criterion_is_first_failure_in_fixed_order():
    # The screener names I1 and E1 failures first in its list, then D: code reports D.
    a = answer(v("I1", "no", "adults aged 50 to 90"), v("E1", "yes", "with diabetes"),
               v("D", "no", "We pooled 40 randomized trials"), v("I2", "yes", "exercise"))
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [a]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"], item["criterion"]) == \
        ("exclude", "agreement", "D")
    assert [x["criterion"] for x in item["a"]["verdicts"]] == ["D", "I1", "I2", "E1"]
    prompt = llm.calls[0][1]
    assert prompt.index("  D: ") < prompt.index("  I1: ") < prompt.index("  E1: ")


def test_unverified_failure_never_decides():
    # D 'no' with an invented quote is downgraded; the verified I1 failure decides.
    a = answer(v("D", "no", "a cross-sectional survey"), v("I1", "no", "aged 50 to 90"),
               v("I2", "yes", "exercise"), v("E1", "no"))
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [a]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert item["criterion"] == "I1"
    assert "unverified_a_D" in item["flags"] and "quote_not_found_a" in item["flags"]
    d = item["a"]["verdicts"][0]
    assert (d["met"], d["met_claimed"]) == ("unclear", "no")


def test_missing_verdict_is_unclear_and_strict_rule_off_without_full_text():
    # A skips I1 (unclear). At title/abstract A excludes on it; without a full text
    # (ft_abstract) A follows the lenient rule, as its prompt says.
    a = answer(v("D", "yes", "randomized trials"), v("I2", "yes", "exercise"), v("E1", "no"))
    llm = ScriptedLLM({"screener_a": [a, a], "screener_b": [answer(*ALL_OK)] * 2,
                       "adjudicator": [answer(*ALL_OK)]})
    ta, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert "verdict_missing_a_I1" in ta["flags"]
    assert (ta["a"]["decision"], ta["a"]["criterion"]) == ("exclude", "I1")
    ft, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ft", rnd=1, mode="ft_abstract")
    assert (ft["a"]["decision"], ft["resolved_by"]) == ("include", "agreement")


def test_counts_by_deciding_criterion():
    papers = [{**PAPER, "paper_id": f"p{i}"} for i in range(3)]
    d_fail = answer(v("D", "no", "this study protocol"), *ALL_OK[1:])
    i_fail = answer(ALL_OK[0], v("I1", "no", "aged 50 to 90"), *ALL_OK[2:])
    script = {"p0": d_fail, "p1": i_fail, "p2": answer(*ALL_OK)}
    pick = lambda prompt: next(a for pid, a in script.items()
                               if f"Title: {pid}" in prompt)
    llm = ScriptedLLM({"screener_a": pick, "screener_b": pick})
    papers = [{**p, "title": p["paper_id"]} for p in papers]
    state = {"protocol": PROTOCOL, "unique_records": papers,
             "completed_stages": ["planner", "search_strategy", "retrieval"]}
    c = make_ta_screening(llm, workers=1)(state)["prisma_counts"]["title_abstract"]
    assert c["excluded_by_criterion"] == {"D": 1, "I1": 1} and c["included"] == 1


# --- Fix 3: primary-study guard on the adjudicator -------------------------------------------

def split(adj):
    """A excludes on D with a verified quote, B includes, the adjudicator answers ``adj``."""
    return ScriptedLLM({"screener_a": [answer(v("D", "no", "network meta-analysis"),
                                              *ALL_OK[1:])],
                        "screener_b": [answer(*ALL_OK)], "adjudicator": [adj]})


def test_adjudicator_cannot_include_a_verified_non_primary_study():
    # Escape: the adjudicator answers D 'yes' with a quote found in the text.
    item, _ = screen_record(split(answer(*ALL_OK)), PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"]) == ("include", "adjudicator")
    # No evidence for D: the guard excludes on D with the screener's quote.
    item, _ = screen_record(split(answer(v("D", "unclear"), *ALL_OK[1:])),
                            PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"], item["criterion"]) == \
        ("exclude", "design_guard", "D")
    assert "design_guard" in item["flags"] and item["guard_quote"] == "network meta-analysis"


def test_design_guard_beats_unresolved_and_invented_evidence():
    # Unresolved would be auto-included in --auto: the guard excludes instead.
    item, _ = screen_record(split(answer(v("D", "unclear"), *ALL_OK[1:], unresolved=True)),
                            PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"]) == ("exclude", "design_guard")
    # A D 'yes' whose quote is not in the text does not count as evidence.
    item, _ = screen_record(split(answer(v("D", "yes", "a two-arm parallel trial"),
                                         *ALL_OK[1:])), PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert item["resolved_by"] == "design_guard"


def test_guard_needs_a_verified_d_failure():
    # A's D 'no' quote is invented -> downgraded; the adjudicator's include stands.
    llm = ScriptedLLM({"screener_a": [answer(v("D", "no", "a case report"), *ALL_OK[1:])],
                       "screener_b": [answer(*ALL_OK)], "adjudicator": [answer(*ALL_OK)]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    # A: D downgraded to unclear -> strict rule still excludes on D (missing information),
    # so A and B disagree; no verified D failure -> no guard.
    assert (item["final"], item["resolved_by"]) == ("include", "adjudicator")
    assert "design_guard" not in item["flags"]


def test_d_no_quoting_the_listed_design_is_a_contradiction():
    # Seen live: "D: no" quoting "A pilot cluster-randomized controlled trial".
    paper = {"paper_id": "p", "title": "A pilot cluster-randomized controlled trial of HIIT",
             "abstract": "Older adults aged 70 did interval training; this protocol paper "
                         "is not it."}
    ev = ta_evidence(paper)
    contradiction = answer(v("D", "no", "A pilot cluster-randomized controlled trial"),
                           *ALL_OK[1:])
    llm = ScriptedLLM({"screener_a": [answer(*ALL_OK)], "screener_b": [contradiction],
                       "adjudicator": [answer(*ALL_OK)]})
    item, _ = screen_record(llm, PROTOCOL, paper, ev, stage="ta", rnd=1)
    assert "contradicting_quote_b_D" in item["flags"]
    assert item["b"]["verdicts"][0]["met"] == "unclear" and item["b"]["decision"] == "include"
    # A 'no' quoting a non-primary marker stands, design word or not.
    for quote in ("a meta-analysis of randomized controlled trials",
                  "this protocol paper"):
        p2 = {**paper, "abstract": "We did a meta-analysis of randomized controlled trials; "
                                   "this protocol paper."}
        llm = ScriptedLLM({"screener_a": [answer(v("D", "no", quote), *ALL_OK[1:])],
                           "screener_b": [answer(v("D", "no", quote), *ALL_OK[1:])]})
        item, _ = screen_record(llm, PROTOCOL, p2, ta_evidence(p2), stage="ta", rnd=1)
        assert (item["final"], item["criterion"]) == ("exclude", "D")
