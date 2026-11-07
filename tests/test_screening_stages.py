"""Part 6c: criteria tagged by stage, multi-arm comparison, same-reason agreement,
positive test sentences."""

from pathlib import Path

from sr_pipeline.agents.planner import apply_answer, make_planner
from sr_pipeline.agents.screening import (criterion_query, deferred_at_ta, screen_record,
                                          screening_prompt, ta_evidence)
from sr_pipeline.evaluation.topics import load_topic, seed_protocol
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import Template, load_template

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "No significant baseline difference between groups"
MEAN_SD = "Data reported as mean and standard deviation"
COMORB = "Older adults with comorbidities"
PROTOCOL = {
    "question": "q", "pico": {}, "study_designs": ["randomized controlled trial"],
    "inclusion_criteria": ["Aged over 60", BASELINE, MEAN_SD],
    "exclusion_criteria": [COMORB],
    "full_text_criteria": [BASELINE, MEAN_SD],
    "criterion_tests": {COMORB: "Participants have a disease other than depression."},
}
PAPER = {"paper_id": "p1", "title": "Walking for older adults: a randomized controlled trial",
         "abstract": "Adults aged 70 were randomized to walking, horticulture or a wait-list "
                     "control. Groups differed at baseline in age. Results are given as "
                     "medians."}
EV = ta_evidence(PAPER)


def v(cid, met, quote=""):
    return {"criterion": cid, "met": met, "quote": quote, "chunk_id": ""}


def answer(*verdicts, comparison="walking arm vs wait-list control", unresolved=False):
    return {"comparison": comparison, "verdicts": list(verdicts), "reason": "r",
            "unresolved": unresolved}


OK = [v("D", "yes", "a randomized controlled trial"), v("I1", "yes", "aged 70")]
# The abstract really does contradict I2 (baseline) and I3 (mean and SD).
FT_FAIL = answer(*OK, v("I2", "no", "Groups differed at baseline in age"),
                 v("I3", "no", "Results are given as medians"), v("E1", "no"))


# --- Fix 1: criteria tagged by stage ----------------------------------------------------

def test_topic_tags_flow_into_the_protocol():
    t3 = load_topic("T3")
    assert all(isinstance(c, str) for c in t3["inclusion_criteria"] + t3["exclusion_criteria"])
    assert t3["full_text_criteria"] == [
        "No significant baseline difference between experimental and control groups",
        "Data reported as mean and standard deviation"]
    assert set(t3["criterion_tests"]) == set(t3["exclusion_criteria"])
    assert "never a comorbidity" in t3["criterion_tests"][t3["exclusion_criteria"][0]]
    p, audit = seed_protocol({"question": "x", "title": "t", "rationale": "r",
                              "pico": {"population": "", "intervention": "",
                                       "comparator": "", "outcomes": []},
                              "inclusion_criteria": [], "exclusion_criteria": [],
                              "study_designs": [], "databases": ["openalex"]}, t3)
    assert p["full_text_criteria"] == t3["full_text_criteria"]
    assert p["criterion_tests"] == t3["criterion_tests"]
    assert {"full_text_criteria", "criterion_tests"} <= set(audit["seeded_fields"])
    assert deferred_at_ta(p) == {"I3", "I6"}


def test_reworded_criterion_loses_its_tag():
    p, _ = seed_protocol({"question": "x", "title": "t", "rationale": "r",
                          "pico": {"population": "", "intervention": "", "comparator": "",
                                   "outcomes": []},
                          "inclusion_criteria": [], "exclusion_criteria": [],
                          "study_designs": [], "databases": ["openalex"]}, load_topic("T3"))
    edited = [c if "baseline" not in c else "Groups comparable at baseline"
              for c in p["inclusion_criteria"]]
    approved, _, _ = apply_answer(p, {"edits": {"inclusion_criteria": edited}})
    assert approved.full_text_criteria == ["Data reported as mean and standard deviation"]


def test_planner_tags_kept_only_when_copied_exactly():
    draft = {"title": "t", "rationale": "r", "study_designs": [],
             "pico": {"population": "older adults", "intervention": "exercise",
                      "comparator": "control", "outcomes": ["depressive symptoms"]},
             "inclusion_criteria": ["Aged 60 or older", "Reports mean and SD"],
             "exclusion_criteria": [],
             "full_text_criteria": ["Reports mean and SD", "Baseline balance"],  # 2nd invented
             "databases": ["openalex"], "database_reason": "r"}
    update = make_planner(ScriptedLLM({"planner": [draft]}),
                          load_template(ROOT / "protocols" / "template.yaml"))(
        {"question": "Does exercise reduce depressive symptoms?", "completed_stages": []})
    assert update["protocol"]["full_text_criteria"] == ["Reports mean and SD"]
    assert update["audit_log"][0]["full_text_criteria_dropped"] == ["Baseline balance"]
    assert Template().inclusion_criteria == []  # template untouched


def test_full_text_criterion_never_decides_at_title_abstract():
    # Both screeners answer 'no' on I2 / I3 with verified quotes; at title/abstract the
    # verdicts are deferred, A's strict rule ignores them, the record is included.
    llm = ScriptedLLM({"screener_a": [FT_FAIL, FT_FAIL], "screener_b": [FT_FAIL, FT_FAIL]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"]) == ("include", "agreement")
    i2 = next(x for x in item["a"]["verdicts"] if x["criterion"] == "I2")
    assert (i2["met"], i2["deferred"], i2["met_claimed"]) == ("unclear", True, "no")
    assert "[full text only]" in llm.calls[0][1] and "I2 [full text only]:" in llm.calls[0][1]
    # The same answers at full text decide: first failure is I2.
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ft", rnd=1)
    assert (item["final"], item["criterion"]) == ("exclude", "I2")


def test_strict_rule_still_applies_to_criteria_an_abstract_can_show():
    a = answer(OK[0], v("I1", "unclear"), v("I2", "unclear"), v("I3", "unclear"),
               v("E1", "no"))
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [answer(*OK, v("E1", "no"))],
                       "adjudicator": [answer(*OK, v("E1", "no"))]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert (item["a"]["decision"], item["a"]["criterion"]) == ("exclude", "I1")


# --- Fix 2: multi-arm comparison ----------------------------------------------------------

def test_comparison_is_stored_and_shown_to_the_adjudicator():
    a = answer(*OK, v("E1", "yes", "horticulture"), comparison="horticulture vs control")
    b = answer(*OK, v("E1", "no"))
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [b], "adjudicator": [b]})
    item, _ = screen_record(llm, PROTOCOL, PAPER, EV, stage="ta", rnd=1)
    assert item["a"]["comparison"] == "horticulture vs control"
    assert item["adjudicator"]["comparison"] == "walking arm vs wait-list control"
    adj_prompt = next(p for agent, p in llm.calls if agent == "adjudicator")
    assert "Comparison judged: horticulture vs control" in adj_prompt
    assert "Multi-arm trials" in llm.calls[0][1]


# --- Fix 3: test sentences ----------------------------------------------------------------

def test_test_sentence_is_shown_and_used_for_retrieval():
    prompt = screening_prompt(PROTOCOL, EV, style="strict", stage="ta")
    assert f"  E1: {COMORB}\n      Test for E1: Participants have a disease other than " \
           "depression." in prompt
    assert criterion_query(PROTOCOL, "E1", COMORB) == \
        f"E1: {COMORB} (Participants have a disease other than depression.)"
    assert criterion_query(PROTOCOL, "I1", "Aged over 60") == "I1: Aged over 60"
