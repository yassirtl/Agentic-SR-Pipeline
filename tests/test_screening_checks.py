"""Part 6d: D title rule and eligibility-list quotes (last screening checks before freeze)."""

import csv
from pathlib import Path

import pytest

from sr_pipeline.agents.screening import (check_verdicts, eligibility_sentence,
                                          make_ta_screening, screen_record, ta_evidence,
                                          title_rule)
from sr_pipeline.llm import ScriptedLLM

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {"question": "q", "pico": {}, "study_designs": ["randomized controlled trial"],
            "inclusion_criteria": ["Aged over 60"],
            "exclusion_criteria": ["Participants with cognitive impairment"]}


def v(cid, met, quote=""):
    return {"criterion": cid, "met": met, "quote": quote, "chunk_id": ""}


def answer(*verdicts, unresolved=False):
    return {"comparison": "exercise vs control", "verdicts": list(verdicts), "reason": "r",
            "unresolved": unresolved}


# --- Fix 2: D title rule -----------------------------------------------------------------

# Real titles from the dev search (D0 topic, then called T3).
NON_PRIMARY = [
    "Exercise for depression in older adults: a meta-analysis of randomized controlled trials",
    "Effects of tai chi on depression in older adults: A Systematic Review and Meta-Analysis",
    "A systematic review of exercise interventions for late-life depression",
    "Comparative effects of exercise types on depression: a Bayesian network meta-analysis",
    "Fun Exercise for Older Adults (FEXO): study protocol for a randomised controlled trial",
    "Technology and physical activity for preventing decline in older adults: Protocol of a "
    "pilot RCT",
    "Aquatic exercise training for falls in older people: A blinded randomized controlled "
    "trial protocol",
    "Exercise interventions for older people with cognitive frailty-a scoping review.",
    "Non-pharmacological depression therapies for older Chinese adults: A systematic review "
    "& meta-analysis.",
    "Editorial: Exercise and diet: strategies to improve mental and cognitive health",
    "Guest editorial",
    "Commentary on “Yoga in Coronary Artery Disease”",
]
PRIMARY = [
    # exercise "protocols" are the intervention, not a study protocol
    "Effect of two physical exercise protocols on cognition and depressive symptoms in "
    "oldest-old people: A randomized controlled trial.",
    "Feasibility, safety, and adherence of a remote physical and cognitive exercise protocol "
    "for older women",
    # T2-style titles: systematic reviews are the subject, not the record type
    "Methodological insights into ChatGPT's screening performance in systematic reviews",
    "Transforming systematic literature reviews: unleashing the potential of GPT-4",
    "Zero-Shot Generative Large Language Models for Systematic Review Screening Automation",
    "Utilizing ChatGPT to select literature for meta-analysis shows workload reduction",
    # a primary trial reported together with a meta-analysis: left to the screeners
    "Exploring the efficacy of traditional Chinese medicine exercise: a comprehensive study "
    "with randomized controlled trial and network meta-analysis",
]


@pytest.mark.parametrize("title", NON_PRIMARY)
def test_title_rule_catches_stated_non_primary_records(title):
    assert title_rule(title)


@pytest.mark.parametrize("title", PRIMARY)
def test_title_rule_leaves_primary_studies_alone(title):
    assert title_rule(title) is None


def test_title_rule_never_fires_on_a_gold_study():
    """Regression guard: no gold study (T1, T3, D0, and the dropped Kim 2025 list: 77
    primary studies) has a title the rule would exclude. A rule change that breaks this
    loses gold studies before any LLM call."""
    rows = [r for f in sorted((ROOT / "evaluation" / "gold").glob("*.csv"))
            if not f.stem.endswith("_coverage") for r in csv.DictReader(f.open())]
    assert len(rows) == 77
    assert [r["title"] for r in rows if title_rule(r["title"])] == []


def test_title_rule_decides_d_without_any_llm_call():
    paper = {"paper_id": "p1", "abstract": "We will randomize 120 adults aged 65+.",
             "title": "Walking for older adults: study protocol for a randomized trial"}
    llm = ScriptedLLM({})  # any call would raise
    item, usages = screen_record(llm, PROTOCOL, paper, ta_evidence(paper), stage="ta", rnd=1)
    assert llm.calls == [] and usages == []
    assert (item["final"], item["criterion"], item["resolved_by"]) == \
        ("exclude", "D", "d_title_rule")
    assert item["quote"] == paper["title"] and item["title_words"] == "study protocol"
    assert item["a"] is None and item["b"] is None and "d_title_rule" in item["flags"]


def test_title_rule_records_are_counted_but_not_in_kappa():
    papers = [{"paper_id": "rev", "title": "Exercise for depression: a meta-analysis",
               "abstract": "We pooled trials."},
              {"paper_id": "rct", "title": "Walking for depression",
               "abstract": "A randomized controlled trial of adults aged 70."}]
    ok = answer(v("D", "yes", "A randomized controlled trial"), v("I1", "yes", "aged 70"),
                v("E1", "unclear"))
    llm = ScriptedLLM({"screener_a": [ok], "screener_b": [ok]})
    out = make_ta_screening(llm, workers=1)({"protocol": PROTOCOL, "unique_records": papers,
                                             "completed_stages": ["retrieval"]})
    c = out["prisma_counts"]["title_abstract"]
    assert (c["records_screened"], c["included"], c["records_excluded"]) == (2, 1, 1)
    assert c["excluded_by_criterion"] == {"D": 1} and c["title_rule"] == 1
    assert c["resolved_by"] == {"d_title_rule": 1, "agreement": 1}
    assert c["pairs"] == 1  # kappa only over records both screeners read


# --- Fix 1: eligibility-list quotes -------------------------------------------------------

FT_TEXT = (
    "Older adults aged 65 or over were recruited. The exclusion criteria were as follows: "
    "(1) support or care certified by the insurance system; (2) dementia diagnosis or "
    "MMSE score of 18 or less. Participants were excluded if they had a dementia diagnosis "
    "or severe cognitive impairment. Inclusion criteria: 1. Aged 65 or over 2. Living at "
    "home. Twelve participants had mild cognitive impairment at baseline.")


@pytest.mark.parametrize("quote", [
    "The exclusion criteria were as follows: (1) support or care certified",  # header quoted
    "dementia diagnosis or severe cognitive impairment",   # fragment: header in its sentence
    "dementia diagnosis or MMSE score of 18 or less",       # list item after "as follows:"
    "Living at home",                                       # "1." does not end the sentence
])
def test_eligibility_sentence_found(quote):
    assert eligibility_sentence(quote, FT_TEXT)


def test_ordinary_sentence_is_not_an_eligibility_sentence():
    assert eligibility_sentence("Twelve participants had mild cognitive impairment",
                                FT_TEXT) is None


def test_exclusion_yes_from_an_eligibility_list_is_downgraded():
    raw = [v("D", "yes", "recruited"), v("I1", "yes", "aged 65 or over"),
           v("E1", "yes", "dementia diagnosis or severe cognitive impairment")]
    out, flags = check_verdicts(raw, ["D", "I1", "E1"], FT_TEXT, "a")
    e1 = out[2]
    assert (e1["met"], e1["met_claimed"]) == ("unclear", "yes")
    assert e1["eligibility_sentence"].startswith("Participants were excluded if")
    assert flags == ["eligibility_quote_a_E1"]


def test_exclusion_yes_from_the_study_population_still_excludes():
    raw = [v("E1", "yes", "Twelve participants had mild cognitive impairment")]
    out, flags = check_verdicts(raw, ["E1"], FT_TEXT, "b")
    assert out[0]["met"] == "yes" and flags == []


def test_only_e_yes_is_checked():
    # An I 'no' quoting an eligibility sentence is a real contradiction (age 18-40 trial).
    text = "Inclusion criteria were age 18 to 40 years."
    out, flags = check_verdicts([v("I1", "no", "age 18 to 40 years")], ["I1"], text, "a")
    assert out[0]["met"] == "no" and flags == []


def test_makizako_case_is_no_longer_excluded():
    """6c: both screeners (and then the adjudicator) excluded a gold study on E1 by quoting
    its exclusion criteria. With the check, the exclusions do not stand."""
    paper = {"paper_id": "mak", "title": "Exercise and horticultural programs for older "
             "adults with depressive symptoms", "abstract": FT_TEXT}
    excl = answer(v("D", "yes", "Older adults aged 65 or over were recruited"),
                  v("I1", "yes", "aged 65 or over"),
                  v("E1", "yes", "The exclusion criteria were as follows"))
    llm = ScriptedLLM({"screener_a": [excl], "screener_b": [excl]})
    item, _ = screen_record(llm, PROTOCOL, paper, ta_evidence(paper), stage="ta", rnd=1)
    assert item["final"] == "include" and item["resolved_by"] == "agreement"
    assert {"eligibility_quote_a_E1", "eligibility_quote_b_E1"} <= set(item["flags"])


def test_adjudicator_quote_from_eligibility_list_is_downgraded_too():
    paper = {"paper_id": "p", "title": "Tai chi for older adults", "abstract": FT_TEXT}
    a = answer(v("D", "yes", "were recruited"), v("I1", "yes", "aged 65 or over"),
               v("E1", "yes", "Twelve participants had mild cognitive impairment"))
    b = answer(v("D", "yes", "were recruited"), v("I1", "yes", "aged 65 or over"),
               v("E1", "no"))
    adj = answer(v("D", "yes", "were recruited"), v("I1", "yes", "aged 65 or over"),
                 v("E1", "yes", "severe cognitive impairment"))
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [b], "adjudicator": [adj]})
    item, _ = screen_record(llm, PROTOCOL, paper, ta_evidence(paper), stage="ta", rnd=1)
    assert "eligibility_quote_adjudicator_E1" in item["flags"]
    assert item["final"] == "include" and item["resolved_by"] == "adjudicator"
