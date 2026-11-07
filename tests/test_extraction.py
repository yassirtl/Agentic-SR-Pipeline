"""Part 7: extraction + risk of bias with a scripted LLM and the fake embedder (no network).

The scripted LLM answers from the prompt (each paper's title holds a marker), so answers
do not depend on call order and the same script works with several threads.
"""

import pytest
from pydantic import ValidationError

from sr_pipeline.agents.extraction import (
    NOT_REPORTED, TOOLS, Extraction, RiskOfBias, check_domains, check_fields, check_results,
    choose_tool, extraction_fields, group_queries, make_extraction, number_in, numbers, overall,
    spelled_numbers)
from sr_pipeline.coordinator import summarize_state
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.rag import FakeEmbedder, HybridIndex, chunk_paper
from sr_pipeline.sources.fulltext import FullText, Section
from sr_pipeline.state import latest

PROTOCOL = {
    "question": "Does exercise reduce depressive symptoms in older adults?",
    "pico": {"population": "older adults", "intervention": "exercise",
             "comparator": "usual care", "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": ["Participants aged 60 years or older",
                           "Depression measured with GDS, GDS-15 or TGDS"],
    "exclusion_criteria": ["Interventions beyond exercise"],
    "study_designs": ["randomized controlled trial"],
}

METHODS = ("In this randomized controlled trial conducted in Japan, we randomized 120 "
           "participants with a mean age of 72.3 years to walking or usual care. ")
RESULTS = ("After 12 weeks the GDS-15 score was 4.1 (2.0) in the exercise group and 5.6 "
           "(2.4) in the control group, mean difference -1.5 (95% CI -2.3 to -0.7). ")
ALLOC = "Allocation was concealed using sealed opaque envelopes prepared by a statistician. "

P1 = {"paper_id": "p1", "title": "FULL walking trial", "year": 2020,
      "abstract": "Older adults were randomized to walking or usual care.", "round": 1}
P2 = {"paper_id": "p2", "title": "ABSTRACT walking trial", "year": 2019,
      "abstract": "A randomised trial of chair exercise in nursing homes in Taiwan.",
      "round": 1}
P3 = {"paper_id": "p3", "title": "MISSING walking trial", "year": 2018,
      "abstract": "Walking for mood in older adults.", "round": 1}


def build_index(folder):
    ft = FullText(paper_id="p1", status="pmc_xml", sections=[
        Section(title="Methods", text=METHODS * 2 + ALLOC * 2),
        Section(title="Results", text=RESULTS * 3)])
    chunks = chunk_paper(P1, ft, size=60, overlap=10) + chunk_paper(P2, None)
    HybridIndex.build(chunks, FakeEmbedder()).save(folder)


def field(key, quote, value):
    return {"key": key, "quote": quote, "chunk_id": "", "value": value}


def extractor(prompt: str):
    if "MISSING walking trial" in prompt:
        raise ValueError("model returned broken JSON")
    if "ABSTRACT walking trial" in prompt:
        return {"fields": [field("design", "a randomized trial", "RCT"),  # abstract says
                           field("country", "in Taiwan", "Taiwan")],          # 'randomised'
                "design_category": "randomized trial", "results": []}
    return {
        "fields": [
            field("design", "In this randomized controlled trial", "randomized controlled trial"),
            field("country", "conducted in Japan", "Japan"),
            field("setting", "", "community"),                               # no quote
            field("n_enrolled", "we randomized 120 participants", "120 (60 per arm)"),
            field("age", "a mean age of 72.3 years", "mean 72.3 years"),
            field("sex", "all participants were women", "100% women"),       # invented
            field("comparator", "walking or usual care", "usual care"),
            field("follow_up", "", "not reported"),
            field("O1_measure", "the GDS-15 score", "GDS-15"),
        ],
        "design_category": "randomized trial",
        "results": [{"outcome": "O1", "quote": RESULTS.strip(), "chunk_id": "",
                     "arms": [{"arm": "exercise", "n": 60, "mean": 4.1, "sd": 2.0},
                              {"arm": "control", "mean": 5.6, "sd": 2.4}],
                     "estimates": [{"measure": "mean difference", "value": -1.5,
                                    "ci_low": -2.3, "ci_high": -0.7, "p_value": "0.01"}]}],
    }


def rob_answer(prompt: str):
    if "MISSING walking trial" in prompt:
        return {"domains": []}
    if "ABSTRACT walking trial" in prompt:
        return {"domains": [{"domain": "D1", "quote": "A randomised trial", "judgment": "low",
                             "reason": "random"}]}
    return {"domains": [
        {"domain": "D1", "quote": "Allocation was concealed using sealed opaque envelopes",
         "judgment": "low", "reason": "concealed"},
        {"domain": "D2", "quote": "", "judgment": "high", "reason": "guess"},      # no quote
        {"domain": "D4", "quote": "GDS-15 score", "judgment": "moderate",          # ROBINS-I
         "reason": "wrong tool level"},
        {"domain": "D5", "quote": "mean difference -1.5", "judgment": "low",
         "reason": "reported"},
    ]}


def script(**extra):
    return ScriptedLLM({"extraction": extractor, "risk_of_bias": rob_answer,
                        "reranker": lambda p: {"scores": []}, **extra})


def ft_decision(pid, status, comparison=""):
    return {"paper_id": pid, "round": 1, "final": "include", "fulltext_status": status,
            "a": {"comparison": comparison}, "adjudicator": None}


def state(**kw):
    return {"question": PROTOCOL["question"], "protocol": PROTOCOL,
            "unique_records": [P1, P2, P3], "included": ["p1", "p2", "p3"],
            "ft_decisions": [ft_decision("p1", "pmc_xml", "walking arm vs usual care"),
                             ft_decision("p2", "abstract_only"),
                             ft_decision("p3", "abstract_only")],
            "completed_stages": list(STAGES[:STAGES.index("extraction")]), **kw}


def run_stage(tmp_path, llm=None, st=None, **kw):
    build_index(tmp_path / "idx")
    llm = llm or script()
    node = make_extraction(llm, FakeEmbedder(), index_dir=tmp_path / "idx", workers=1,
                           use_rerank=False, **kw)
    return node(st or state()), llm


# --- fields and queries from the protocol ------------------------------------------------

def test_fields_come_from_the_protocol():
    keys = [f.key for f in extraction_fields(PROTOCOL)]
    assert keys[:3] == ["design", "country", "setting"]
    assert {"O1_measure", "O1_timepoints"} <= set(keys)
    extra = extraction_fields({**PROTOCOL, "extraction_fields": {"adherence": "Adherence %"},
                               "pico": {**PROTOCOL["pico"],
                                        "outcomes": ["depressive symptoms", "anxiety"]}})
    assert {"O2_measure", "adherence"} <= {f.key for f in extra}
    # The outcome query carries the criterion that names its instruments (BM25 finds tables).
    assert "GDS-15" in group_queries(PROTOCOL)["O1"]


def test_number_helpers():
    assert numbers("n = 1,234; 72.3 (5.1)") == [1234.0, 72.3, 5.1]
    assert number_in(-1.5, "difference -1.5") and number_in(72.3, "72.3 years")
    assert not number_in(72.0, "72.3 years")


def test_number_words_are_read_but_totals_are_not_computed():
    assert spelled_numbers("Thirty two participants") == [32.0]
    assert spelled_numbers("Fifty-two subjects; one hundred and twenty") == [52.0, 120.0]
    assert spelled_numbers("twice a week for twelve weeks") == [2.0, 12.0]
    chunks = [{"chunk_id": "x::0", "text": "Thirty two participants completed the trial "
               "(n = 16 per group). Forty-four in the PE group and 35 in the control group."}]
    fields, flags = check_fields([
        field("n_analysed", "Thirty two participants completed the trial", "32"),
        field("n_enrolled", "Forty-four in the PE group and 35 in the control group", "79"),
    ], extraction_fields(PROTOCOL), chunks)
    assert fields["n_analysed"]["status"] == "reported"
    assert fields["n_enrolled"]["status"] == NOT_REPORTED  # 44 + 35 is computed, not printed


def test_result_numbers_and_arm_sizes():
    text = ("GDS-15 after training: exercise 2.18 (1.24), control 4.33 (2.06). "
            "A significant decrease was found.")
    chunks = [{"chunk_id": "x::0", "text": text}]
    raw = [{"outcome": "O1", "quote": "exercise 2.18 (1.24), control 4.33 (2.06)",
            "arms": [{"arm": "exercise", "n": 17, "mean": 2.18, "sd": 1.24},
                     {"arm": "control", "n": 18, "mean": 4.33, "sd": 2.06}]}]
    # Arm sizes printed in the verified sample-size quote are accepted.
    (res,), _ = check_results(raw, ["O1"], chunks, ("17 intervention, 18 control",))
    assert [a["n"] for a in res["arms"]] == [17, 18]
    (res,), _ = check_results(raw, ["O1"], chunks)
    assert [a["n"] for a in res["arms"]] == [None, None] and res["status"] == "reported"
    # Arm names without one printed number are not a result.
    empty = [{"outcome": "O1", "quote": "A significant decrease was found",
              "arms": [{"arm": "exercise", "n": 17, "mean": 3.0}, {"arm": "control"}]}]
    (res,), flags = check_results(empty, ["O1"], chunks, ("17 intervention",))
    assert res["status"] == NOT_REPORTED and "result_no_numbers_O1" in flags


# --- never invented ----------------------------------------------------------------------

def test_values_need_a_verified_quote_and_printed_numbers():
    chunks = [{"chunk_id": "x::0", "text": METHODS}]
    specs = extraction_fields(PROTOCOL)
    fields, flags = check_fields([
        field("design", "In this randomized controlled trial", "RCT"),
        field("n_enrolled", "we randomized 120 participants", "120 (60 per arm)"),
        field("sex", "all participants were women", "women"),
        field("setting", "", "community"),
        field("age", "a mean age of 72.3 years", "72.3"),
        field("country", "", "NR"),
    ], specs, chunks)
    assert fields["design"]["status"] == "reported" and fields["design"]["chunk_id"] == "x::0"
    assert fields["age"]["value"] == "72.3"
    n = fields["n_enrolled"]
    assert n["status"] == NOT_REPORTED and n["value"] == NOT_REPORTED
    assert n["value_claimed"] == "120 (60 per arm)" and n["numbers_not_in_quote"] == [60.0]
    assert fields["sex"]["status"] == NOT_REPORTED and fields["setting"]["status"] == NOT_REPORTED
    assert fields["country"]["status"] == NOT_REPORTED and "value_claimed" not in fields["country"]
    assert fields["intervention"]["status"] == NOT_REPORTED       # key skipped by the model
    assert {"number_not_in_quote_n_enrolled", "quote_not_found_sex",
            "no_quote_setting"} <= set(flags)


def test_extraction_stage(tmp_path):
    update, llm = run_stage(tmp_path)
    ext = {e["paper_id"]: e for e in update["extractions"]}
    assert update["completed_stages"] == ["extraction"]
    assert all(e["round"] == 1 for e in ext.values())

    p1 = ext["p1"]
    assert p1["fields"]["design"]["status"] == "reported"
    assert p1["fields"]["country"]["value"] == "Japan"
    assert p1["fields"]["n_enrolled"]["status"] == NOT_REPORTED            # "60" invented
    assert p1["fields"]["sex"]["status"] == NOT_REPORTED                   # quote invented
    assert p1["fields"]["intervention"]["status"] == NOT_REPORTED          # skipped
    assert p1["fields"]["comparator"]["chunk_id"].startswith("p1::")       # cited by code
    assert p1["design_category_verified"] and "abstract_only" not in p1["flags"]
    # Screening's arm pair is passed on, so both agents judge the same comparison.
    assert p1["comparison_hint"] == "walking arm vs usual care"
    prompt = next(p for a, p in llm.calls if a == "extraction" and "FULL" in p)
    assert "walking arm vs usual care" in prompt and METHODS.strip()[:40] in prompt

    (res,) = p1["results"]
    arms = {a["arm"]: a for a in res["arms"]}
    assert res["status"] == "reported" and arms["control"]["mean"] == 5.6
    assert arms["exercise"]["n"] is None and arms["exercise"]["sd"] == 2.0  # n=60 not printed
    est = res["estimates"][0]
    assert (est["value"], est["ci_low"], est["ci_high"]) == (-1.5, -2.3, -0.7)
    assert est["p_value"] == ""                                            # 0.01 not printed
    assert "exercise.n=60" in res["numbers_dropped"]

    p2 = ext["p2"]  # abstract only: extracted from the abstract, flagged
    assert "abstract_only" in p2["flags"] and p2["fields"]["country"]["value"] == "Taiwan"
    assert p2["fields"]["design"]["status"] == NOT_REPORTED  # 'randomized' vs 'randomised'

    p3 = ext["p3"]  # not in the index, LLM error: still written, nothing invented
    assert {"abstract_only", "not_in_index"} <= set(p3["flags"])
    assert any(f.startswith("llm_error") for f in p3["flags"])
    assert p3["reported"] == 0 and p3["requested"] == len(extraction_fields(PROTOCOL))

    audit = update["audit_log"][0]
    assert audit["studies"] == 3 and audit["flags"]["llm_error"] == 1
    assert audit["reported_by_text"]["abstract_only"].startswith("1/")


def test_risk_of_bias_stage(tmp_path):
    update, _ = run_stage(tmp_path)
    rob = {r["paper_id"]: r for r in update["risk_of_bias"]}

    p1 = rob["p1"]
    assert (p1["tool"], p1["tool_chosen_by"]) == ("rob2", "extracted design")
    assert p1["outcome"] == "depressive symptoms"
    d = {x["domain"]: x for x in p1["domains"]}
    assert d["D1"]["judgment"] == "low" and d["D1"]["verified"]
    assert d["D2"]["judgment"] == "some concerns" and d["D2"]["judgment_claimed"] == "high"
    assert d["D3"]["judgment"] == "some concerns"                           # domain skipped
    assert d["D4"]["judgment"] == "some concerns"                           # not a RoB 2 level
    assert p1["overall"] == "some concerns"
    assert {"unverified_D2", "domain_missing_D3", "invalid_level_D4"} <= set(p1["flags"])

    # Design not verified (abstract says 'randomised'): the protocol's single design decides.
    assert rob["p2"]["tool"] == "rob2" and rob["p2"]["tool_chosen_by"].startswith("protocol")
    assert rob["p3"]["overall"] == "some concerns"                          # no evidence at all


def test_tool_choice():
    assert choose_tool("randomized trial", True, {}) == ("rob2", "extracted design")
    assert choose_tool("non-randomized intervention study", True, {})[0] == "robins_i"
    assert choose_tool("diagnostic accuracy study", True, {})[0] == "quadas2"
    assert choose_tool("other", True, PROTOCOL)[0] == "checklist"  # verified design wins
    assert choose_tool("randomized trial", False, PROTOCOL)[0] == "rob2"
    assert choose_tool("randomized trial", False, {"study_designs": []}) == (
        "checklist", "default (design not verified)")
    assert choose_tool(None, False, {"study_designs": ["RCT", "cohort"]})[0] == "checklist"


def test_overall_and_no_evidence_levels():
    rob2, robins, quadas, chk = (TOOLS[k] for k in ("rob2", "robins_i", "quadas2", "checklist"))
    assert overall(rob2, ["low"] * 5) == "low"
    assert overall(rob2, ["low", "high", "some concerns", "low", "low"]) == "high"
    assert overall(robins, ["low", "moderate", "no information"]) == "no information"
    assert overall(robins, ["low", "serious", "no information"]) == "serious"
    assert overall(robins, ["critical", "moderate"]) == "critical"
    assert overall(quadas, ["high", "low", "low", "low"]) is None  # no summary score
    assert overall(chk, ["low", "unclear"]) == "unclear"
    chunks = [{"chunk_id": "c::0", "text": "Assessors were blinded."}]
    for tool, level in ((robins, "no information"), (quadas, "unclear"), (chk, "unclear")):
        raw = [{"domain": did, "quote": "invented words", "judgment": "low", "reason": "r"}
               for did, _, _ in tool.domains]
        domains, flags = check_domains(raw, tool, chunks)
        assert all(d["judgment"] == level for d in domains)
        assert len(flags) == len(tool.domains)


def test_rob2_d1_low_needs_an_allocation_method():
    chunks = [{"chunk_id": "c::0", "text": "Participants were randomly assigned to two "
               "groups. Allocation was concealed using sealed opaque envelopes."}]
    raw = [{"domain": "D1", "quote": "Participants were randomly assigned to two groups",
            "judgment": "low", "reason": "r"}]
    (d1, *_), flags = check_domains(raw, TOOLS["rob2"], chunks)
    assert d1["judgment"] == "some concerns" and "d1_no_method" in flags
    raw[0]["quote"] = "Allocation was concealed using sealed opaque envelopes"
    (d1, *_), flags = check_domains(raw, TOOLS["rob2"], chunks)
    assert d1["judgment"] == "low" and "d1_no_method" not in flags


def test_schemas_reject_invalid_values():
    with pytest.raises(ValidationError):
        RiskOfBias.model_validate({"domains": [{"domain": "D1", "judgment": "fine",
                                                "reason": "r"}]})
    with pytest.raises(ValidationError):
        Extraction.model_validate({"fields": [], "design_category": "case series"})
    ok = Extraction.model_validate({"fields": [], "design_category": "other"})
    assert ok.results == []


# --- rounds, re-ranking, graph ----------------------------------------------------------------

def test_second_round_is_read_through_latest(tmp_path):
    first, _ = run_stage(tmp_path)
    st = state(completed_stages=state()["completed_stages"] + ["extraction"],
               extractions=first["extractions"])
    second, _ = run_stage(tmp_path, st=st)
    assert {e["round"] for e in second["extractions"]} == {2}
    items = latest(first["extractions"] + second["extractions"])
    assert len(items) == 3 and {e["round"] for e in items} == {2}


def test_rerank_is_used_and_usage_counted(tmp_path):
    build_index(tmp_path / "idx")
    llm = script()
    node = make_extraction(llm, FakeEmbedder(), index_dir=tmp_path / "idx", workers=4,
                           passages=2, use_rerank=True)
    update = node(state())
    reranks = [a for a, _ in llm.calls if a == "reranker"]
    # p1 and p2 are in the index: 4 extraction groups + 5 RoB 2 domains each.
    assert len(reranks) == 2 * (len(group_queries(PROTOCOL)) + 5)
    p1 = next(e for e in update["extractions"] if e["paper_id"] == "p1")
    assert all(len(ids) <= 2 for ids in p1["passages"].values())
    assert update["audit_log"][0]["llm_calls"] == len(llm.calls) - 1  # p3 extraction raised


def test_missing_index_extracts_from_abstracts(tmp_path):
    node = make_extraction(script(), FakeEmbedder(), index_dir=tmp_path / "none", workers=1,
                           use_rerank=False)
    update = node(state())
    assert all("not_in_index" in e["flags"] for e in update["extractions"])
    assert update["audit_log"][0]["index"] is None


def test_graph_runs_extraction_and_coordinator_sees_it(tmp_path):
    build_index(tmp_path / "idx")
    llm = script(coordinator=[{"next": s, "rationale": s} for s in STAGES])
    node = make_extraction(llm, FakeEmbedder(), index_dir=tmp_path / "idx", workers=1,
                           use_rerank=False)
    graph = build_graph(llm, stages={"extraction": node})
    init = {k: v for k, v in state().items() if k != "completed_stages"}
    final = graph.invoke(init, run_config("x"))
    assert final["completed_stages"] == list(STAGES)
    assert len(final["extractions"]) == 3 and len(final["risk_of_bias"]) == 3
    assert "Extraction (round 1): 3 studies" in summarize_state(final)
