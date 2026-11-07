"""Part 8: 2x2 extraction (step 0), meta-analysis maths, synthesis paths, narrative check,
critic and the critic loop through the graph. Scripted LLM, fake embedder, no network."""

import math
import re

import pytest

from sr_pipeline.agents.critic import loops_used, make_critic
from sr_pipeline.agents.extraction import (DiagnosticExtraction, check_two_by_two,
                                           make_extraction)
from sr_pipeline.agents.synthesis import (arm_pair, check_narrative, make_synthesis,
                                          number_supported, study_list, synthesise_outcome)
from sr_pipeline.coordinator import loop_rule, summarize_state
from sr_pipeline.evaluation.topics import load_topic, seed_protocol
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.meta import (dersimonian_laird, hedges_g, pooled_proportion, sign_test,
                              wilson)
from sr_pipeline.protocol import Protocol, better_direction
from sr_pipeline.rag import FakeEmbedder, HybridIndex, chunk_paper
from sr_pipeline.sources.fulltext import FullText, Section

# --- maths: hand-computed references (independent arithmetic, not this code) -------------
# Three trials, intervention vs control, n / mean / SD per arm:
#   A: 20, 10, 4 vs 20, 12, 4   -> s_p = 4, d = -0.5,      J = 1 - 3/151 = 0.980132
#   B: 30,  8, 5 vs 30,  9, 5   -> s_p = 5, d = -0.2,      J = 1 - 3/231 = 0.987013
#   C: 25, 15, 6 vs 25, 20, 6   -> s_p = 6, d = -0.833333, J = 1 - 3/191 = 0.984293
# g = J d; V_g = J^2 ((n1+n2)/(n1 n2) + d^2 / (2 (n1+n2))):
#   A: g -0.490066, V 0.099068;  B: g -0.197403, V 0.065271;  C: g -0.820244, V 0.084235
# DerSimonian-Laird: w = 1/V = 10.0941, 15.3207, 11.8716; fixed -0.474938;
#   Q = 2.597931 (df 2); C = sum w - sum w^2 / sum w = 24.478786; tau^2 = 0.024426;
#   pooled -0.482186, SE 0.187485, 95% CI -0.849650 to -0.114723; I^2 = 0.230156;
#   weights 0.284633, 0.391879, 0.323488.
TRIALS = [(20, 10, 4, 20, 12, 4), (30, 8, 5, 30, 9, 5), (25, 15, 6, 25, 20, 6)]


def test_hedges_g_and_dersimonian_laird_match_hand_computation():
    gv = [hedges_g(*t) for t in TRIALS]
    for (g, v), (g_ref, v_ref) in zip(gv, [(-0.490066, 0.099068), (-0.197403, 0.065271),
                                           (-0.820244, 0.084235)]):
        assert g == pytest.approx(g_ref, abs=1e-6) and v == pytest.approx(v_ref, abs=1e-6)
    m = dersimonian_laird([g for g, _ in gv], [v for _, v in gv])
    ref = {"fixed": -0.474938, "q": 2.597931, "tau2": 0.024426, "pooled": -0.482186,
           "se": 0.187485, "ci_low": -0.849650, "ci_high": -0.114723, "i2": 0.230156}
    for k, x in ref.items():
        assert m[k] == pytest.approx(x, abs=1e-5), k
    assert m["weights"] == pytest.approx([0.284633, 0.391879, 0.323488], abs=1e-5)
    assert m["df"] == 2 and m["p"] < 0.05
    # Homogeneous effects: Q < df -> tau^2 = 0 and I^2 = 0 (fixed = random).
    h = dersimonian_laird([0.3, 0.3, 0.3], [0.1, 0.2, 0.3])
    assert h["tau2"] == 0 and h["i2"] == 0 and h["pooled"] == pytest.approx(0.3)


def test_wilson_interval_known_values():
    # 5/10 -> (0.2366, 0.7634); 0/10 -> (0, 0.2775); 10/10 -> (0.7225, 1)
    # (Wilson 1927; e.g. Newcombe 1998, Table I method 3).
    assert wilson(5, 10)[1:] == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson(0, 10) == pytest.approx((0.0, 0.0, 0.2775), abs=1e-4)
    assert wilson(10, 10)[1:] == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(81, 263)[1:] == pytest.approx((0.2553, 0.3662), abs=1e-4)


def test_pooled_proportion_and_sign_test():
    # Logit pooling of identical proportions returns that proportion.
    p = pooled_proportion([90, 45, 180], [100, 50, 200])
    assert p["pooled"] == pytest.approx(0.9) and p["tau2"] == 0
    # A zero cell gets the 0.5 correction instead of an infinite logit.
    z = pooled_proportion([10, 20, 30], [10, 21, 32])
    assert 0.8 < z["pooled"] < 1 and math.isfinite(z["logit_pooled"])
    assert sign_test(5, 0) == pytest.approx(0.0625)      # 2 * 0.5^5
    assert sign_test(3, 1) == pytest.approx(0.625)       # 2 * (1 + 4) / 16
    assert sign_test(0, 0) is None


# --- step 0: 2x2 tables --------------------------------------------------------------------

ROW = ("Apple Watch vs 12-lead ECG: TP 34, FP 2, FN 4, TN 60; sensitivity 89.5%, "
       "specificity 96.8%.")


def two(quote=ROW, **kw):
    base = {"index_test": "Apple Watch", "reference_standard": "12-lead ECG",
            "threshold": "", "quote": quote, "chunk_id": "",
            "tp": 34, "fp": 2, "fn": 4, "tn": 60, "sensitivity": 89.5, "specificity": 96.8}
    return base | kw


def test_two_by_two_consistent_inconsistent_incomplete():
    chunks = [{"chunk_id": "d::1", "text": "[Table 2] " + ROW + " Sensitivity was 0.80."}]
    (t,), flags = check_two_by_two([two()], chunks)
    assert t["status"] == "reported" and t["complete"] and t["consistent"] is True
    assert t["chunk_id"] == "d::1" and not flags
    # Printed sensitivity disagrees with the cells (34/38 = 89.5%, not 0.80): flagged, kept.
    q = ROW + " Sensitivity was 0.80."
    (t,), flags = check_two_by_two([two(q, sensitivity=0.80)], chunks)
    assert t["consistent"] is False and t["tp"] == 34
    assert t["mismatch"]["sensitivity"]["from_cells"] == pytest.approx(0.8947, abs=1e-4)
    assert "2x2_inconsistent_sensitivity_1" in flags
    # A cell that is not printed is dropped, never derived from n or sensitivity.
    (t,), flags = check_two_by_two([two(tn=61)], chunks)
    assert t["tn"] is None and not t["complete"] and t["consistent"] is None
    assert {"number_not_in_quote_2x2_1", "2x2_incomplete_1"} <= set(flags)
    (t,), flags = check_two_by_two([two("TP 34 and FP 2 in a made-up sentence")], chunks)
    assert t["status"] == "not reported" and "2x2_quote_not_found_1" in flags


DIAG_PROTOCOL = {
    "question": "How accurate are smartwatches at detecting atrial fibrillation?",
    "pico": {"population": "adults", "intervention": "smartwatch ECG",
             "comparator": "12-lead ECG", "outcomes": ["diagnostic accuracy"]},
    "inclusion_criteria": [], "exclusion_criteria": [], "study_designs": [],
    "outcome_directions": [{"outcome": "diagnostic accuracy", "better": "higher"}],
}
DTA = {"paper_id": "dta", "title": "WATCH accuracy study", "year": 2020,
       "abstract": "A diagnostic accuracy study of a smartwatch.", "round": 1}
RCT = {"paper_id": "rct", "title": "TRIAL of watches", "year": 2021,
       "abstract": "A randomized trial of watch feedback.", "round": 1}
UNV = {"paper_id": "unv", "title": "UNVERIFIED ring study", "year": 2019,
       "abstract": "Smart ring readings were compared with Holter monitoring.", "round": 1}


def diag_index(folder):
    ft = FullText(paper_id="dta", status="pmc_xml", sections=[
        Section(title="Methods", text="This prospective diagnostic accuracy study enrolled "
                "100 patients. " * 5),
        Section(title="Table 2", text=ROW)])
    HybridIndex.build(chunk_paper(DTA, ft) + chunk_paper(RCT, None) + chunk_paper(UNV, None),
                      FakeEmbedder()).save(folder)


def diag_extractor(prompt):
    if prompt.startswith("You extract diagnostic accuracy results"):
        assert "TRIAL" not in prompt, "2x2 call made for a non-diagnostic study"
        return DiagnosticExtraction(tables=[two(chunk_id="")] if "WATCH" in prompt else [])
    if "UNVERIFIED" in prompt:  # design quote not in the passages: category unverified
        return {"fields": [{"key": "design", "quote": "a diagnostic accuracy study",
                            "value": "diagnostic accuracy study"}],
                "design_category": "diagnostic accuracy study", "results": []}
    if "WATCH" in prompt:
        return {"fields": [{"key": "design", "quote": "diagnostic accuracy study",
                            "value": "diagnostic accuracy study"}],
                "design_category": "diagnostic accuracy study", "results": []}
    return {"fields": [{"key": "design", "quote": "A randomized trial", "value": "RCT"}],
            "design_category": "randomized trial", "results": []}


def test_two_by_two_extracted_for_diagnostic_studies_only(tmp_path):
    diag_index(tmp_path / "idx")
    llm = ScriptedLLM({"extraction": diag_extractor,
                       "risk_of_bias": lambda p: {"domains": []},
                       "reranker": lambda p: {"scores": []}})
    node = make_extraction(llm, FakeEmbedder(), index_dir=tmp_path / "idx", workers=1,
                           use_rerank=False)
    update = node({"protocol": DIAG_PROTOCOL, "unique_records": [DTA, RCT, UNV],
                   "included": ["dta", "rct", "unv"], "ft_decisions": [],
                   "completed_stages": list(STAGES[:STAGES.index("extraction")])})
    ext = {e["paper_id"]: e for e in update["extractions"]}
    (t,) = ext["dta"]["diagnostic"]
    assert t["complete"] and t["consistent"] and t["chunk_id"].startswith("dta::")
    assert ext["rct"]["diagnostic"] is None                 # RoB 2 study: no 2x2 call
    # Classified diagnostic without a verified design quote: checklist RoB, 2x2 still tried.
    assert not ext["unv"]["design_category_verified"] and ext["unv"]["diagnostic"] == []
    rob = {r["paper_id"]: r["tool"] for r in update["risk_of_bias"]}
    assert rob == {"dta": "quadas2", "rct": "rob2", "unv": "checklist"}
    diag_calls = [p for a, p in llm.calls if a == "extraction"
                  and p.startswith("You extract diagnostic")]
    assert len(diag_calls) == 2
    assert "[Table 2]" in next(p for p in diag_calls if "WATCH" in p)  # tables always added
    assert "diagnostic_tables" in ext["dta"]["passages"]
    assert update["audit_log"][0]["two_by_two"] == {"studies_assessed": 2, "reported": 1,
                                                    "complete_consistent": 1}


# --- synthesis -------------------------------------------------------------------------------

PROTOCOL = {
    "question": "Does exercise reduce depressive symptoms in older adults?",
    "pico": {"population": "older adults", "intervention": "exercise",
             "comparator": "usual care", "outcomes": ["depressive symptoms"]},
    "study_designs": ["randomized controlled trial"],
    "outcome_directions": [{"outcome": "depressive symptoms", "better": "lower"}],
}


def arm(name, n, mean, sd, tp="post-intervention"):
    return {"arm": name, "timepoint": tp, "n": n, "mean": mean, "sd": sd}


def ext_item(pid, arms=(), estimates=(), abstract=False, design="randomized trial"):
    results = []
    if arms or estimates:
        results = [{"outcome": "O1", "status": "reported", "quote": "q", "arms": list(arms),
                    "estimates": list(estimates)}]
    return {"paper_id": pid, "round": 1, "flags": ["abstract_only"] if abstract else [],
            "fields": {"O1_measure": {"value": "GDS-15", "status": "reported"},
                       "n_enrolled": {"value": "40", "status": "reported"}},
            "results": results, "design_category": design, "reported": 2, "requested": 14,
            "design_category_verified": True, "diagnostic": None}


HET = {"outcome": "O1", "studies": ["S1"],
       "text": "I2 and tau2 are in the tables; doses and instruments varied."}


def records(n):
    return [{"paper_id": f"p{i}", "title": f"Trial {i}", "year": 2010 + i,
             "authors": [f"Author{i}, A"]} for i in range(1, n + 1)]


def rob(pid, overall="some concerns"):
    return {"paper_id": pid, "round": 1, "tool": "rob2", "tool_chosen_by": "extracted design",
            "overall": overall, "flags": []}


def ft_counts(n):
    return {"round": 1, "reports_sought": n, "reports_not_retrieved": 0,
            "studies_included": n}


def syn_state(exts, **kw):
    ids = [e["paper_id"] for e in exts]
    return {"question": PROTOCOL["question"], "protocol": PROTOCOL,
            "unique_records": records(len(ids)), "included": ids, "extractions": exts,
            "risk_of_bias": [rob(i) for i in ids],
            "prisma_counts": {"full_text": ft_counts(len(ids))},
            "completed_stages": list(STAGES[:STAGES.index("synthesis")])} | kw


def trial_exts(k):
    return [ext_item(f"p{i}", [arm("exercise", *t[:3]), arm("usual care", *t[3:])])
            for i, t in zip(range(1, k + 1), TRIALS)]


def cite_all(prompt):
    labels = sorted(set(re.findall(r"^(S\d+) \|", prompt, re.M)))
    return {"heterogeneity": [HET | {"studies": labels}], "sentences": [
        {"section": "O1", "text": "Studies were grouped by outcome and compared by direction "
         "of effect.", "studies": labels},
        {"section": "overall", "text": "Risk of bias was some concerns in most studies.",
         "studies": labels}]}


def test_arm_pair_and_direction():
    i, c, tp = arm_pair([arm("Health Qigong", 17, 2.18, 1.24), arm("control group", 18,
                                                                    4.33, 2.06)])
    assert (i["arm"], c["arm"], tp) == ("Health Qigong", "control group", "post-intervention")
    # Multi-arm without a single pair -> no pair (never guessed).
    assert arm_pair([arm("walking", 10, 1, 1), arm("yoga", 10, 1, 1),
                     arm("control", 10, 2, 1)]) is None
    studies = study_list(syn_state([
        ext_item("p1", [arm("exercise", 16, 15.06, 1.83), arm("control", 16, 12.19, 1.47)]),
        ext_item("p2", estimates=[{"measure": "p-value", "value": None, "p_value": "0.003"}]),
        ext_item("p3", [arm("exercise", 26, 2.5, None), arm("control", 26, 4.0, None)])]))
    g = synthesise_outcome("O1", "depressive symptoms", "lower", studies)
    per = {p["label"]: p for p in g["per_study"]}
    assert per["S1"]["direction"] == "favours control" and per["S1"]["g"] > 0
    assert per["S2"]["direction"] == "not extractable"     # a p value alone has no direction
    assert per["S3"]["direction"] == "favours intervention" and per["S3"]["g"] is None
    assert g["vote_count"]["favours intervention"] == 1 and g["method"] == "narrative (SWiM)"
    unknown = synthesise_outcome("O1", "depressive symptoms", "unknown", studies)
    assert all(p["direction"] == "not extractable" for p in unknown["per_study"])


def test_two_comparable_studies_give_narrative_only(tmp_path):
    llm = ScriptedLLM({"synthesis": cite_all})
    update = make_synthesis(llm, out_dir=tmp_path / "fig")(syn_state(trial_exts(2)))
    s = update["synthesis"]
    (g,) = s["outcomes"]
    assert g["method"] == "narrative (SWiM)" and g["meta"] is None and g["plot"] is None
    assert g["effects"] == 2 and not (tmp_path / "fig").exists()
    assert "vote counting" in g["reason"]
    assert s["checks"] == {} and s["round"] == 1


def test_three_comparable_studies_are_pooled_with_a_forest_plot(tmp_path):
    llm = ScriptedLLM({"synthesis": cite_all})
    update = make_synthesis(llm, out_dir=tmp_path / "fig")(syn_state(trial_exts(3)))
    (g,) = update["synthesis"]["outcomes"]
    assert g["method"] == "meta-analysis" and g["meta"]["k"] == 3
    assert g["meta"]["pooled"] == pytest.approx(-0.482186, abs=1e-5)
    assert g["vote_count"]["favours intervention"] == 3
    png = tmp_path / "fig" / "forest_O1_r1.png"
    assert g["plot"] == str(png) and png.read_bytes()[:4] == b"\x89PNG"
    evidence = update["synthesis"]["evidence"]
    assert "SMD (Hedges' g) -0.48 (95% CI -0.85 to -0.11)" in evidence
    assert "risk of bias rob2: some concerns" in evidence


def test_diagnostic_synthesis_pools_three_consistent_tables(tmp_path):
    def dta_ext(pid, **kw):
        t = {"status": "reported", "index_test": "watch", "reference_standard": "ECG",
             "threshold": "", "complete": True, "consistent": True,
             "tp": 34, "fp": 2, "fn": 4, "tn": 60} | kw
        return ext_item(pid, design="diagnostic accuracy study") | {"diagnostic": [t]}
    exts = [dta_ext("p1"), dta_ext("p2", tp=45, fn=5), dta_ext("p3", tp=20, fn=10),
            dta_ext("p4", consistent=False)]
    st = syn_state(exts, protocol=DIAG_PROTOCOL)
    update = make_synthesis(ScriptedLLM({"synthesis": cite_all}),
                            out_dir=tmp_path / "fig")(st)
    d = update["synthesis"]["diagnostic"]
    assert d["studies_with_2x2"] == 3 and d["pooled"]["studies"] == ["S1", "S2", "S3"]
    s4 = d["per_study"][3]
    assert "sens" not in s4 and "disagree" in s4["reason"]   # inconsistent: left out
    assert d["per_study"][0]["sens"][0] == pytest.approx(34 / 38)
    assert (tmp_path / "fig" / "forest_DTA_r1.png").exists()
    assert "sensitivity 89.5% (95% CI" in update["synthesis"]["evidence"]


def test_narrative_check_flags_unknown_studies_and_unsupported_numbers():
    evidence = "S1: GDS-15 exercise mean 5.3 SD 2.5; Hedges' g -0.4821 (95% CI -0.85 to -0.11)"
    out = check_narrative([
        {"section": "O1", "text": "Exercise lowered GDS-15 scores (g -0.48).",
         "studies": ["S1"]},
        {"section": "O1", "text": "One trial reported 7.4 points less.", "studies": ["S9"]},
        {"section": "overall", "text": "Evidence is limited.", "studies": []},
        {"section": "O1", "text": "Twelve studies agreed.", "studies": ["S1"]},
    ], ["S1", "S2"], evidence, ["O1", "overall"])
    assert out[0]["flags"] == []                                  # -0.48 rounds -0.4821
    assert "unknown_study: S9" in out[1]["flags"] and "no_citation" in out[1]["flags"]
    assert "number_not_supported: 7.4" in out[1]["flags"]
    assert out[2]["flags"] == ["no_citation"]
    assert out[3]["flags"] == ["number_not_supported: 12"]       # spelled numbers count
    assert number_supported("2", [2.18]) is False                 # integers must match
    assert number_supported("2.2", [2.18]) is True


# --- critic and loop -------------------------------------------------------------------------

def critic_state(tmp_path, narrative, n=3, **kw):
    st = syn_state(trial_exts(n), **kw)
    llm = ScriptedLLM({"synthesis": lambda p: narrative})
    update = make_synthesis(llm, out_dir=tmp_path)(st)
    return st | {"synthesis": update["synthesis"],
                 "completed_stages": st["completed_stages"] + ["synthesis"]}


def test_critic_code_checks(tmp_path):
    narrative = {"sentences": [{"section": "overall", "text": "Exercise was studied in 99 "
                                "trials.", "studies": ["S1"]}]}
    st = critic_state(tmp_path, narrative, prisma_counts={"full_text": ft_counts(4)})
    llm = ScriptedLLM({"critic": [{"issues": [
        {"sentence": 1, "problem": "invented count", "evidence": "not in the tables at all",
         "severity": "blocker"}]}]})
    qa = make_critic(llm)(st)["qa_report"]
    kinds = {i["kind"]: i["severity"] for i in qa["issues"]}
    assert kinds["outcome_not_addressed"] == "blocker"
    assert kinds["number_not_supported"] == "blocker"
    assert kinds["count_mismatch"] == "blocker"
    assert kinds["study_not_cited"] == "blocker"
    # LLM blocker whose evidence quote is not in the tables -> downgraded to a warning.
    assert kinds["llm_unsupported_claim"] == "warning"
    assert not qa["passed"] and qa["loop_back"] and qa["loops_used"] == 0


def test_critic_passes_and_llm_blocker_needs_verified_evidence(tmp_path):
    narrative = {"sentences": [{"section": "O1", "text": "All trials favoured exercise.",
                                "studies": ["S1", "S2", "S3"]}], "heterogeneity": [HET]}
    st = critic_state(tmp_path, narrative)
    qa = make_critic(ScriptedLLM({"critic": [{"issues": []}]}))(st)["qa_report"]
    assert qa["passed"] and not qa["loop_back"] and qa["issues"] == []
    line = "favours intervention 3, favours control 0"
    llm = ScriptedLLM({"critic": [{"issues": [{"sentence": 1, "problem": "p", "evidence": line,
                                               "severity": "blocker"}]}]})
    qa = make_critic(llm)(st)["qa_report"]
    assert qa["blockers"] == 1 and qa["issues"][0]["evidence_verified"]


def test_loops_used_and_loop_rule():
    base = list(STAGES[:STAGES.index("synthesis")])
    assert loops_used(base + ["synthesis", "critic"]) == 0
    assert loops_used(base + ["synthesis", "critic", "synthesis", "critic"]) == 1
    # A new extraction resets the count.
    assert loops_used(base + ["synthesis", "critic", "extraction", "synthesis"]) == 0
    done = base + ["synthesis", "critic"]
    st = {"completed_stages": done, "qa_report": {"passed": True, "loop_back": False}}
    assert "no blocker" in loop_rule("synthesis", st)
    assert loop_rule("report", st) is None
    assert loop_rule("synthesis", st | {"qa_report": {"loop_back": True}}) is None
    assert loop_rule("synthesis", {"completed_stages": done}) is None   # placeholder critic


def loop_graph(tmp_path, synth_answers, coordinator):
    llm = ScriptedLLM({"synthesis": synth_answers, "critic": lambda p: {"issues": []},
                       "coordinator": [{"next": s, "rationale": s} for s in coordinator]})
    graph = build_graph(llm, stages={"synthesis": make_synthesis(llm, out_dir=tmp_path),
                                     "critic": make_critic(llm)})
    init = {k: v for k, v in syn_state(trial_exts(3)).items() if k != "completed_stages"}
    final = graph.invoke(init, run_config("loop"))
    # The coordinator's fallback would hide a crash in its summary: none allowed here.
    assert not [e for e in final["audit_log"] if e.get("outcome") == "llm_error"]
    return final, llm


def missing_s2(prompt):
    return {"sentences": [{"section": "O1", "text": "Exercise was compared with usual care.",
                           "studies": ["S1", "S3"]}], "heterogeneity": [HET]}


def test_critic_loop_reruns_synthesis_with_feedback_then_reports(tmp_path):
    def answer(prompt):
        if "A critic found these problems" in prompt:
            assert "study_not_cited: S2" in prompt
            return {"sentences": [{"section": "O1", "text": "Exercise was compared with "
                                   "usual care.", "studies": ["S1", "S2", "S3"]}],
                    "heterogeneity": [HET]}
        return missing_s2(prompt)

    final, llm = loop_graph(tmp_path, answer, list(STAGES[:-1]) + ["synthesis", "critic",
                                                                     "report"])
    assert final["completed_stages"][-5:] == ["synthesis", "critic", "synthesis", "critic",
                                              "report"]
    assert final["qa_report"]["passed"] and final["synthesis"]["round"] == 2
    assert final["synthesis"]["feedback"] and "Critic (round 2" in summarize_state(final)


def test_critic_loop_stops_after_two_loops(tmp_path):
    coordinator = list(STAGES[:-1]) + ["synthesis", "critic"] * 2 + ["synthesis", "report"]
    final, _ = loop_graph(tmp_path, missing_s2, coordinator)
    done = final["completed_stages"]
    assert done.count("synthesis") == 3 and done[-1] == "report"     # 1 run + 2 loops
    qa = final["qa_report"]
    assert not qa["passed"] and not qa["loop_back"] and qa["loops_used"] == 2
    assert "loop budget spent" in qa["note"]
    refused = [e for e in final["audit_log"] if e["node"] == "coordinator"
               and e.get("outcome") == "refused"]
    assert refused[-1]["proposed"] == "synthesis" and refused[-1]["substituted"] == "report"


# --- protocol: direction of effect -----------------------------------------------------------

def test_outcome_directions_follow_outcomes_and_topic_seeding():
    p = Protocol.model_validate({
        "title": "t", "question": "q", "databases": ["openalex"], "rationale": "r",
        "pico": {"population": "a", "intervention": "b", "comparator": "c",
                 "outcomes": ["depressive symptoms"]},
        "inclusion_criteria": [], "exclusion_criteria": [], "study_designs": [],
        "outcome_directions": [{"outcome": "depressive symptoms", "better": "lower"},
                               {"outcome": "anxiety", "better": "lower"}]})
    assert [d.outcome for d in p.outcome_directions] == ["depressive symptoms"]
    assert better_direction(p.model_dump(), "depressive symptoms") == "lower"
    assert better_direction(p.model_dump(), "anxiety") == "unknown"
    seeded, audit = seed_protocol(p.model_dump(mode="json") | {"outcome_directions": []},
                                  load_topic("T3"))
    assert better_direction(seeded, "depressive symptoms") == "lower"
    assert "outcome_directions" in audit["seeded_fields"]


def test_table_row_quote_may_skip_columns_inside_one_passage():
    from sr_pipeline.agents.extraction import check_results, table_quote_found
    row = ("[Table 3] | Exercise (n = 27) | Horticulture (n = 26) | Control (n = 28) "
           "GDS-15, score | 7.1 (2.5) | 5.3 (2.5) | 0.002 | 6.9 (4.7) | 4.7 (2.7) | <0.001 | "
           "6.4 (2.5) | 5.1 (3.1) | 0.001")
    chunks = [{"chunk_id": "m::23", "text": row},
              {"chunk_id": "m::24", "text": "[Table 4] GDS-15, score | 9.9 (1.1)"}]
    skip = "GDS-15, score | 7.1 (2.5) | 5.3 (2.5) | 0.002 | 6.4 (2.5) | 5.1 (3.1) | 0.001"
    assert table_quote_found(skip, chunks) == "m::23"
    assert table_quote_found("GDS-15, score | 5.1 (3.1) | 5.3 (2.5)", chunks) == ""  # order
    assert table_quote_found("GDS-15, score | 5.3 (2.5) | 9.9 (1.1)", chunks) == ""  # 2 chunks
    raw = [{"outcome": "O1", "quote": skip,
            "arms": [arm("exercise", 27, 5.3, 2.5), arm("control", 28, 5.1, 3.1)]}]
    (res,), flags = check_results(raw, ["O1"], chunks)
    assert res["status"] == "reported" and res["chunk_id"] == "m::23"
    assert [a["n"] for a in res["arms"]] == [None, None]     # 27 / 28 not in the quote
    assert "table_row_quote_O1" in flags
