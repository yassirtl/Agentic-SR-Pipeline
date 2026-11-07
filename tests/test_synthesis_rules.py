"""Part 8b: synthesis comparison rules. Within-group results never give a direction, one
metric per study chosen by the protocol's order (conflicts flagged and named in the text),
a heterogeneity statement per outcome. Scripted LLM, no network."""

import pytest
from pydantic import ValidationError

from sr_pipeline.agents.critic import make_critic
from sr_pipeline.agents.extraction import check_results
from sr_pipeline.agents.synthesis import (arm_pair, estimate_role, make_synthesis,
                                          study_list, synthesise_outcome)
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import DEFAULT_METRIC_ORDER, Protocol, Template, metric_order

from test_synthesis import PROTOCOL, arm, ext_item, syn_state

LOWER = "lower"


def full_arm(name, n, mean, sd, baseline=None, change=None):
    return arm(name, n, mean, sd) | {"baseline_mean": baseline, "change_mean": change}


def one(arms=(), estimates=(), quote="q", order=tuple(DEFAULT_METRIC_ORDER)):
    e = ext_item("p1", arms, estimates)
    if e["results"]:
        e["results"][0]["quote"] = quote
    (s,) = study_list(syn_state([e]))
    g = synthesise_outcome("O1", "depressive symptoms", LOWER, [s], order=order)
    return g, g["per_study"][0]


# Makizako 2019 (Table 3, post-intervention row): exercise 7.1 -> 5.3, control 6.4 -> 5.1;
# the 0.002 is the exercise arm's own before/after p value.
MAKIZAKO_QUOTE = "GDS-15, score | 7.1 (2.5) | 5.3 (2.5) | 0.002 | 6.4 (2.5) | 5.1 (3.1) | 0.001"
MAKIZAKO_ARMS = [full_arm("exercise", None, 5.3, 2.5, baseline=7.1),
                 full_arm("control", None, 5.1, 3.1, baseline=6.4)]


# --- Fix 1: only between-group results give a direction -------------------------------------

def test_estimate_role_within_between_and_unknown():
    arms = [arm("ACTIVE", 16, None, None), arm("CON", 16, None, None)]
    assert estimate_role({"comparison": "within_group"}, "ACTIVE vs CON", arms) == (
        "within_group", False)
    assert estimate_role({"comparison": "between_group"}, "ACTIVE vs CON", arms) == (
        "between_group", True)
    # The LLM's between_group label is checked too (Huang 2015, live: no control arm named).
    huang = ("Moreover, after intervention, participants in the physical fitness exercise "
             "program group had decreased GDS-15 scores at three time-point comparisons")
    assert estimate_role({"comparison": "between_group"}, huang,
                         [arm("physical fitness exercise program group", 57, None, None),
                          arm("control group", None, None, None)]) == ("between_group", False)
    # Unstated: counts only when the quote names a control arm and another arm (Coletta).
    coletta = "An effect of the intervention was observed for depression (ACTIVE: -1.94; " \
              "CON: -0.07; p=0.015)."
    assert estimate_role({}, coletta, arms) == ("unknown", True)
    # A table row naming no arm: the p value may be within-group (Makizako) -> not counted.
    assert estimate_role({"comparison": "unknown"}, MAKIZAKO_QUOTE, MAKIZAKO_ARMS) == (
        "unknown", False)
    # 'group' alone is not an arm name.
    assert estimate_role({}, "the group improved",
                         [arm("exercise group", 1, 1, 1), arm("control group", 1, 1, 1)]
                         ) == ("unknown", False)


def test_within_group_estimate_never_gives_a_direction():
    md = {"measure": "mean difference", "value": -2.0, "p_value": "0.002",
          "contrast": "intervention minus control"}
    named = [arm("exercise", 20, None, None), arm("control", 20, None, None)]
    # Same estimate, labelled within-group: shown, not used; no other metric -> no direction.
    g, p = one(named, [md | {"comparison": "within_group"}], quote="exercise vs control")
    assert p["direction"] == "not extractable" and p["metric_used"] is None
    assert g["vote_count"]["favours intervention"] == 0
    assert "between-group estimate: no between-group difference" in p["reason"]
    # Labelled between-group, both arms named, contrast stated: metric (1) gives it.
    _, p = one(named, [md | {"comparison": "between_group"}], quote="exercise vs control")
    assert p["direction"] == "favours intervention"
    assert p["metric_used"] == "between_group_estimate"
    # A difference without a stated contrast order has no known sign: not used.
    _, p = one(named, [md | {"comparison": "between_group", "contrast": ""}],
               quote="exercise vs control")
    assert p["direction"] == "not extractable"
    assert "without a stated contrast order" in p["reason"]


def test_within_group_shown_in_evidence_but_not_counted(tmp_path):
    e = ext_item("p1", MAKIZAKO_ARMS, [{"measure": "mean difference", "value": None,
                                         "p_value": "0.002", "comparison": "unknown"}])
    e["results"][0]["quote"] = MAKIZAKO_QUOTE
    llm = ScriptedLLM({"synthesis": lambda p: {"sentences": [], "heterogeneity": []}})
    s = make_synthesis(llm, out_dir=tmp_path)(syn_state([e]))["synthesis"]
    assert ("reported comparison not stated mean difference, p 0.002 [not a between-group "
            "result: not used for direction]") in s["evidence"]


def test_arm_pair_skips_a_baseline_timepoint():
    arms = [arm("exercise", 20, 9.0, 2, tp="baseline"), arm("control", 20, 8.0, 2,
                                                             tp="baseline"),
            arm("exercise", 20, 5.0, 2), arm("control", 20, 7.0, 2)]
    i, c, tp = arm_pair(arms)
    assert tp == "post-intervention" and (i["mean"], c["mean"]) == (5.0, 7.0)
    assert arm_pair(arms[:2]) is None


# --- Fix 2: one metric per study, fixed order, conflict flag ---------------------------------

def test_makizako_post_means_used_and_conflict_with_derived_change():
    g, p = one(MAKIZAKO_ARMS, quote=MAKIZAKO_QUOTE)
    assert p["metric_used"] == "post_means" and p["direction"] == "favours control"
    # Change derived by code from verified baseline and post means: -1.8 vs -1.3.
    ch = p["metrics"]["change_means"]
    assert (ch["intervention"], ch["control"]) == (pytest.approx(-1.8), pytest.approx(-1.3))
    assert ch["direction"] == "favours intervention" and "derived" in ch["detail"]
    assert p["metric_conflict"] and g["conflicts"] == ["S1"]
    assert "change from baseline favours intervention" in p["conflict"]
    # Another order (pre-specified in the protocol) -> change used, conflict still flagged.
    _, p = one(MAKIZAKO_ARMS, order=["change_means", "post_means"])
    assert p["metric_used"] == "change_means" and p["direction"] == "favours intervention"
    assert p["metric_conflict"]


def test_printed_change_and_estimate_first_in_order():
    # Ali 2026: post 15.06 vs 12.19 favours control; printed change -3.56 vs -0.88 favours
    # exercise -> conflict.
    ali = [full_arm("exercise", 16, 15.06, 1.83, baseline=18.63, change=-3.56),
           full_arm("control", 16, 12.19, 1.47, baseline=13.06, change=-0.88)]
    _, p = one(ali)
    assert p["direction"] == "favours control" and p["metric_conflict"]
    assert "printed" in p["metrics"]["change_means"]["detail"]
    assert p["g"] is not None                      # pooling still uses post means
    # A between-group estimate comes first in the default order.
    est = {"measure": "mean difference", "value": 2.1, "comparison": "between_group",
           "contrast": "control minus intervention", "p_value": "0.01"}
    _, p = one(ali, [est], quote="exercise ... control")
    assert p["metric_used"] == "between_group_estimate"
    assert p["direction"] == "favours intervention"
    assert p["metric_conflict"]                    # post means point the other way


def test_no_difference_is_not_a_conflict():
    arms = [full_arm("exercise", 10, 5.0, 1, baseline=7.0),
            full_arm("control", 10, 5.0, 1, baseline=6.0)]
    _, p = one(arms)
    assert p["direction"] == "no difference" and not p["metric_conflict"]


def test_metric_order_in_protocol_and_template():
    base = {"title": "t", "question": "q", "databases": ["openalex"], "rationale": "r",
            "pico": {"population": "a", "intervention": "b", "comparator": "c",
                     "outcomes": ["o"]},
            "inclusion_criteria": [], "exclusion_criteria": [], "study_designs": []}
    assert Protocol.model_validate(base).direction_metric_order == DEFAULT_METRIC_ORDER
    assert metric_order({}) == DEFAULT_METRIC_ORDER          # protocols written before 8b
    with pytest.raises(ValidationError):
        Protocol.model_validate(base | {"direction_metric_order": ["post_means",
                                                                   "post_means"]})
    with pytest.raises(ValidationError):
        Protocol.model_validate(base | {"direction_metric_order": ["p_value"]})
    t = Template.model_validate({"direction_metric_order": ["change_means", "post_means"]})
    assert t.direction_metric_order == ["change_means", "post_means"]


def test_extraction_checks_baseline_and_change_numbers_and_defaults_comparison():
    chunks = [{"chunk_id": "m::1", "text": "[Table 3] " + MAKIZAKO_QUOTE}]
    raw = [{"outcome": "O1", "quote": MAKIZAKO_QUOTE,
            "arms": [full_arm("exercise", None, 5.3, 2.5, baseline=7.1, change=-1.8),
                     full_arm("control", None, 5.1, 3.1, baseline=6.4)],
            "estimates": [{"measure": "p", "value": None, "p_value": "0.002"}]}]
    (res,), flags = check_results(raw, ["O1"], chunks)
    ex, ct = res["arms"]
    assert ex["baseline_mean"] == 7.1 and ct["baseline_mean"] == 6.4
    assert ex["change_mean"] is None              # -1.8 is not printed: never computed
    assert "exercise.change_mean=-1.8" in res["numbers_dropped"]
    assert res["estimates"][0]["comparison"] == "unknown"


# --- Fix 3 + critic checks ------------------------------------------------------------------

def critic_on(tmp_path, exts, narrative, critic_issues=()):
    st = syn_state(exts)
    llm = ScriptedLLM({"synthesis": lambda p: narrative})
    syn = make_synthesis(llm, out_dir=tmp_path)(st)["synthesis"]
    st |= {"synthesis": syn, "completed_stages": st["completed_stages"] + ["synthesis"]}
    qa = make_critic(ScriptedLLM({"critic": [{"issues": list(critic_issues)}]}))(st)
    return syn, qa["qa_report"]


def conflict_exts():
    makizako = ext_item("p1", MAKIZAKO_ARMS)
    makizako["results"][0]["quote"] = MAKIZAKO_QUOTE
    agree = [ext_item(f"p{i}", [arm("exercise", 20, 4.0, 2), arm("control", 20, 6.0, 2)])
             for i in (2, 3)]
    return [makizako, *agree]


ALL = ["S1", "S2", "S3"]


def test_heterogeneity_missing_is_a_blocker(tmp_path):
    narrative = {"sentences": [
        {"section": "O1", "text": "Two trials favoured exercise and one favoured control; "
         "in that trial the change from baseline points the opposite way.",
         "studies": ALL}]}
    syn, qa = critic_on(tmp_path, conflict_exts(), narrative)
    kinds = {i["kind"]: i["severity"] for i in qa["issues"]}
    assert kinds["heterogeneity_missing"] == "blocker" and qa["loop_back"]
    assert "conflict_not_mentioned" not in kinds
    with_het = narrative | {"heterogeneity": [{"outcome": "O1", "text": "Directions "
                                               "differed; doses and instruments varied.",
                                               "studies": ALL}]}
    syn, qa = critic_on(tmp_path, conflict_exts(), with_het)
    assert qa["passed"] and qa["blockers"] == 0
    assert "*Heterogeneity.* Directions differed" in syn["text"]
    assert syn["heterogeneity"][0]["section"] == "O1"


def test_conflict_must_be_named_and_majority_warning(tmp_path):
    het = [{"outcome": "O1", "text": "Doses varied.", "studies": ALL}]
    narrative = {"sentences": [{"section": "O1", "text": "Two trials favoured exercise.",
                                "studies": ALL}], "heterogeneity": het}
    syn, qa = critic_on(tmp_path, conflict_exts(), narrative)
    (g,) = syn["outcomes"]
    assert g["conflicts"] == ["S1"] and g["majority"] == "favours intervention"
    assert g["against_majority"] == ["S1"]
    assert "METRIC CONFLICT: post-intervention means favours control" in syn["evidence"]
    assert "Metric conflicts (direction depends on the metric): S1." in syn["evidence"]
    assert "Against the majority direction: S1." in syn["evidence"]
    kinds = {i["kind"]: i["severity"] for i in qa["issues"]}
    assert kinds["conflict_not_mentioned"] == "blocker"
    assert kinds["direction_against_majority"] == "warning"
    # Named in a sentence citing S1 -> no blocker; the majority warning stays (never loops).
    named = narrative | {"sentences": narrative["sentences"] + [
        {"section": "O1", "text": "In one trial the direction depends on the metric: "
         "post-intervention means favour control, change from baseline favours exercise.",
         "studies": ["S1"]}]}
    _, qa = critic_on(tmp_path, conflict_exts(), named)
    assert qa["passed"] and qa["warnings"] == 1 and not qa["loop_back"]


def test_tie_has_no_majority():
    exts = [ext_item("p1", [arm("exercise", 20, 4.0, 2), arm("control", 20, 6.0, 2)]),
            ext_item("p2", [arm("exercise", 20, 7.0, 2), arm("control", 20, 6.0, 2)])]
    studies = study_list(syn_state(exts))
    g = synthesise_outcome("O1", "depressive symptoms", LOWER, studies)
    assert g["majority"] is None and g["against_majority"] == []
    assert PROTOCOL["outcome_directions"][0]["better"] == LOWER


def test_pooled_heterogeneity_needs_i2_and_required_items_in_prompt(tmp_path):
    exts = [ext_item(f"p{i}", [arm("exercise", 20, m, 2), arm("control", 20, 6.0, 2)])
            for i, m in ((1, 4.0), (2, 5.0), (3, 5.5))]
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return {"sentences": [{"section": "O1", "text": "Results were pooled.",
                               "studies": ALL}],
                "heterogeneity": [{"outcome": "O1", "text": "Doses varied.",
                                   "studies": ALL}]}
    st = syn_state(exts)
    syn = make_synthesis(ScriptedLLM({"synthesis": answer}), out_dir=tmp_path)(st)["synthesis"]
    assert syn["outcomes"][0]["meta"] and "Required in this draft" in prompts[0]
    assert "'heterogeneity' entry for O1 giving I2" in prompts[0]
    st |= {"synthesis": syn, "completed_stages": st["completed_stages"] + ["synthesis"]}
    qa = make_critic(ScriptedLLM({"critic": [{"issues": []}]}))(st)["qa_report"]
    assert [i["kind"] for i in qa["issues"]] == ["heterogeneity_without_i2"]
    syn["heterogeneity"][0]["text"] = "I² was 56% and tau² 0.1."
    assert make_critic(ScriptedLLM({"critic": [{"issues": []}]}))(st)["qa_report"]["passed"]


def test_conflicting_study_is_named_in_required_items(tmp_path):
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return {"sentences": [], "heterogeneity": []}
    make_synthesis(ScriptedLLM({"synthesis": answer}), out_dir=tmp_path)(
        syn_state(conflict_exts()))
    assert ("a sentence in section O1 with studies ['S1'] saying its direction depends on "
            "the metric: post-intervention means favours control") in prompts[0]
