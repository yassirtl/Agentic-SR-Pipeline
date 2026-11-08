"""Part 10c: search recall fixes from the T1/T3 v1 runs (runs/20261108-173514-4dabd6,
runs/20261108-164532-b6cfd7).

(1) broadening when all databases together return too few hits; (2) short keywords,
code flags for long and mixed keywords; (3) a setting block that repeats the
intervention is merged into it, an outcome block of measurement terms is dropped.
The blocks below are the ones gpt-4o-mini wrote in the v1 runs.
"""

import pytest

from sr_pipeline.agents.search_strategy import (build_prompt, build_refine_prompt,
                                                choose_attempt, is_metric_block,
                                                keyword_flags, make_search_strategy,
                                                merge_setting)
from sr_pipeline.config import load_config
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.sources import Concept

from test_search_refine import FakeCounts, http_for

# --- v1 blocks ------------------------------------------------------------------------

T1_POP = {"name": "population", "role": "population",
          "keywords": ["cardiac arrhythmias", "arrhythmia", "atrial fibrillation"],
          "mesh_terms": ["Arrhythmias, Cardiac"]}
T1_INT = {"name": "intervention", "role": "intervention",
          "keywords": ["smartwatches", "wearable technology", "wrist-worn devices",
                       "fitness trackers", "smart watch", "health monitoring devices"],
          "mesh_terms": ["Wearable Electronic Devices"]}
T1_SET = {"name": "setting", "role": "setting",
          "keywords": ["smartwatch", "wearable", "device", "monitoring device",
                       "health device"]}
T1_OUT = {"name": "outcome", "role": "outcome",
          "keywords": ["sensitivity", "specificity", "diagnostic accuracy", "detection rate",
                       "true positive", "false positive", "true negative", "false negative"]}
T2_OUT = {"name": "outcome", "role": "outcome",
          "keywords": ["abnormal ECG", "normal ECG", "classification", "binary output",
                       "accuracy", "efficiency"]}
T3_POP = {"name": "population", "role": "population",
          "keywords": ["machine learning", "ML models", "classifiers", "algorithms",
                       "ECG classifiers"]}
T3_INT = {"name": "intervention", "role": "intervention",
          "keywords": ["classify ECG", "ECG classification", "heartbeat classification",
                       "ECG analysis", "heart signal classification"]}

T1_PROTOCOL = {
    "question": "How accurate are smartwatches at detecting cardiac arrhythmias?",
    "title": "t", "rationale": "r",
    "pico": {"population": "individuals with cardiac arrhythmias",
             "intervention": "smartwatches", "comparator": "none",
             "outcomes": ["sensitivity", "specificity"], "setting": "smartwatch"},
    "inclusion_criteria": [], "exclusion_criteria": [], "study_designs": [],
    "databases": ["openalex", "pubmed"], "years": {"start": 2001, "end": None},
    "languages": ["English"],
}


def plan(*blocks, why="r"):
    return {"concepts": list(blocks), "rationale": why}


def refined(*blocks, why="r"):
    return {**plan(*blocks, why=why), "removed_terms": [], "changed_terms": []}


def run(answers, hits, protocol=T1_PROTOCOL, **kw):
    llm = ScriptedLLM({"search_strategy": answers})
    kw.setdefault("max_records", 1000)
    node = make_search_strategy(llm, http_for(FakeCounts(hits)), min_hits=20, env={}, **kw)
    out = node({"protocol": protocol, "completed_stages": ["planner"]})
    return out["search_strategy"], out["audit_log"][0], llm


# --- (3) setting merged into intervention, outcome of measurement terms dropped ---------

def test_t1_setting_block_repeating_the_intervention_is_merged():
    blocks, merged = merge_setting([Concept(**c) for c in (T1_POP, T1_INT, T1_SET)])
    assert [c.role for c in blocks] == ["population", "intervention"]
    # every v1 setting keyword is already said by an intervention keyword
    assert merged == {"covered": T1_SET["keywords"], "added": []}
    assert blocks[1].keywords == T1_INT["keywords"]


def test_partly_new_setting_keywords_are_added_to_the_intervention():
    setting = Concept(**{**T1_SET, "keywords": ["smartwatch", "wearable", "Apple Watch"]})
    blocks, merged = merge_setting([Concept(**T1_INT), setting])
    assert merged is not None and merged["added"] == ["Apple Watch"] and len(blocks) == 1
    assert blocks[0].keywords[-1] == "Apple Watch"


def test_distinct_setting_block_is_kept():
    setting = Concept(**{**T1_SET, "keywords": ["hospital", "outpatient", "wearable"]})
    blocks, merged = merge_setting([Concept(**T1_INT), setting])
    assert merged is None and [c.role for c in blocks] == ["intervention", "setting"]


@pytest.mark.parametrize("block, metric", [
    (T1_OUT, True),                                     # T1 v1: 8 of 8 measurement terms
    (T2_OUT, False),                                    # T2: 2 of 6, the task defines it
    ({"keywords": ["depression", "depressive symptoms"]}, False),
    ({"keywords": ["F1-score", "AUC", "latency"]}, True),
])
def test_metric_outcome_block(block, metric):
    assert is_metric_block(block["keywords"]) is metric


def test_t1_first_plan_without_setting_and_outcome_blocks():
    def hits(q):
        return 900 if "sensitivity" not in q.lower() else 40
    s, entry, llm = run([plan(T1_POP, T1_INT, T1_SET, T1_OUT)], hits, broaden_below=300)
    a = s["attempts"][0]
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention"]
    assert a["outcome_dropped"] == T1_OUT["keywords"]
    assert a["setting_merged"]["covered"] == T1_SET["keywords"]
    assert not a["setting_block_missing"]                # searched in the intervention block
    assert "sensitivity" not in s["queries"]["openalex"]
    assert entry["outcome_dropped"] and entry["setting_merged"]
    assert len(llm.calls) == 1 and s["counts"] == {"openalex": 900, "pubmed": 900}


def test_fixed_blocks_are_never_merged_or_dropped():
    protocol = {**T1_PROTOCOL, "search_terms": {"setting": ["smartwatch", "wearable"],
                                                "outcome": ["sensitivity", "specificity"]}}
    s, _, _ = run([plan(T1_POP, T1_INT)], lambda q: 900, protocol=protocol)
    roles = [c["role"] for c in s["concepts"]]
    assert roles == ["population", "intervention", "setting", "outcome"]
    assert not s["attempts"][0]["outcome_dropped"] and not s["attempts"][0]["setting_merged"]


# --- (2) short keywords -----------------------------------------------------------------

def test_prompt_asks_for_words_not_phrases_and_no_measurement_outcome():
    prompt = build_prompt(T1_PROTOCOL)
    assert "single words or short established phrases" in prompt
    assert "never a phrase that restates the question or joins two concepts" in prompt
    assert "product or brand names" in prompt
    assert "measurement terms (accuracy, sensitivity" in prompt
    assert "do not add a setting block that repeats the intervention" in prompt


def test_t3_v1_phrases_joining_two_blocks_are_flagged():
    flags = keyword_flags([T3_POP, T3_INT])
    assert flags["mixed_keywords"] == ["classify ECG", "ECG analysis", "ECG classification",
                                       "ECG classifiers"]
    assert flags["long_keywords"] == []
    four = {**T3_INT, "keywords": ["ECG beat type classification"]}
    assert keyword_flags([four])["long_keywords"] == ["ECG beat type classification"]
    # a fixed block (topic terms) is never flagged: the LLM cannot change it
    fixed = {"role": "setting", "keywords": ["edge AI"], "fixed": True}
    llm_int = {"role": "intervention", "keywords": ["AI", "machine learning"]}
    assert keyword_flags([llm_int, fixed])["mixed_keywords"] == []


def test_flags_are_shown_to_the_llm_at_refinement():
    s, _, llm = run([plan(T3_POP, T3_INT), refined(T3_POP, T3_INT, why="2")],
                    lambda q: 5)                          # too narrow everywhere
    assert s["attempts"][0]["mixed_keywords"]
    assert "keywords sharing a word with another block: classify ECG" in llm.calls[1][1]


# --- (1) broadening ---------------------------------------------------------------------

NARROW_INT = {**T1_INT, "keywords": ["smartwatches", "wrist-worn devices"]}
BROAD_INT = {**T1_INT, "keywords": ["smartwatch", "smart watch", "Apple Watch", "wearable"]}


def by_int(q):
    """Fake database: the narrow intervention gives 60 hits, the broad one 400."""
    return 400 if "apple watch" in q.lower() else 60


def test_too_few_hits_in_total_ask_for_broadening():
    s, entry, llm = run([plan(T1_POP, NARROW_INT), refined(T1_POP, BROAD_INT, why="broader")],
                        by_int, broaden_below=300)
    first, second = s["attempts"]
    assert first["status"] == {"openalex": "ok", "pubmed": "ok"}   # in range per database
    assert first["short"] and first["total"] == 120
    assert "120 hits in all databases, below 300" in first["reason"]
    prompt = llm.calls[1][1]
    assert "all databases together: 120 hits -> too few (at least 300 wanted)" in prompt
    assert "broaden. These are the ONLY allowed moves" in prompt
    assert "remove the outcome block (unless it is fixed" in prompt
    assert second["total"] == 800 and not second["short"]
    assert second["reason"] == "all databases within range"
    assert s["chosen_attempt"] == 2 and "Apple Watch" in s["queries"]["openalex"]
    assert entry["broadening_requested"] and not entry["broaden_failed"]
    assert s["targets"]["broaden_below"] == 300


def test_broadening_that_does_not_raise_the_total_stops_and_is_not_chosen():
    same_size = {**NARROW_INT, "keywords": ["smartwatch", "wrist-worn device"]}
    s, entry, llm = run([plan(T1_POP, NARROW_INT), refined(T1_POP, same_size),
                         refined(T1_POP, BROAD_INT)], by_int, broaden_below=300)
    assert len(llm.calls) == 2 and len(s["attempts"]) == 2
    second = s["attempts"][1]
    assert second["broaden_failed"] and "did not raise the total" in second["reason"]
    assert s["chosen_attempt"] == 1 and entry["broaden_failed"]


def test_broadening_that_overshoots_keeps_the_earlier_plan_and_narrows_again():
    def hits(q):
        q = q.lower()
        return 5000 if "wearable" in q and "pacemaker" not in q else \
            500 if "pacemaker" in q else 60
    huge = {**BROAD_INT, "keywords": ["smartwatch", "wearable"]}
    narrower = {**BROAD_INT, "keywords": ["smartwatch", "wearable", "pacemaker"]}
    s, _, llm = run([plan(T1_POP, NARROW_INT), refined(T1_POP, huge),
                     refined(T1_POP, narrower)], hits, broaden_below=300)
    first, second, third = s["attempts"]
    assert second["status"] == {"openalex": "too broad", "pubmed": "too broad"}
    assert not second["short"]                           # too broad: never "too few"
    assert "broaden. These are" not in llm.calls[2][1]   # third prompt narrows
    assert third["total"] == 1000 and s["chosen_attempt"] == 3
    assert choose_attempt([first, second])["attempt"] == 1


def test_no_broadening_above_the_threshold_or_when_off():
    s, entry, llm = run([plan(T1_POP, NARROW_INT)], lambda q: 200, broaden_below=300)
    assert len(llm.calls) == 1 and not s["attempts"][0]["short"]     # 400 >= 300
    s, entry, llm = run([plan(T1_POP, NARROW_INT)], by_int)          # broaden_below=0
    assert len(llm.calls) == 1 and not entry["broadening_requested"]


def test_refine_prompt_without_broadening_has_no_broaden_branch():
    a = {"attempt": 1, "concepts": [T1_POP], "status": {"pubmed": "too broad"},
         "counts": {"pubmed": 9000}}
    prompt = build_refine_prompt(T1_PROTOCOL, [a], max_records=3000, max_hits=3000,
                                 min_hits=20)
    assert "broaden. These are" not in prompt


def test_broaden_threshold_from_config():
    assert load_config(env={}).search_broaden_below() == 300            # 3 x 100
    assert load_config(env={"SR_SCREEN_LIMIT": "50"}).search_broaden_below() == 150
    assert load_config(env={"SR_SCREEN_LIMIT": "0"}).search_broaden_below() == 300
    assert load_config(env={"SR_BROADEN_BELOW": "0"}).search_broaden_below() == 0
    assert load_config(env={"SR_BROADEN_BELOW": "1000"}).search_broaden_below() == 1000


# --- generic task words (from the T3 smoke run of these fixes, 20261108-180553-f320b0) --

T3_SMOKE_POP = {"name": "population", "role": "population",
                "keywords": ["machine learning", "ML models", "classifiers", "algorithms",
                             "neural networks", "deep learning"]}
T3_SMOKE_INT = {"name": "intervention", "role": "intervention",
                "keywords": ["classify", "classification", "ECG", "electrocardiogram",
                             "heartbeat", "arrhythmia detection"]}


def test_generic_task_words_are_removed_so_ecg_stays_required():
    t3 = {**T1_PROTOCOL, "search_terms": {"setting": ["embedded", "FPGA"], "outcome": []}}
    s, entry, _ = run([plan(T3_SMOKE_POP, T3_SMOKE_INT)], lambda q: 900, protocol=t3)
    a = s["attempts"][0]
    assert a["generic_removed"] == ["algorithms", "classify", "classification"]
    blocks = {c["role"]: c["keywords"] for c in s["concepts"]}
    assert blocks["intervention"] == ["ECG", "electrocardiogram", "heartbeat",
                                      "arrhythmia detection"]
    assert "algorithms" not in blocks["population"]
    assert "classification" not in s["queries"]["openalex"]
    assert entry["generic_removed"] == a["generic_removed"]


def test_block_of_generic_words_only_is_dropped_and_prompt_explains_or():
    only = {"name": "x", "role": "outcome", "keywords": ["detection", "classification"]}
    s, _, _ = run([plan(T1_POP, T1_INT, only)], lambda q: 900)
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention"]
    prompt = build_prompt(T1_PROTOCOL)
    assert "(ECG OR classification) matches every classification paper" in prompt


def test_refine_prompt_shows_what_code_undid():
    t3 = {**T1_PROTOCOL, "search_terms": {"outcome": []}}
    out_block = {"name": "o", "role": "outcome", "keywords": ["accuracy", "latency"]}
    _, _, llm = run([plan(T3_SMOKE_POP, T3_SMOKE_INT, out_block),
                     refined(T3_SMOKE_POP, T3_SMOKE_INT)], lambda q: 5, protocol=t3)
    prompt = llm.calls[1][1]
    assert "(generic words removed by code: algorithms, classify, classification)" in prompt
    assert "your outcome block was replaced or removed by code" in prompt
    assert "removing or changing their terms has no effect" in prompt
