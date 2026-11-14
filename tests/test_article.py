"""Part 11f: the article's findings (study table, evidence map, summary of findings) and the
writer's code checks. Synthetic state, no network."""

from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.reporting import findings as F
from sr_pipeline.reporting import writer as W

SPEC = {
    "rows": {"field": "model", "label": "Model family",
             "groups": {"Spiking": ["spiking", "SNN"], "CNN": ["CNN", "convolutional"]}},
    "cols": {"field": "platform", "label": "Device",
             "groups": {"Microcontroller": ["microcontroller", "MCU"], "FPGA": ["FPGA"]}},
    "datasets": {"MIT-BIH": ["MIT-BIH"]},
    "table": {"columns": ["model", "platform", "dataset", "latency"],
              "numeric": ["latency"]},
}


def _f(value):
    return {"status": "reported", "value": value} if value else {"status": "not_reported"}


def state():
    studies = [  # label, title, model, platform, dataset, latency, metrics
        ("S1", "A CNN on a microcontroller", "1D-CNN", "STM32 microcontroller", "MIT-BIH",
         "2.1 ms", "Accuracy: 98.2%"),
        ("S2", "Spiking network for ECG", None, None, None, "milliseconds", None),
        ("S3", "Arrhythmia detection on FPGA", "LightGBM", "Xilinx FPGA", "own data", None,
         "an accuracy of 91 %"),
        ("S4", "ECG classification", None, None, None, None, None),
    ]
    syn = [{"label": lab, "paper_id": f"p{lab}", "author": "A", "year": 2024, "title": t,
            "abstract_only": lab in ("S2", "S4"), "rob_overall": "unclear"}
           for lab, t, *_ in studies]
    ext = [{"paper_id": f"p{lab}", "round": 1, "fields": {
        "model": _f(m), "platform": _f(pl), "dataset": _f(d), "latency": _f(lat),
        "classification_metrics": _f(acc)}} for lab, _, m, pl, d, lat, acc in studies]
    protocol = {"question": "Which models?", "evidence_map": SPEC,
                "extraction_fields": {"model": "m", "platform": "p", "dataset": "d",
                                      "latency": "l", "classification_metrics": "c"}}
    included = [{"paper_id": f"p{lab}", "round": 1, "stage": "ft", "final": "include",
                 "title": t} for lab, t, *_ in studies]
    return {"protocol": protocol, "synthesis": {"studies": syn}, "extractions": ext,
            "ft_decisions": included, "included": [f"p{lab}" for lab, *_ in studies],
            "unique_records": [
                {"paper_id": f"p{lab}", "title": t} for lab, t, *_ in studies]}


def test_classify_whole_word_plural_first_group():
    groups = {"CNN": ["CNN"], "Edge": ["edge"]}
    assert F.classify("two CNNs at the edge", groups) == "CNN"
    assert F.classify("knowledge graph", groups) is None          # 'edge' inside a word
    assert F.classify("Edge devices", groups) == "Edge"


def test_evidence_map_counts_every_study_once_with_title_fallback():
    st = state()
    em = F.evidence_map(st, SPEC)
    assert em["n"] == 4
    cells = em["cells"]
    assert cells["CNN|Microcontroller"] == ["S1"]
    assert cells["Spiking|Not reported"] == ["S2"]               # model from the title
    assert cells["Other|FPGA"] == ["S3"]                          # LightGBM: no keyword
    assert cells["Not reported|Not reported"] == ["S4"]
    assert em["sources"]["rows"] == {"value": 2, "title": 1, "none": 1}
    rows, cols = F.evidence_map_rows(em)
    assert rows[-1]["t"] == "**4**"


def test_study_table_numeric_needs_a_digit():
    rows, cols = F.study_table(state(), SPEC)
    assert [c for c, _ in cols] == ["who", "model", "platform", "dataset", "latency", "used"]
    assert rows[0]["latency"] == "2.1 ms"
    assert rows[1]["latency"] == "–"                              # "milliseconds"
    assert rows[1]["used"] == "abstract only †"


def test_summary_lists_accuracy_range_without_ranking():
    st = state()
    sm = F.summary(st, SPEC, F.evidence_map(st, SPEC))
    assert sm["accuracy_studies"] == ["S1", "S3"]
    assert (sm["accuracy_min"], sm["accuracy_max"]) == (91.0, 98.2)
    assert sm["reported"]["latency"] == 1
    lines = F.summary_lines(sm, st["protocol"])
    assert any("range from 91% to 98.2%" in x and "not comparable" in x for x in lines)
    assert not any("best" in x.lower() for x in lines)


def test_no_evidence_map_without_spec_or_fields():
    st = state()
    st["protocol"].pop("evidence_map")
    assert F.evidence_map(st, None) is None
    st["protocol"].pop("extraction_fields")
    assert F.summary(st, None, None) is None


FACTS = "Included studies: 4. Accuracy range 91% to 98.2%."


def test_writer_check_flags_numbers_labels_and_outside_references():
    labels = ["S1", "S2", "S3", "S4"]
    assert W.check("Four studies; accuracy up to 98.2% (S1).", FACTS, labels, 4) == []
    assert W.check("Accuracy reached 99.5%.", FACTS, labels, 4)[0].startswith("number")
    assert W.check("As in S9.", FACTS, labels, 4)[0].startswith("unknown study")
    assert W.check("Smith et al. showed this.", FACTS, labels, 4)[0].startswith("outside")
    assert W.check("See [3].", FACTS, labels, 4)


def _answer(**over):
    base = {"abstract_background": "Small devices matter.", "abstract_results": "Four "
            "studies were included.", "abstract_conclusion": "Evidence is limited.",
            "introduction": ["Context.", "Aim."], "discussion": ["A.", "B.", "C."],
            "conclusion": "More work is needed."}
    return {**base, **over}


def test_writer_keeps_passing_sections_and_replaces_failing_ones():
    llm = ScriptedLLM({"writer": [_answer(conclusion="Accuracy is 99.9% everywhere.")]})
    out = W.write(llm, {"question": "Q?"}, FACTS, {"included": 4}, ["S1"])
    assert out["source"]["introduction"] == "ai" and out["sections"]["discussion"] == ["A.", "B.", "C."]
    assert out["source"]["conclusion"] == "code" and "number" in out["flags"]["conclusion"][0]
    assert out["usage"] is not None


def test_writer_without_llm_or_on_error_uses_the_code_text():
    out = W.write(None, {"question": "Q?", "rationale": "The PICO framework helps."}, FACTS,
                  {"included": 4}, [])
    assert set(out["source"].values()) == {"code"}
    assert "PICO" not in " ".join(map(str, out["sections"].values()))   # rationale not printed
    bad = W.write(ScriptedLLM({"writer": [RuntimeError("down")]}), {"question": "Q?"},
                  FACTS, {}, [])
    assert bad["error"].startswith("RuntimeError") and set(bad["source"].values()) == {"code"}
