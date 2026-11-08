"""Part 10: revised evaluation topics (T2 without gold, T3 embedded ECG, old T3 = D0).

Topic extraction fields and the review's own risk-of-bias domains reach the protocol,
the extraction fields and the risk-of-bias tool; a topic without gold is evaluated on
labels only; a run made under an old topic id is not compared with the new topic's gold.
"""

import json
from pathlib import Path

import pytest

from sr_pipeline.agents.extraction import choose_tool, extraction_fields, overall, tool_for
from sr_pipeline.evaluation.__main__ import context
from sr_pipeline.evaluation.gold import load_gold
from sr_pipeline.evaluation.topics import load_topic, seed_protocol

ROOT = Path(__file__).resolve().parents[1]

BASE = {"title": "t", "pico": {"population": "ECG recordings", "intervention": "ML model",
                               "comparator": "none", "outcomes": ["classification accuracy"]},
        "inclusion_criteria": ["x"], "exclusion_criteria": [], "study_designs": [],
        "rationale": "r", "databases": ["openalex"]}


def test_revised_topics_scope():
    t2, t3, d0 = load_topic("T2"), load_topic("T3"), load_topic("D0")
    assert t2["gold"] is None and "gold_recall" not in t2["metrics"]
    assert t2["years"] == {"start": 2021, "end_date": None}
    assert str(t3["years"]["end_date"]) == "2026-09-21" and t3["years"]["start"] == 2021
    assert d0["question"] == "Does exercise reduce depressive symptoms in older adults?"
    assert d0["gold"] == "gold/D0.csv"
    with pytest.raises(KeyError):          # shared YAML anchors are not topics
        load_topic("_engineering_fields")


def test_t3_gold_is_the_30_embedded_implementations():
    rows = load_gold(ROOT / "evaluation" / "gold" / "T3.csv")
    assert [int(r["ref"]) for r in rows] == list(range(27, 57))   # Table 2 of the review
    assert sum(int(r["year"]) < 2021 for r in rows) == 3          # outside the window
    cov = load_gold(ROOT / "evaluation" / "gold" / "T3_coverage.csv")
    assert all(r["found_doi"] for r in cov)                       # every row identifiable


def test_topic_fields_and_domains_reach_extraction_and_rob():
    p, audit = seed_protocol(BASE, load_topic("T3"))
    assert "checklist_domains" in audit["seeded_fields"]
    extra = [f for f in extraction_fields(p) if f.group == "extra"]
    assert {"partition", "latency", "parameters", "platform"} <= {f.key for f in extra}
    key, how = choose_tool("diagnostic accuracy study", True, p)   # design does not decide
    assert (key, how) == ("checklist", "protocol instrument (review's own domains)")
    tool = tool_for(key, p)
    assert [d[0] for d in tool.domains] == ["D1", "D2", "D3", "D4", "D5"]
    assert "inter-patient" in tool.domains[1][2] and "Arredondo" in tool.name
    assert overall(tool, ["low", "high", "low", "unclear", "low"]) == "high"


def test_protocol_without_domains_keeps_the_generic_tools():
    p, _ = seed_protocol(BASE, load_topic("D0"))
    assert p["checklist_domains"] == [] and p["extraction_fields"] == {}
    assert choose_tool("randomized trial", True, p)[0] == "rob2"
    assert [d[0] for d in tool_for("checklist", p).domains][0] == "C1"


def _run(tmp_path, topic, question):
    folder = tmp_path / "run1"
    folder.mkdir()
    (folder / "run.json").write_text(json.dumps({"topic": topic}))
    (folder / "state.json").write_text(json.dumps({"protocol": {"question": question,
                                                                "topic": topic}}))
    return folder


def test_topic_without_gold_is_evaluated_on_labels_only(tmp_path):
    folder = _run(tmp_path, "T2", load_topic("T2")["question"])
    ctx = context(folder)
    assert ctx["topic"]["id"] == "T2" and ctx["gold_rows"] is None


def test_old_topic_id_is_not_compared_with_the_new_gold(tmp_path):
    folder = _run(tmp_path, "T3", load_topic("D0")["question"])   # a run made as old T3
    with pytest.raises(SystemExit, match="pass --topic"):
        context(folder)
    ctx = context(folder, "D0")
    assert ctx["topic"]["id"] == "D0" and len(ctx["gold_rows"]) == 15
