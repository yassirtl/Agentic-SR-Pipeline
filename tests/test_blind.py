"""Part 11g: blind labels page. Labels are saved at once, "not sure" is left out of the
score, the page walks the sheet one record at a time and shows no pipeline decision."""

import csv
from pathlib import Path

import pytest

from sr_pipeline.evaluation import labels as LB

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def sheet(tmp_path: Path, n: int = 3) -> Path:
    folder = tmp_path / "labels" / "20261108-000000-aaaaaa"
    folder.mkdir(parents=True)
    path = folder / "labels.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LB.SHEET_COLS)
        w.writeheader()
        for i in range(1, n + 1):
            w.writerow({"n": i, "paper_id": f"p{i}", "title": f"Paper {i}",
                        "abstract": f"Abstract {i}", "human": "", "note": ""})
    (folder / "criteria.md").write_text("# Criteria\n\n- **I1**: something\n")
    return path


def test_set_label_saves_and_unsure_is_not_scored(tmp_path):
    path = sheet(tmp_path)
    LB.set_label(path, "p1", "i", "fits I1")
    LB.set_label(path, "p2", "unsure")
    rows = LB.sheet_rows(path)
    assert (rows[0]["human"], rows[0]["note"], rows[1]["human"]) == ("include", "fits I1",
                                                                     "unsure")
    labels, problems = LB.read_labels(path)
    assert labels == {"p1": "include"} and problems == []
    with pytest.raises(ValueError):
        LB.set_label(path, "p3", "maybe")


def page(tmp_path, runs):
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              "from sr_pipeline.ui import blind as B\n"
              f"P.RUNS_DIR = Path({str(runs)!r})\n"
              f"B.LABELS_ROOT = Path({str(tmp_path / 'labels')!r})\nB.page_labels()\n")
    return AppTest.from_string(script, default_timeout=30).run()


def test_page_walks_the_sheet_and_shows_no_ai_decision(tmp_path):
    path = sheet(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    at = page(tmp_path, runs)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Paper 1 of 3" in text and "Abstract 1" in text
    assert "pipeline" not in text.lower() and "AI includes" not in text
    at = at.button(key="lb_in:0").click().run()
    assert "Paper 2 of 3" in " ".join(m.value for m in at.markdown)
    at = at.button(key="lb_ex:1").click().run()
    at = at.button(key="lb_un:2").click().run()
    assert not at.exception
    assert [r["human"] for r in LB.sheet_rows(path)] == ["include", "exclude", "unsure"]
    assert any("All 3 papers labelled" in s.value for s in at.success)
    # No run on this computer: the score is not computed here, and the page says so.
    assert any("cannot be computed here" in i.value for i in at.info)


def test_score_after_the_last_label_on_a_fake_run(tmp_path):
    import json
    from sr_pipeline import cli
    from test_report import INJECT, QUESTION, e2e_config, e2e_llm
    cfg = e2e_config(tmp_path)
    rid, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True, ask=lambda p: "",
                         out=lambda *a: None, llm=e2e_llm(), **INJECT)
    folder = tmp_path / "labels" / rid
    LB.make_sheet(state, folder, n=3)
    for r in LB.sheet_rows(folder / "labels.csv"):
        LB.set_label(folder / "labels.csv", r["paper_id"], "include")
    at = page(tmp_path, cfg.runs_dir)
    assert not at.exception
    at = at.button(key="lb_score").click().run()
    assert not at.exception
    sc = LB.score_labels(json.loads((cfg.runs_dir / rid / "state.json").read_text()),
                         LB.read_labels(folder / "labels.csv")[0])
    assert sc["labelled"] == 3
    shown = " ".join(m.value for m in at.markdown)
    assert "Accuracy" in shown and f"{round(100 * sc['accuracy'])}%" in shown
