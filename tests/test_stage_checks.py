"""Part 11c: per-stage human check. Synthetic run folders only, no API, no network.
Every file goes to tmp_path (evaluation/stage_checks/ holds the user's own marks)."""

import json
from pathlib import Path

import pytest

from sr_pipeline.evaluation import stage_checks as SC
from sr_pipeline.evaluation.__main__ import context, render
from sr_pipeline.evaluation.metrics import run_metrics

RUN = "20261109-100000-eeeeee"


def make_state(n_records: int = 30) -> dict:
    papers = [{"paper_id": f"p{i:02d}", "title": f"Paper {i}", "abstract": f"Abstract <b>{i}</b>",
               "year": 2020, "doi": f"10.1/{i}", "rank": i, "sources": ["openalex"], "round": 1}
              for i in range(n_records)]
    ta = []
    for i, p in enumerate(papers[:25]):
        final = "include" if i % 3 == 0 else "exclude"
        ta.append({"paper_id": p["paper_id"], "round": 1, "stage": "ta", "title": p["title"],
                   "final": final, "resolved_by": "agreement",
                   "criterion": None if final == "include" else "I1",
                   "a": {"decision": final, "quote": "",
                         "verdicts": [{"criterion": "I1", "met": "no", "quote": f"quote {i}"}]}})
    # p00 re-screened in round 2 as an exclusion: only the latest round counts
    ta.append({**ta[0], "round": 2, "final": "exclude", "criterion": "D",
               "resolved_by": "adjudicator",
               "adjudicator": {"decision": "exclude", "quote": "a narrative review",
                               "chunk_id": ""}})
    ft = [{"paper_id": f"p{i:02d}", "round": 1, "stage": "ft", "title": f"Paper {i}",
           "final": "include", "resolved_by": "agreement", "criterion": None,
           "fulltext_status": "pmc_xml",
           "a": {"decision": "include", "quote": "we randomised", "chunk_id": f"p{i:02d}::3"}}
          for i in (3, 6, 9, 12)]
    extractions = [{"paper_id": f"p{i:02d}", "round": 1, "title": f"Paper {i}",
                    "fulltext_status": "pmc_xml",
                    "fields": {"country": {"status": "reported", "value": "Japan",
                                           "quote": "in Japan", "chunk_id": f"p{i:02d}::3"},
                               "age": {"status": "reported", "value": "70", "quote": "aged 70"},
                               "sex": {"status": "not reported"}}}
                   for i in (3, 6, 9, 12)]
    rob = [{"paper_id": f"p{i:02d}", "round": 1, "title": f"Paper {i}", "tool": "rob2",
            "domains": [{"domain": f"D{k}", "name": f"domain {k}", "judgment": "low",
                         "quote": "computer generated", "chunk_id": f"p{i:02d}::3",
                         "verified": k != 2, "reason": "sequence random"} for k in (1, 2, 3)]}
           for i in (3, 6, 9, 12)]
    synthesis = {"round": 1,
                 "evidence": "Studies (label | ...):\nS1 | Kim 2019 | RCT | n = 40\n"
                             "S2 | Lee 2020 | RCT | n = 50",
                 "narrative": [{"section": "O1", "text": "Exercise lowers scores.",
                                "studies": ["S1", "S2"]},
                               {"section": "overall", "text": "Evidence is weak.",
                                "studies": ["S2"]}],
                 "heterogeneity": [{"section": "O1", "text": "Scales differ.", "studies": ["S1"]}]}
    return {"protocol": {"question": "Q?", "inclusion_criteria": ["adults", "exercise"],
                         "exclusion_criteria": ["no control"]},
            "unique_records": papers, "ta_decisions": ta, "ft_decisions": ft,
            "extractions": extractions, "risk_of_bias": rob, "synthesis": synthesis}


# --- Items per stage --------------------------------------------------------------------

def test_stage_items_units_and_latest_round():
    s = make_state()
    chunks = {"p03::3": "Full passage of paper 3."}
    sizes = {st: len(SC.stage_items(s, st, chunks)) for st in SC.STAGES}
    assert sizes == {"search": 30, "ta": 25, "ft": 4, "extraction": 8, "rob": 12, "synthesis": 3}
    ta = {it["item_id"]: it for it in SC.stage_items(s, "ta")}
    assert ta["p00"]["decision"] == "exclude"                          # round 2 wins
    assert ta["p00"]["quote"] == "a narrative review"                  # adjudicator's quote
    assert "resolved by adjudicator" in ta["p00"]["detail"]
    assert ta["p01"]["quote"] == "quote 1"                             # deciding verdict's quote
    ft = SC.stage_items(s, "ft", chunks)[0]
    assert ft["context"] == "Full passage of paper 3." and ft["link"] == "https://doi.org/10.1/3"
    ex = SC.stage_items(s, "extraction", chunks)
    assert {it["item_id"] for it in ex} >= {"p03::country", "p03::age"}
    assert not any(it["item_id"].endswith("::sex") for it in ex)       # not reported: not sampled
    rob = {it["item_id"]: it for it in SC.stage_items(s, "rob")}
    assert "quote not verified" in rob["p03::D2"]["detail"]
    syn = SC.stage_items(s, "synthesis")
    assert syn[0]["decision"] == "Exercise lowers scores."
    assert "Kim 2019" in syn[0]["context"] and "Lee 2020" in syn[0]["context"]
    with pytest.raises(KeyError):
        SC.stage_items(s, "planner")


def test_stage_items_tolerate_missing_keys():
    for stage in SC.STAGES:
        assert SC.stage_items({}, stage) == []


# --- Sampling ---------------------------------------------------------------------------

def test_draw_is_seeded_and_prefix_stable():
    items = SC.stage_items(make_state(), "search")
    a = [r["item_id"] for r in SC.draw(items, 10, seed=7)]
    assert a == [r["item_id"] for r in SC.draw(list(reversed(items)), 10, seed=7)]  # order-free
    assert a == [r["item_id"] for r in SC.draw(items, 20, seed=7)][:10]  # larger n extends
    assert a != [r["item_id"] for r in SC.draw(items, 10, seed=8)]
    assert len(set(a)) == 10
    assert len(SC.draw(SC.stage_items(make_state(), "ft"), 10)) == 4      # all when fewer


def test_create_store_and_never_redraw(tmp_path):
    s = make_state()
    out = SC.create_check(s, RUN, "ta", 8, 3, root=tmp_path)
    csv_path, meta_path = SC.paths(RUN, "ta", tmp_path)
    assert csv_path.exists() and meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert (meta["seed"], meta["sampled"], meta["population"], meta["blind"]) == (3, 8, 25, False)
    loaded = SC.load_check(RUN, "ta", tmp_path)
    assert [r["item_id"] for r in loaded["rows"]] == [r["item_id"] for r in out["rows"]]
    assert loaded["rows"][0]["title"] and loaded["rows"][0]["decision"]   # self-contained copy
    with pytest.raises(FileExistsError):
        SC.create_check(s, RUN, "ta", 8, 4, root=tmp_path)
    for n in (4, 21):
        with pytest.raises(ValueError):
            SC.create_check(s, RUN, "search", n, root=tmp_path)
    with pytest.raises(ValueError):                                        # empty stage
        SC.create_check({}, RUN, "rob", 5, root=tmp_path)


def test_marks_save_and_discard_rules(tmp_path):
    out = SC.create_check(make_state(), RUN, "rob", 5, root=tmp_path)
    first = out["rows"][0]["item_id"]
    SC.discard_check(RUN, "rob", tmp_path)                                 # no mark: allowed
    assert SC.load_check(RUN, "rob", tmp_path) is None
    out = SC.create_check(make_state(), RUN, "rob", 5, root=tmp_path)
    SC.set_mark(RUN, "rob", first, "w", "D1 should be some concerns", root=tmp_path)
    SC.set_mark(RUN, "rob", first, None, None, root=tmp_path)             # None = unchanged
    row = SC.load_check(RUN, "rob", tmp_path)["rows"][0]
    assert (row["mark"], row["note"]) == ("wrong", "D1 should be some concerns")
    with pytest.raises(ValueError):
        SC.set_mark(RUN, "rob", first, "maybe", root=tmp_path)
    with pytest.raises(KeyError):
        SC.set_mark(RUN, "rob", "nope", "correct", root=tmp_path)
    with pytest.raises(PermissionError):
        SC.discard_check(RUN, "rob", tmp_path)


# --- Scoring ----------------------------------------------------------------------------

def test_wilson_matches_hand_values():
    assert SC.wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-3)
    lo, hi = SC.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-3) and hi == 1.0
    lo, hi = SC.wilson(0, 5)
    assert lo == 0.0 and hi == pytest.approx(0.4345, abs=1e-3)
    assert SC.wilson(0, 0) is None


def test_score_leaves_unsure_and_blank_out():
    rows = [{"item_id": str(i), "mark": m, "title": "t", "decision": "d", "note": ""}
            for i, m in enumerate(["correct"] * 8 + ["wrong"] * 2 + ["unsure", "", "?", "C"])]
    s = SC.score(rows)
    assert (s["sampled"], s["correct"], s["wrong"], s["unsure"], s["unmarked"]) == (14, 9, 2, 2, 1)
    assert s["judged"] == 11 and s["share_correct"] == pytest.approx(9 / 11, abs=1e-4)
    assert s["ci95"] == SC.wilson(9, 11)
    assert len(s["wrong_items"]) == 2
    assert SC.score([])["share_correct"] is None


# --- evaluate ---------------------------------------------------------------------------

def _run_folder(root: Path) -> Path:
    d = root / "runs" / RUN
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps(make_state()))
    (d / "run.json").write_text(json.dumps({"run_id": RUN, "topic": None}))
    return d


def test_evaluate_includes_stage_checks(tmp_path):
    folder = _run_folder(tmp_path)
    checks = tmp_path / "checks"
    ctx = context(folder, checks_root=checks)
    assert ctx["stage_checks"] is None
    assert "stage_checks" not in run_metrics(ctx)["accuracy"]

    out = SC.create_check(make_state(), RUN, "extraction", 5, root=checks)
    for i, r in enumerate(out["rows"]):
        SC.set_mark(RUN, "extraction", r["item_id"], "wrong" if i == 0 else "correct",
                    "value is 72" if i == 0 else "", root=checks)
    ctx = context(folder, checks_root=checks)
    # a topic with an explicit metric list (T2) names stage_checks too
    results = run_metrics(ctx, ["human_labels", "stage_checks"])
    v = results["accuracy"]["stage_checks"]["extraction"]
    assert (v["correct"], v["wrong"], v["population"], v["seed"]) == (4, 1, 8, 10)
    md = render(ctx, run_metrics(ctx))
    assert "Per-stage human check" in md and "No human labels yet" not in md
    assert "| Extraction |" in md and "80.0%" in md and "value is 72" in md


def test_topics_with_metric_lists_include_stage_checks():
    import yaml

    from sr_pipeline.evaluation.topics import TOPICS_FILE
    topics = yaml.safe_load(TOPICS_FILE.read_text())
    for tid, t in topics.items():
        if isinstance(t, dict) and t.get("metrics"):
            assert "stage_checks" in t["metrics"], tid


# --- Page (Streamlit test harness) ------------------------------------------------------

def _page(root: Path, checks: Path, stage: str | None = None):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              f"P.RUNS_DIR = Path({str(root / 'runs')!r})\n"
              f"P.CHECKS_DIR = Path({str(checks)!r})\nP.setup()\nP.page_check()\n")
    at = AppTest.from_string(script, default_timeout=60)
    at.session_state["run_id"] = RUN
    if stage:
        at.session_state["check_stage"] = stage
    return at.run()


def test_check_page_draw_and_mark(tmp_path):
    _run_folder(tmp_path)
    checks = tmp_path / "checks"
    at = _page(tmp_path, checks, "ta")
    assert not at.exception
    assert any("25 items in this run" in c.value for c in at.caption)
    at.button(key=f"draw:{RUN}:ta").click().run()
    assert not at.exception
    rows = SC.load_check(RUN, "ta", checks)["rows"]
    assert len(rows) == SC.DEFAULT_N
    html = "".join(m.value for m in at.markdown)
    assert "Abstract &lt;b&gt;" in html and "Abstract <b>" not in html  # run text escaped
    discard = f"discard:{RUN}:ta"
    assert discard in [b.key for b in at.button]                      # no mark yet
    key = f"chk:{RUN}:ta:{rows[0]['item_id']}:mark"
    at.radio(key=key).set_value("✗ wrong").run()
    assert not at.exception
    assert SC.load_check(RUN, "ta", checks)["rows"][0]["mark"] == "wrong"
    assert len(at.dataframe) == 1                                     # per-stage summary table
    assert discard not in [b.key for b in at.button]                  # marked: kept
