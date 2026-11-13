"""Part 10: evaluation (gold funnel, human labels, metrics, CLI) and the years fix.

A small hand-built run: 6 records, screening limit 4 (records r5, r6 not screened),
TA includes r1 r2 r3, FT includes r1 r2. Gold list of 6 articles:
  G1 = r1 by DOI (included)      G2 = r3 by PMID (lost at full text, I2)
  G3 = r4 by title (lost at TA, E1)   G4 = r5 (outside the screening limit)
  G5 covered but not found       G6 not covered
"""

import csv
import json
from datetime import date

import pytest

from sr_pipeline.evaluation.__main__ import compare, main
from sr_pipeline.evaluation.labels import make_sheet, read_labels, score_labels
from sr_pipeline.evaluation.metrics import METRICS, manual_estimate, metric, run_metrics
from sr_pipeline.evaluation.recall import gold_file, gold_funnel, match_paper
from sr_pipeline.protocol import YearRange


def rec(i, **kw):
    return {"paper_id": f"r{i}", "title": f"Exercise trial number {i} in older adults "
            "with depressive symptoms", "year": 2015, "abstract": f"abstract {i}",
            "rank": i, **kw}


def ta(i, final, criterion=None, by="agreement"):
    return {"paper_id": f"r{i}", "round": 1, "stage": "ta", "final": final,
            "criterion": criterion, "resolved_by": by}


def ft(i, final, criterion=None, status="pmc_xml"):
    return {"paper_id": f"r{i}", "round": 1, "stage": "ft", "final": final,
            "criterion": criterion, "resolved_by": "adjudicator", "fulltext_status": status}


PROTOCOL = {"question": "Does exercise reduce depression?",
            "study_designs": ["randomized controlled trial"],
            "inclusion_criteria": ["Randomized controlled trial", "Older adults over 60"],
            "exclusion_criteria": ["Comorbidities"],
            "full_text_criteria": ["Older adults over 60"],
            "criterion_labels": [{"criterion": "Comorbidities", "label": "comorbid population"}]}


def make_state():
    records = [rec(1, doi="10.1/one"), rec(2, doi="10.1/two"), rec(3, pmid="333"),
               rec(4, title="Tai chi and mood in community elders: a randomized trial",
                   year=2012),
               rec(5, doi="10.1/five"), rec(6, doi="10.1/six")]
    # an older round of r1 (excluded) must be ignored: latest() reads round 2
    old = {**ta(1, "exclude", "D"), "round": 0}
    return {
        "protocol": PROTOCOL, "unique_records": records,
        "ta_decisions": [old, ta(1, "include"), ta(2, "include"), ta(3, "include"),
                         ta(4, "exclude", "E1", "adjudicator")],
        "ft_decisions": [ft(1, "include"), ft(2, "include"),
                         ft(3, "exclude", "I2", "abstract_only")],
        "included": ["r1", "r2"],
        "prisma_counts": {
            "identification": {"databases": {"openalex": {"capped": False, "identified": 6}},
                               "identified": 6, "not_retrieved": 0, "duplicates_removed": 0},
            "title_abstract": {"records_screened": 4, "records_excluded": 1, "included": 3,
                               "screen_limit": 4, "not_screened": 2, "kappa": 0.5,
                               "agreement": 0.75, "pairs": 4,
                               "resolved_by": {"agreement": 3, "adjudicator": 1}},
            "full_text": {"reports_sought": 3, "reports_excluded": 1, "studies_included": 2,
                          "reports_not_retrieved": 1, "kappa": 1.0, "agreement": 1.0,
                          "pairs": 3, "resolved_by": {"adjudicator": 3}}},
        "qa_report": {"passed": True, "blockers": 0, "warnings": 2, "loops_used": 1},
        "extractions": [
            {"paper_id": "r1", "round": 1, "flags": [], "fields": {
                "n": {"status": "reported"}, "age": {"status": "not reported"}}},
            {"paper_id": "r2", "round": 1, "flags": ["abstract_only"], "fields": {
                "n": {"status": "reported"}, "age": {"status": "reported"}}}],
    }


GOLD = [
    {"row": "1", "first_author": "One", "year": "2015", "title": "x", "doi": "10.1/ONE",
     "pmid": "", "in_openalex": "yes", "in_pubmed": "no", "found_doi": "", "found_pmid": ""},
    {"row": "2", "first_author": "Three", "year": "2015", "title": "y", "doi": "",
     "pmid": "", "in_openalex": "no", "in_pubmed": "yes", "found_doi": "", "found_pmid": "333"},
    {"row": "3", "first_author": "Four", "year": "2013",
     "title": "Tai chi and mood in community elders: a randomised trial", "doi": "",
     "pmid": "", "in_openalex": "yes", "in_pubmed": "yes", "found_doi": "", "found_pmid": ""},
    {"row": "4", "first_author": "Five", "year": "2015", "title": "z", "doi": "10.1/five",
     "pmid": "", "in_openalex": "yes", "in_pubmed": "yes", "found_doi": "", "found_pmid": ""},
    {"row": "5", "first_author": "Missed", "year": "2015", "title": "never found",
     "doi": "10.1/missed", "pmid": "", "in_openalex": "yes", "in_pubmed": "no",
     "found_doi": "", "found_pmid": ""},
    {"row": "6", "first_author": "Chinese", "year": "2015", "title": "not covered",
     "doi": "", "pmid": "", "in_openalex": "no", "in_pubmed": "no", "found_doi": "",
     "found_pmid": ""},
]


def test_funnel_counts_and_reasons():
    f = gold_funnel(make_state(), GOLD)
    assert f["counts"] == {"gold": 6, "covered": 5, "found": 4, "screened": 3,
                           "ta_included": 2, "included": 1}
    by = {s["first_author"]: s for s in f["studies"]}
    assert by["One"]["reached"] == "included" and by["One"]["matched_by"] == "doi"
    assert by["Three"]["matched_by"] == "pmid"          # PMID found by the coverage lookup
    assert by["Three"]["lost_at"] == "included"
    assert "I2 (Older adults over 60)" in by["Three"]["reason"]
    assert "abstract_only" in by["Three"]["reason"]
    assert by["Four"]["matched_by"] == "title"          # randomised/randomized, year ±1
    assert by["Four"]["lost_at"] == "ta_included"
    assert "E1 (comorbid population)" in by["Four"]["reason"]
    assert by["Missed"]["lost_at"] == "found" and "query missed" in by["Missed"]["reason"]
    assert by["Chinese"]["lost_at"] == "covered"
    assert f["lost_by_stage"]["included"] == ["Three 2015"]


def test_not_screened_is_not_an_exclusion():
    f = gold_funnel(make_state(), GOLD)
    five = next(s for s in f["studies"] if s["first_author"] == "Five")
    assert five["lost_at"] == "screened" and five["ta"] == "not screened"
    assert "outside the 4 records screened" in five["reason"]
    # screening recall counts only the gold studies the screeners saw (3), not Five
    assert f["recall"]["screening_ta"] == round(2 / 3, 4)


def test_recall_both_denominators_and_precision():
    r = gold_funnel(make_state(), GOLD)
    assert r["recall"]["search_vs_all"] == round(4 / 6, 4)
    assert r["recall"]["search_vs_covered"] == round(4 / 5, 4)
    assert r["recall"]["final_vs_all"] == round(1 / 6, 4)
    assert r["recall"]["final_vs_covered"] == round(1 / 5, 4)
    assert r["precision_lower_bound"] == 0.5               # 1 gold of 2 included


def test_download_cap_named_when_a_database_was_capped():
    state = make_state()
    state["prisma_counts"]["identification"]["databases"]["openalex"]["capped"] = True
    missed = next(s for s in gold_funnel(state, GOLD)["studies"]
                  if s["first_author"] == "Missed")
    assert "download cap of openalex" in missed["reason"]


def test_title_match_rejects_conflicting_doi():
    papers = [rec(9, title="Tai chi and mood in community elders: a randomized trial",
                  doi="10.9/other", year=2012)]
    row = {**GOLD[2], "doi": "10.1/journal-version"}
    assert match_paper(row, papers) == (None, None)


def test_gold_file_prefers_coverage(tmp_path):
    (tmp_path / "T9.csv").write_text("row\n")
    assert gold_file(tmp_path / "T9.csv") == tmp_path / "T9.csv"
    (tmp_path / "T9_coverage.csv").write_text("row\n")
    assert gold_file(tmp_path / "T9.csv") == tmp_path / "T9_coverage.csv"


def test_label_sheet_is_blind_and_scores(tmp_path):
    state = make_state()
    out = make_sheet(state, tmp_path, n=10, seed=1)
    assert out["records"] == 4                            # only screened records
    rows = list(csv.DictReader(out["sheet"].open()))
    assert set(rows[0]) == {"n", "paper_id", "title", "abstract", "human", "note"}
    assert "include" not in out["sheet"].read_text().split("\n", 1)[1].lower()  # blind
    assert "*(full text only)*" in out["criteria"].read_text()
    with pytest.raises(FileExistsError):                  # labels are never overwritten
        make_sheet(state, tmp_path)

    answers = {"r1": "i", "r2": "exclude", "r3": "e", "r4": "include"}
    for r in rows:
        r["human"] = answers[r["paper_id"]]
    rows.append({"n": "99", "paper_id": "r5", "human": "maybe"})
    with out["sheet"].open("w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    labels, problems = read_labels(out["sheet"])
    assert problems == ["row 99: unknown label 'maybe'"]
    s = score_labels(state, labels)
    # human vs pipeline: r1 i/i (tp), r2 e/i (fp), r3 e/i (fp), r4 i/e (fn)
    assert s["confusion"] == {"tp": 1, "fn": 1, "fp": 2, "tn": 0}
    assert s["accuracy"] == 0.25 and s["sensitivity"] == 0.5 and s["specificity"] == 0.0
    # kappa: p_o = 1/4, p_e = (2*3 + 2*1)/16 = 0.5 -> (0.25 - 0.5) / 0.5 = -0.5
    assert s["kappa"] == -0.5
    miss = next(d for d in s["disagreements"] if d["paper_id"] == "r4")
    assert miss["criterion"] == "E1" and miss["label"] == "comorbid population"


def test_metrics_registry_and_selection():
    ctx = {"state": make_state(), "gold_rows": GOLD, "labels": None, "manual": None,
           "summary": {"seconds": 3600, "steps_taken": 9, "llm": {"calls": 10, "cached": 2,
                                                                 "cost_usd": 0.2},
                       "embeddings": {"cost_usd": 0.01}, "coordinator": {}}}
    res = run_metrics(ctx)
    assert set(res) == {"completeness", "accuracy", "consistency", "time"}
    assert "human_labels" not in res["accuracy"]            # not applicable: no labels
    e = res["completeness"]["extraction_completeness"]
    assert e["share_reported"] == 0.75
    assert e["by_text"]["abstract only"]["share"] == 1.0
    assert res["completeness"]["prisma_consistency"]["consistent"] is True
    assert res["consistency"]["screener_agreement"]["ta"]["adjudicated_share"] == 0.25
    t = res["time"]["time_cost"]
    assert t["cost_usd"] == 0.21 and t["hours"] == 1.0 and "manual" not in t
    only = run_metrics(ctx, ["gold_precision"])
    assert list(only["accuracy"]) == ["gold_precision"] and not only["completeness"]
    with pytest.raises(KeyError):
        run_metrics(ctx, ["nope"])


def test_plug_in_metric():
    @metric("toy", "accuracy")
    def toy(ctx):
        return {"n": len(ctx["state"]["included"])}
    try:
        assert run_metrics({"state": make_state()}, ["toy"])["accuracy"]["toy"] == {"n": 2}
    finally:
        METRICS.pop("toy")


def test_manual_estimate_never_guessed():
    counts = {"screened": 100, "full_text": 10, "included": 5}
    empty = {"items_per_day": {"title_abstract": 200, "full_text": None,
                               "extraction": 5, "risk_of_bias": 10}}
    assert manual_estimate(empty, counts) is None
    full = {"source": "test",
            "items_per_day": {"title_abstract": 200, "full_text": 40,
                              "extraction": 5, "risk_of_bias": 10},
            "checked_percent": {"title_abstract": 10, "full_text": 10}}
    m = manual_estimate(full, counts)
    # 100/200*1.1 + 10/40*1.1 + 5/5 + 5/10 = 0.55 + 0.275 + 1 + 0.5
    assert m["days_per_task"]["title_abstract"] == 0.55
    assert m["person_days"] == round(0.55 + 0.28 + 1 + 0.5, 1)


def write_run(tmp_path, name, limit):
    folder = tmp_path / name
    (folder / "report").mkdir(parents=True)
    (folder / "state.json").write_text(json.dumps(make_state()))
    (folder / "run.json").write_text(json.dumps({"run_id": name, "screen_limit": limit}))
    (folder / "report" / "run_summary.json").write_text(json.dumps(
        {"seconds": 1800, "llm": {"calls": 3, "cached": 0, "cost_usd": 0.1}}))
    return folder


def test_cli_evaluate_and_compare(tmp_path, monkeypatch):
    a = write_run(tmp_path, "run-a", 4)
    b = write_run(tmp_path, "run-b", 0)
    monkeypatch.setenv("SR_RUNS_DIR", str(tmp_path))
    assert main(["evaluate", "run-a", "--metrics", "critic,time_cost,coordinator",
                 "--results", str(tmp_path / "results")]) == 0
    assert (tmp_path / "results" / "run-a.md").exists()       # tracked copy
    md = (a / "evaluation" / "evaluation.md").read_text()
    assert "Critic: passed, 0 blockers, 2 warnings, 1 loops." in md
    assert "No human labels yet" in md
    data = json.loads((a / "evaluation" / "evaluation.json").read_text())
    assert data["time"]["time_cost"]["hours"] == 0.5
    table = compare([a, b])
    assert "| run-a | 4 |" in table and "| run-b | 0 |" in table
    with pytest.raises(SystemExit):
        main(["evaluate", "no-such-run"])


def test_years_explicit_null_start_means_no_limit():
    # Part 10 fix: the template's recent: 5 no longer replaces an explicit start: null
    assert YearRange.model_validate({"recent": 5, "start": None}).start is None
    assert YearRange.model_validate({"recent": 5}).start == date.today().year - 5
    # a validated protocol re-validates to the same years (start is then explicit)
    y = YearRange.model_validate(YearRange(recent=5).model_dump(mode="json"))
    assert y.start == date.today().year - 5 and y.recent == 5


def test_cli_sample_writes_to_tracked_folder(tmp_path, monkeypatch):
    write_run(tmp_path, "run-a", 4)
    monkeypatch.setenv("SR_RUNS_DIR", str(tmp_path))
    import sr_pipeline.evaluation.__main__ as cli
    monkeypatch.setattr(cli, "EVAL_DIR", tmp_path / "evaluation")
    assert main(["sample", "run-a", "--n", "2"]) == 0
    assert (tmp_path / "evaluation" / "labels" / "run-a" / "labels.csv").exists()


def test_extraction_completeness_counts_topic_fields_only():
    state = make_state()
    state["protocol"] = {**PROTOCOL, "extraction_fields": {"age": "Age?"}}
    e = run_metrics({"state": state, "gold_rows": None, "labels": None, "manual": None,
                     "summary": {}}, ["extraction_completeness"])
    e = e["completeness"]["extraction_completeness"]
    assert e["scope"] == "topic fields" and e["topic_fields"] == ["age"]
    assert (e["reported"], e["fields"], e["share_reported"]) == (1, 2, 0.5)
    assert e["by_text"]["full text"] == {"reported": 0, "fields": 1, "share": 0.0}
    assert e["all_fields"] == {"reported": 3, "fields": 4, "share": 0.75}


def test_render_keeps_stage_checks_with_topic_extraction_fields():
    """Regression (2026-11-13): the topic-fields extraction line reused the name of the
    accuracy dict, so the per-stage human check vanished from evaluation.md."""
    from pathlib import Path
    from sr_pipeline.evaluation.__main__ import render
    ext = {"scope": "topic fields", "reported": 1, "fields": 2, "topic_fields": ["a"],
           "share_reported": 0.5, "studies": 1, "by_text": {},
           "all_fields": {"reported": 1, "fields": 3, "share": 1 / 3}}
    check = {"name": "Full-text screening", "unit": "record", "sampled": 2, "population": 5,
             "seed": 10, "correct": 1, "wrong": 1, "unsure": 0, "unmarked": 0,
             "share_correct": 0.5, "ci95": (0.1, 0.9), "wrong_items": []}
    results = {"completeness": {"extraction_completeness": ext},
               "accuracy": {"stage_checks": {"ft": check}}, "consistency": {}, "time": {}}
    md = render({"run": {"run_id": "r"}, "folder": Path("r"), "topic": None}, results)
    assert "Per-stage human check" in md and "No human labels yet" not in md
