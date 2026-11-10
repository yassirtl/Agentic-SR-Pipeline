"""Part 11a: run viewer + cost estimator. Synthetic run folders only (runs/ is git-ignored)."""

import json
from pathlib import Path

import pytest

from sr_pipeline.llm import compute_cost
from sr_pipeline.ui import data as D

MODELS = {a: "gpt-4o-mini" for a in ("coordinator", "planner", "search_strategy", "screener_a",
                                      "screener_b", "adjudicator", "extraction", "synthesis",
                                      "critic")}
FULL = ["planner", "search_strategy", "retrieval", "ta_screening", "ft_screening", "extraction",
        "synthesis", "critic", "synthesis", "critic", "report"]


def usage(calls, cached, inp, out):
    billed = compute_cost("gpt-4o-mini", inp, out) * (calls - cached) / calls
    return {"calls": calls, "cached": cached, "input_tokens": inp, "output_tokens": out,
            "cost_usd": round(billed, 6), "unpriced": 0}


def make_run(root: Path, run_id: str, *, screened=100, ta_inc=40, ft_inc=20, seconds=600.0,
             completed=FULL, scale=1.0, topic="T1", budget=False, report=True) -> Path:
    d = root / run_id
    (d / "report").mkdir(parents=True)
    (d / "evaluation").mkdir()
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "topic": topic, "screen_limit": 100,
                                            "models": MODELS, "question": None,
                                            "started_at": "2026-11-08T10:00:00+00:00"}))
    per_node = {
        "coordinator": usage(10, 0, 4000, 300),
        "planner": usage(1, 0, 2000, 500),
        "search_strategy": usage(2, 0, 3000, 600),
        "ta_screening": usage(250, 50, int(500_000 * scale), int(50_000 * scale)),
        "ft_screening": usage(120, 0, int(400_000 * scale), int(30_000 * scale)),
        "extraction": usage(300, 100, int(600_000 * scale), int(60_000 * scale)),
        "synthesis": usage(2, 0, 20_000, 2000),
        "critic": usage(2, 0, 20_000, 500),
    }
    emb = {"model": "text-embedding-3-small", "input_tokens": 200_000, "api_calls": 3,
           "texts_embedded": 100, "texts_cached": 0, "cost_usd": 0.004}
    llm = {k: sum(n[k] for n in per_node.values())
           for k in ("calls", "cached", "input_tokens", "output_tokens", "cost_usd", "unpriced")}
    llm["cost_usd"] = round(llm["cost_usd"] + emb["cost_usd"], 6)
    audit = [
        {"node": "coordinator", "step": 1, "proposed": "planner", "rationale": "start",
         "outcome": "accepted", "model": "gpt-4o-mini", "input_tokens": 400, "output_tokens": 30,
         "cached": False, "cost_usd": 0.0001},
        {"node": "planner", "step": 1, "event": "protocol drafted"},
        {"node": "coordinator", "step": 5, "proposed": "search_strategy",
         "rationale": "search <again>", "outcome": "refused",
         "reason": "screening included 40 records: no reason to search again",
         "substituted": "ft_screening", "model": "gpt-4o-mini", "input_tokens": 400,
         "output_tokens": 30, "cached": False, "cost_usd": 0.0001},
        {"node": "extraction", "step": 6, "event": "extraction and risk of bias", "studies": ft_inc},
        {"node": "synthesis", "step": 7, "event": "synthesis", "studies": ft_inc, "sentences": 12},
        {"node": "critic", "step": 8, "event": "quality check", "blockers": 0, "warnings": 2,
         "passed": True},
        {"node": "report", "step": 11, "event": "report written", "files": {"a": 1, "b": 2}},
    ]
    if budget:
        audit.append({"node": "coordinator", "event": "budget exhausted"})
    (d / "audit.jsonl").write_text("\n".join(json.dumps(a) for a in audit) + "\n")
    if not report:
        return d
    prisma = {
        "identification": {"databases": {"openalex": {}, "pubmed": {}}, "identified": 130,
                           "records_screened": screened},
        "title_abstract": {"records_screened": screened, "included": ta_inc, "kappa": 0.41,
                           "agreement": 0.7, "pairs": screened,
                           "excluded_by_criterion": {"D": 30, "I1": screened - ta_inc - 30}},
        "full_text": {"reports_sought": ta_inc, "studies_included": ft_inc, "kappa": 0.6,
                      "reports_not_retrieved": 10, "excluded_by_criterion": {"I2": ta_inc - ft_inc}},
    }
    (d / "report" / "prisma_counts.json").write_text(json.dumps(prisma))
    summary = {"run_id": run_id, "question": "Do <smartwatches> detect AF?", "seconds": seconds,
               "started_at": "2026-11-08T10:00:00+00:00", "steps_taken": 11,
               "stages_run": completed[:-1] if completed[-1:] == ["report"] else completed,
               "coordinator": {"decisions": {"accepted": 10, "refused": 1}},
               "budget_exhausted": budget, "llm": {"per_node": per_node, **llm},
               "embeddings": emb}
    (d / "report" / "run_summary.json").write_text(json.dumps(summary))
    decision = {"stage": "ta", "title": "Paper 1", "flags": [],
                "a": {"decision": "include", "quote": "", "verdicts": []},
                "b": {"decision": "exclude"}}
    state = {
        "completed_stages": completed,
        "protocol": {"inclusion_criteria": ["I1", "I2"], "exclusion_criteria": ["E1"]},
        "ta_decisions": [
            {**decision, "paper_id": "p1", "round": 1, "final": "include", "resolved_by": "agreement",
             "criterion": "all"},
            {**decision, "paper_id": "p1", "round": 2, "final": "exclude",
             "resolved_by": "adjudicator", "criterion": "D",
             "adjudicator": {"decision": "exclude", "quote": "a narrative review"}},
            {**decision, "paper_id": "p2", "round": 1, "final": "exclude", "resolved_by": "agreement",
             "criterion": "I1",
             "a": {"decision": "exclude", "quote": "",
                   "verdicts": [{"criterion": "I1", "met": "no", "quote": "in mice"}]}},
        ],
        "ft_decisions": [],
        "synthesis": {"text": "### O1\n\nNarrative."},
        "qa_report": {"passed": True, "blockers": 0, "warnings": 1,
                      "issues": [{"severity": "warning", "kind": "llm_unsupported_claim",
                                  "detail": "sentence 2", "evidence": "table row"}]},
    }
    (d / "state.json").write_text(json.dumps(state))
    (d / "evaluation" / "evaluation.json").write_text(json.dumps({"completeness": {"gold_recall": {
        "counts": {"gold": 10, "covered": 10, "found": 8, "screened": 8, "ta_included": 7,
                   "included": 6},
        "recall": {"search_vs_all": 0.8, "final_vs_all": 0.6, "screening_ta_ft": 0.75},
        "studies": [{"first_author": "Kim", "year": "2019", "lost_at": "found",
                     "reason": "query missed it"}]}}}))
    return d


@pytest.fixture
def runs_root(tmp_path):
    make_run(tmp_path, "20261108-100000-aaaaaa")
    make_run(tmp_path, "20261108-110000-bbbbbb", screened=200, ta_inc=100, ft_inc=50,
             seconds=1200, scale=2.0, topic="T3")
    make_run(tmp_path, "20261108-090000-cccccc", report=False)          # stopped run
    (tmp_path / "results").mkdir()                                      # not a run folder
    (tmp_path / "results" / "x.json").write_text("{}")
    return tmp_path


def test_discover_runs_statuses_and_skips(runs_root, tmp_path):
    make_run(tmp_path, "20261108-120000-dddddd", budget=True)
    runs = D.discover_runs(runs_root)
    by_id = {r.run_id: r for r in runs}
    assert [r.run_id for r in runs] == sorted(by_id, reverse=True)       # newest first
    assert "results" not in by_id
    assert by_id["20261108-090000-cccccc"].status == "incomplete"
    assert by_id["20261108-120000-dddddd"].status == "budget exhausted"
    r = by_id["20261108-100000-aaaaaa"]
    assert (r.status, r.topic, r.screened, r.included, r.seconds) == ("finished", "T1", 100, 20, 600)
    summary = json.loads((r.path / "report" / "run_summary.json").read_text())
    assert r.cost_usd == summary["llm"]["cost_usd"]
    assert D.discover_runs(tmp_path / "missing") == []


def test_stage_cards_match_summary(runs_root):
    run = D.load_run(runs_root / "20261108-100000-aaaaaa")
    cards = D.stage_cards(run)
    assert [c.stage for c in cards] == list(D.STAGE_LABELS)
    assert all(c.status == "done" for c in cards)
    per_node = run.summary["llm"]["per_node"]
    for c in cards:
        u = per_node.get(c.stage, {})
        assert (c.calls, c.cached, c.cost_usd) == (u.get("calls", 0), u.get("cached", 0),
                                                  u.get("cost_usd", 0.0))
    # stage cards + coordinator + embeddings = the run total in run_summary.json
    total = (sum(c.cost_usd for c in cards) + per_node["coordinator"]["cost_usd"]
             + run.summary["embeddings"]["cost_usd"])
    assert total == pytest.approx(run.summary["llm"]["cost_usd"], abs=1e-6)
    by = {c.stage: c for c in cards}
    assert (by["ta_screening"].count_in, by["ta_screening"].count_out) == ("100 screened", "40 included")
    assert by["ft_screening"].count_out == "20 included"
    assert by["synthesis"].runs == 2
    assert by["search_strategy"].refused == 1
    assert by["planner"].count_out == "3 criteria"
    assert by["critic"].count_out == "passed"


def test_stale_and_not_run_status(runs_root):
    d = runs_root / "20261108-100000-aaaaaa"
    state = json.loads((d / "state.json").read_text())
    state["completed_stages"] = ["planner", "search_strategy", "retrieval", "ta_screening",
                                 "retrieval"]
    (d / "state.json").write_text(json.dumps(state))
    status = {c.stage: c.status for c in D.stage_cards(D.load_run(d))}
    assert status["retrieval"] == "done"
    assert status["ta_screening"] == "stale"        # retrieval re-ran after it
    assert status["ft_screening"] == "not run"


def test_completed_without_state_uses_summary(runs_root):
    d = runs_root / "20261108-100000-aaaaaa"
    (d / "state.json").unlink()                     # demo copies may lack big files
    run = D.load_run(d)
    assert run.completed == FULL
    assert D.screening_rows(run, "ta") == []


def test_coordinator_steps(runs_root):
    steps = D.coordinator_steps(D.load_run(runs_root / "20261108-100000-aaaaaa"))
    assert [s["outcome"] for s in steps] == ["accepted", "refused"]
    refused = steps[1]
    assert (refused["proposed"], refused["ran"]) == ("search_strategy", "ft_screening")
    assert "no reason to search again" in refused["reason"]


def test_screening_rows_latest_round_and_quote(runs_root):
    rows = D.screening_rows(D.load_run(runs_root / "20261108-100000-aaaaaa"), "ta")
    assert [r["paper_id"] for r in rows] == ["p1", "p2"]                 # one row per record
    p1, p2 = rows
    assert (p1["decision"], p1["resolved_by"], p1["quote"]) == ("exclude", "adjudicator",
                                                                "a narrative review")
    assert p2["quote"] == "in mice"                 # verdict quote of the deciding criterion


def test_run_rates_use_all_tokens_not_billed_cost(runs_root):
    run = D.load_run(runs_root / "20261108-100000-aaaaaa", with_state=False)
    r = D.run_rates(run)
    assert (r.ta_rate, r.ft_rate) == (0.4, 0.5)
    assert r.per_unit["ta_screening"] == {"calls": 2.5, "input": 5000, "output": 500}
    assert r.per_unit["extraction"]["input"] == 30_000                   # per included study
    assert r.sec_per_call == pytest.approx(600 / (687 - 150))
    # fresh cost re-prices every token: higher than billed (cached calls billed $0)
    assert D.fresh_cost(run) > run.summary["llm"]["cost_usd"] - run.summary["embeddings"]["cost_usd"]
    assert D.run_rates(D.load_run(runs_root / "20261108-090000-cccccc")) is None


def test_estimate_single_run_by_hand(runs_root):
    r = D.run_rates(D.load_run(runs_root / "20261108-100000-aaaaaa", with_state=False))
    est = D.estimate([r], limit=200)
    by = {s.stage: s for s in est["stages"]}
    ta = by["ta_screening"]
    assert ta.cost.mid == pytest.approx(compute_cost("gpt-4o-mini", 5000, 500) * 200)
    assert ta.calls.mid == pytest.approx(500)
    assert est["volume"]["sought"].mid == pytest.approx(80)
    assert est["volume"]["included"].mid == pytest.approx(40)
    ext = by["extraction"]
    assert ext.cost.mid == pytest.approx(compute_cost("gpt-4o-mini", 30_000, 3000) * 40)
    assert by["coordinator"].calls.mid == 10                               # per run, not scaled
    emb = compute_cost("text-embedding-3-small", 5000, 0) * 80            # 200k tokens / 40 sought
    assert est["embeddings"].mid == pytest.approx(emb)
    assert est["cost"].mid == pytest.approx(sum(s.cost.mid for s in est["stages"]) + emb)
    assert est["cost"].low == est["cost"].mid == est["cost"].high


def test_estimate_range_is_over_projected_runs_and_model_choice(runs_root):
    rates = [D.run_rates(D.load_run(runs_root / rid, with_state=False))
             for rid in ("20261108-100000-aaaaaa", "20261108-110000-bbbbbb")]
    est = D.estimate(rates, limit=100)
    projections = [D.project(r, limit=100) for r in rates]
    assert est["cost"].low == pytest.approx(min(p["cost"] for p in projections))
    assert est["cost"].high == pytest.approx(max(p["cost"] for p in projections))
    assert est["cost"].low < est["cost"].high
    dear = D.estimate(rates, limit=100, models={"extraction": "gpt-4o"})
    ext = {s.stage: s for s in dear["stages"]}["extraction"]
    assert ext.model == "gpt-4o"
    assert dear["cost"].mid > est["cost"].mid
    with pytest.raises(ValueError):
        D.estimate([], limit=100)


# --- Pages render (Streamlit test harness, no browser, no API) -------------------------

def _page(runs_root: Path, fn: str, run_id: str | None = None):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              f"P.RUNS_DIR = Path({str(runs_root)!r})\nP.setup()\nP.{fn}()\n")
    at = AppTest.from_string(script, default_timeout=60)
    if run_id:
        at.session_state["run_id"] = run_id
    return at.run()


@pytest.mark.parametrize("run_id", ["20261108-100000-aaaaaa", "20261108-090000-cccccc"])
def test_detail_page_renders(runs_root, run_id):
    at = _page(runs_root, "page_detail", run_id)
    assert not at.exception
    html = "".join(m.value for m in at.markdown)
    if run_id.endswith("aaaaaa"):
        assert "Title/abstract screening" in html
        assert "&lt;smartwatches&gt;" in html                              # run text is escaped
        assert "<smartwatches>" not in html


def test_runs_and_estimator_pages_render(runs_root):
    at = _page(runs_root, "page_runs")
    assert not at.exception
    assert len(at.dataframe[0].value) == 2                                 # finished runs only
    at = _page(runs_root, "page_estimator")
    assert not at.exception
    assert any("Estimated cost" in m.value for m in at.markdown)
