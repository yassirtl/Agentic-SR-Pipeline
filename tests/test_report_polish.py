"""Part 9b: report polish. Criterion labels + text inside the PRISMA boxes (fix 1), the
automation-tools box (fix 2), wording vs heterogeneity (fix 3), --report-only (fix 4), last-5-
years default and the screening limit (fix 5). Scripted LLM, no network."""

import json
import re
from datetime import date

import pytest

from sr_pipeline import cli
from sr_pipeline.agents.critic import code_checks, make_critic
from sr_pipeline.agents.retrieval import make_retrieval
from sr_pipeline.agents.screening import criterion_labels, screening_sample
from sr_pipeline.agents.synthesis import wording_problems
from sr_pipeline.evaluation.topics import load_topic, seed_protocol
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import Protocol, YearRange
from sr_pipeline.reporting.prisma import NBSP, draw_flow, flow_counts
from sr_pipeline.sources.base import Record, SearchResult

from test_report import (INJECT, QUESTION, check_outputs, e2e_config, e2e_llm)
from test_synthesis import HET, critic_state

# --- Fix 1: short labels -------------------------------------------------------------------

BASE = {"question": "q", "title": "t", "rationale": "r", "databases": ["openalex"],
        "pico": {"population": "older adults", "intervention": "exercise",
                 "comparator": "usual care", "outcomes": ["depressive symptoms"]},
        "inclusion_criteria": ["Participants aged 60 years or older in a community setting"],
        "exclusion_criteria": ["No control group"], "study_designs": []}


def test_labels_from_planner_are_checked_and_have_a_fallback():
    p = Protocol.model_validate(BASE | {"criterion_labels": [
        {"criterion": "No control group", "label": "no control group"},
        {"criterion": "No control group", "label": "duplicate, ignored"},
        {"criterion": "Reworded criterion", "label": "stale"},                 # not a criterion
        {"criterion": BASE["inclusion_criteria"][0],
         "label": "far too many words for a diagram label"}]}).model_dump()
    assert p["criterion_labels"] == [{"criterion": "No control group",
                                      "label": "no control group"}]
    labels = criterion_labels(p)
    assert labels["E1"] == "no control group"
    assert labels["I1"] == "Participants aged 60 years or..."      # fallback: first 5 words
    assert labels["D"] == "not a primary study"                     # no design restriction
    assert labels["human"] == "excluded by a human"


def test_topic_labels_reach_the_protocol_and_the_diagram():
    topic = load_topic("T3")
    p, audit = seed_protocol(Protocol.model_validate(BASE).model_dump(mode="json"), topic)
    assert "criterion_labels" in audit["seeded_fields"]
    labels = criterion_labels(p)
    assert labels["D"] == "not a primary RCT" and labels["E1"] == "comorbidities or under 60"
    assert all(len(v.split()) <= 5 for v in labels.values())
    c = flow_counts({"title_abstract": {"records_excluded": 24,
                                        "excluded_by_criterion": {"D": 21, "E1": 3}}}, p)
    assert c["ta_excluded"]["lines"][1:] == [
        f"  not a primary RCT (D):{NBSP}n{NBSP}={NBSP}21",
        f"  comorbidities or under 60 (E1):{NBSP}n{NBSP}={NBSP}3"]


# --- Fix 1: no text outside its box --------------------------------------------------------

def stress_counts():
    """Long database names, many reasons, 6-digit numbers: what overflowed before."""
    crit = {f"I{i}": i for i in range(1, 9)} | {f"E{i}": 10 + i for i in range(1, 6)}
    return {
        "identification": {"databases": {f"database number {i} with a long name":
                                         {"identified": 123456} for i in range(4)},
                           "identified": 493824, "not_retrieved": 480000,
                           "duplicates_removed": 3000, "max_records": 3000,
                           "errors": {"arxiv": "timeout"}},
        "title_abstract": {"records_screened": 100, "records_excluded": 80,
                           "excluded_by_criterion": crit | {"D": 30}, "title_rule": 12,
                           "screen_limit": 100, "not_screened": 10724, "included": 20},
        "full_text": {"reports_sought": 20, "reports_not_retrieved": 9, "reports_excluded": 8,
                      "excluded_by_criterion": crit, "studies_included": 12,
                      "included_on_abstract_only": 5},
    }


@pytest.mark.parametrize("protocol", [None, BASE | {"criterion_labels": []}])
def test_every_text_stays_inside_its_box(protocol):
    fig, pairs, title = draw_flow(stress_counts(), protocol)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    assert len(pairs) == 9 + 3                      # 9 boxes + 3 phase bars
    for rect, text in pairs:
        box, ext = rect.get_window_extent(r), text.get_window_extent(r)
        assert box.x0 <= ext.x0 and ext.x1 <= box.x1, text.get_text()
        assert box.y0 <= ext.y0 and ext.y1 <= box.y1, text.get_text()
        # "I5 n = / 1" (Part 9 preview): a count never splits from its "n =".
        for line in text.get_text().splitlines():
            assert not re.search(r"(?:\bn|=)\s*$", line), line
    fb = fig.bbox
    t = title.get_window_extent(r)
    assert fb.x0 <= t.x0 and t.x1 <= fb.x1 and t.y1 <= fb.y1
    # Long lines were wrapped, not shortened.
    removed = next(tx.get_text() for _, tx in pairs if "removed before" in tx.get_text())
    assert "\n" in removed and "automation tools" in removed


# --- Fix 2 + 5b: automation box and not-screened records ----------------------------------

def test_automation_and_screening_limit_boxes_add_up():
    pc = stress_counts()
    pc["identification"].update(identified=14000, not_retrieved=0, duplicates_removed=3176)
    c = flow_counts(pc)
    n = c["numbers"]
    # The 12 title-rule records leave "records excluded" (and D) for the automation box.
    assert (n["automation"], n["screened"], n["ta_excluded"]) == (12, 88, 68)
    assert f"(D):{NBSP}n{NBSP}={NBSP}18" in "\n".join(c["ta_excluded"]["lines"])
    assert c["removed"]["n"] == 3176 + 12 + 10724
    assert any("automation tools" in x for x in c["removed"]["lines"])
    assert any("screening limit of 100" in x for x in c["removed"]["lines"])
    # 14000 - 0 - 3176 - 12 - 10724 = 88 screened; 88 - 68 = 20 sought; 20 - 8 = 12.
    assert c["checks"] == dict.fromkeys(c["checks"], True) and c["consistent"]
    pc["title_abstract"]["not_screened"] = 0             # a gap is never hidden
    assert not flow_counts(pc)["consistent"]


def test_retrieval_ranks_papers_and_the_limit_takes_the_best_of_every_database():
    def rec(src, i, doi):
        return Record(id=f"{src}:{i}", source=src, source_id=str(i), doi=doi,
                      title=f"Paper {src} {i} on walking and mood", year=2020)

    def search(strategy, http, **kw):
        return ({"openalex": SearchResult(source="openalex", query="q", total_hits=4, records=[
                    rec("openalex", i, f"10.1/o{i}") for i in range(1, 5)]),
                 "pubmed": SearchResult(source="pubmed", query="q", total_hits=3, records=[
                     rec("pubmed", 1, "10.1/o4"), rec("pubmed", 2, "10.1/p2"),
                     rec("pubmed", 3, "10.1/p3")])}, {})

    papers = make_retrieval(None, search=search)(
        {"search_strategy": {"queries": {}}, "completed_stages": []})["unique_records"]
    rank = {p["paper_id"]: p["rank"] for p in papers}
    assert rank["doi:10.1/o4"] == 1            # openalex 4th, but pubmed's top hit
    picked = [p["paper_id"] for p in screening_sample(papers, 3)]
    # Ranks: o1 1, o4 1, o2 2, p2 2, ... -> ties keep retrieval order; pubmed is not starved.
    assert picked == ["doi:10.1/o1", "doi:10.1/o2", "doi:10.1/o4"]
    assert screening_sample(papers, 0) == papers and screening_sample(papers, None) == papers


def test_cli_limit_screens_the_top_records_and_the_diagram_says_so(tmp_path):
    cfg = e2e_config(tmp_path)
    run_id, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True, screen_limit=3,
                            ask=lambda p: "", out=lambda *a: None, llm=e2e_llm(), **INJECT)
    ta = state["prisma_counts"]["title_abstract"]
    assert (ta["records_screened"], ta["not_screened"], ta["screen_limit"]) == (3, 2, 3)
    assert ta["records_available"] == 5
    assert json.loads((cfg.runs_dir / run_id / "run.json").read_text())["screen_limit"] == 3
    assert state["report"]["prisma_consistent"]
    md = (cfg.runs_dir / run_id / "report" / "manuscript.md").read_text()
    assert "2 records above the screening limit of 3 (not screened)" in md
    assert "Only the 3 best-ranked unique records were screened" in md
    # The 3 best-ranked are the three trials: the review still pools them.
    assert state["synthesis"]["outcomes"][0]["method"] == "meta-analysis"


def test_default_limit_comes_from_the_config(tmp_path):
    cfg = e2e_config(tmp_path).model_copy(update={"screen_limit": 4})
    run_id, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True,
                            ask=lambda p: "", out=lambda *a: None, llm=e2e_llm(), **INJECT)
    assert state["prisma_counts"]["title_abstract"]["records_screened"] == 4


# --- Fix 3: wording vs heterogeneity -------------------------------------------------------

def group(i2=None, sign_p=0.6):
    g = {"id": "O1", "vote_count": {"sign_test_p": sign_p}, "meta": None}
    if i2 is not None:
        g["meta"] = {"i2": i2, "k": 3}
    return g


def sent(text, studies=("S1", "S2", "S3")):
    return {"text": text, "studies": list(studies)}


def test_wording_rules():
    high = group(i2=0.94)
    # The Part 9 live sentence.
    live = sent("The overall result indicated a negligible effect of exercise on depressive "
                "symptoms with an SMD of -0.10 (95% CI -1.48 to 1.28).")
    assert [w["kind"] for w in wording_problems(high, [live])] == ["effect_size_wording"]
    for bad in ("The pooled effect was small.", "Exercise had a moderate benefit overall.",
                "Exercise had no effect on depressive symptoms."):
        assert wording_problems(high, [sent(bad)]), bad
    for ok in ("The pooled SMD is not interpretable as a single effect.",
               "Two large trials favoured exercise.", "Sample sizes were small.",
               "Heterogeneity was substantial (I2 94%)."):
        assert not wording_problems(high, [sent(ok)]), ok
    # One study described on its own (no pooled word) may get a size word.
    assert not wording_problems(high, [sent("This trial showed a large reduction.", ["S2"])])
    # Below the I2 limit, the size of a pooled result may be described.
    assert not wording_problems(group(i2=0.30), [sent("The pooled effect was small.")])
    # Narrative path: no 'no effect' while the sign test is not significant (or missing)...
    for p in (0.625, None):
        w = wording_problems(group(sign_p=p), [sent("Exercise had no effect.")])
        assert w and "no evidence of a direction" in w[0]["reason"]
    assert not wording_problems(group(), [sent("There was no evidence of a direction.")])
    # ...but it is not this rule's business when the sign test is significant.
    assert not wording_problems(group(sign_p=0.01), [sent("Exercise had no effect.")])


def test_high_i2_wording_is_a_critic_blocker(tmp_path):
    narrative = {"sentences": [
        {"section": "O1", "text": "The pooled result showed a negligible effect of exercise.",
         "studies": ["S1", "S2", "S3"]}], "heterogeneity": [HET]}
    st = critic_state(tmp_path, narrative)
    g = st["synthesis"]["outcomes"][0]
    assert g["meta"], "fixture should pool 3 trials"
    g["meta"]["i2"] = 0.94
    kinds = {i["kind"]: i for i in code_checks(st)}
    assert kinds["effect_size_wording"]["severity"] == "blocker"
    assert "not interpretable as a single effect" in kinds["effect_size_wording"]["detail"]
    qa = make_critic(ScriptedLLM({"critic": [{"issues": []}]}))(st)["qa_report"]
    assert not qa["passed"] and qa["loop_back"]


# --- Fix 4: --report-only ------------------------------------------------------------------

def test_report_only_rebuilds_a_finished_run_without_changing_its_facts(tmp_path):
    cfg = e2e_config(tmp_path)
    run_id, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True,
                            ask=lambda p: "", out=lambda *a: None, llm=e2e_llm(), **INJECT)
    report = cfg.runs_dir / run_id / "report"
    before = json.loads((report / "run_summary.json").read_text())
    (report / "manuscript.md").unlink()
    (report / "prisma_flow.png").write_bytes(b"")

    said = []
    files = cli.report_only(cfg, run_id, out=said.append)       # no LLM, no network
    assert "rebuilt" in said[0] and "add up" in said[0]
    check_outputs(cfg, run_id, state)
    after = json.loads((report / "run_summary.json").read_text())
    for key in ("finished_at", "seconds", "embeddings", "llm"):
        assert after[key] == before[key], key
    assert set(files) == set(state["report"]["files"])
    audit = (cfg.runs_dir / run_id / "audit.jsonl").read_text().splitlines()
    assert json.loads(audit[-1])["event"] == "report re-rendered (--report-only)"


def test_report_only_refuses_an_unfinished_run(tmp_path, capsys):
    cfg = e2e_config(tmp_path)
    (cfg.runs_dir / "r1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="--resume r1"):
        cli.report_only(cfg, "r1")
    (cfg.runs_dir / "r1" / "state.json").write_text(json.dumps({"completed_stages": []}))
    with pytest.raises(ValueError, match="before the report stage"):
        cli.report_only(cfg, "r1")
    with pytest.raises(SystemExit):
        cli.main(["--report-only", "r1", "--auto"])
    with pytest.raises(SystemExit):
        cli.main(["--resume", "r1", "--limit", "5"])
    with pytest.raises(SystemExit):
        cli.main(["question", "--limit", "-1"])


# --- Fix 5a: last 5 years by default -------------------------------------------------------

def test_recent_years():
    assert YearRange(recent=5).start == date.today().year - 5
    assert YearRange(recent=5, end_date="2024-01-07").start == 2019
    assert YearRange(recent=5, end=2020).start == 2015
    assert YearRange(recent=5, start=2010).start == 2010              # explicit start wins
    assert YearRange().start is None                                    # no limit
    assert load_topic("T3")["years"]["start"] is None   # a topic's window replaces the default
