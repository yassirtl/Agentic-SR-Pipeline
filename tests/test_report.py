"""Part 9: critic rule (step 0), report maths and content rules, report files, CLI and a full
fake-LLM end-to-end run. Scripted LLM, fake embedder, fake search and full text, no network."""

import json
import math
import re
import sqlite3
from pathlib import Path

import pytest

from sr_pipeline import cli
from sr_pipeline.agents.critic import confirm_mismatch, make_critic
from sr_pipeline.config import Config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.meta import dersimonian_laird, prediction_interval, t_cdf, t_quantile
from sr_pipeline.rag import FakeEmbedder
from sr_pipeline.sources.base import Record, SearchResult
from sr_pipeline.sources.fulltext import FullText, Section

from test_synthesis import HET, critic_state

# --- Step 0: an LLM objection is a warning unless code confirms a wrong value ----------------

SUPPORTS = "favours intervention 3, favours control 0"   # a real line that SUPPORTS sentence 1


def objection(**kw):
    return {"issues": [{"sentence": 1, "problem": "p", "severity": "blocker"} | kw]}


def narrative(text, studies=("S1", "S2", "S3")):
    return {"sentences": [{"section": "O1", "text": text, "studies": list(studies)}],
            "heterogeneity": [HET]}


def test_supportive_quote_no_longer_triggers_a_loop(tmp_path: Path):
    st = critic_state(tmp_path, narrative("All three trials favoured exercise."))
    qa = make_critic(ScriptedLLM({"critic": [objection(evidence=SUPPORTS)]}))(st)["qa_report"]
    issue = qa["issues"][0]
    assert issue["evidence_verified"] and not issue["confirmed_by_code"]
    assert issue["severity"] == "warning" and issue["severity_claimed"] == "blocker"
    assert qa["passed"] and not qa["loop_back"]
    # Naming a "wrong" value that the quote also prints (3 = 3) changes nothing.
    qa = make_critic(ScriptedLLM({"critic": [objection(evidence=SUPPORTS, wrong_value="three",
                                                       table_value="3")]}))(st)["qa_report"]
    assert qa["passed"] and qa["issues"][0]["check"] == "sentence value equals the table value"


def test_wrong_count_confirmed_by_code_is_a_blocker(tmp_path: Path):
    # The 8 run's error: a count written in words that the code-computed count line contradicts.
    st = critic_state(tmp_path, narrative("Two trials favoured exercise."))
    line = "Vote count by direction of effect: favours intervention 3"
    qa = make_critic(ScriptedLLM({"critic": [objection(evidence=line, wrong_value="Two",
                                                       table_value="3")]}))(st)["qa_report"]
    issue = qa["issues"][0]
    assert issue["confirmed_by_code"] and issue["severity"] == "blocker"
    assert not qa["passed"] and qa["loop_back"]


def test_confirm_mismatch_rules():
    tables = "S1: Hedges' g -0.49 (95% CI -1.11 to 0.13). Risk of bias: high 2, some concerns 1."
    sent = "S1 had g -0.94."
    assert confirm_mismatch(sent, "-0.94", "-0.49", "Hedges' g -0.49", tables)[0]
    # Each condition that blocks a promotion:
    assert confirm_mismatch(sent, "", "", "Hedges' g -0.49", tables)[1] == \
        "no checkable value named"
    assert "not written in the sentence" in confirm_mismatch(sent, "0.5", "-0.49",
                                                             "Hedges' g -0.49", tables)[1]
    assert "not found" in confirm_mismatch(sent, "-0.94", "-0.49", "g was -0.49", tables)[1]
    assert "not printed in the quote" in confirm_mismatch(sent, "-0.94", "0.13",
                                                          "Hedges' g -0.49", tables)[1]
    # The quote prints the sentence's count too: code cannot tell which one was meant.
    assert "also prints 1" in confirm_mismatch("One study was at high risk.", "One", "2",
                                               "high 2, some concerns 1", tables)[1]
    # Study labels are not numbers.
    assert not confirm_mismatch("S2 favoured exercise.", "2", "1", "some concerns 1",
                                tables)[0]


# --- Report maths: prediction interval ---------------------------------------------------------

def test_t_quantile_matches_printed_t_table():
    # Two-sided 95% critical values as printed in standard t tables (3 decimals).
    table = {1: 12.706, 2: 4.303, 3: 3.182, 5: 2.571, 10: 2.228, 30: 2.042, 120: 1.980}
    for df, t in table.items():
        assert t_quantile(0.975, df) == pytest.approx(t, abs=5e-4)
    assert t_cdf(0.0, 4) == pytest.approx(0.5)
    assert t_quantile(0.025, 3) == pytest.approx(-3.182, abs=5e-4)
    assert t_quantile(0.975, 1e6) == pytest.approx(1.95996, abs=1e-4)   # -> normal


def test_prediction_interval_on_the_hand_example():
    # Hand-checkable inputs: pooled 0.5, SE^2 = 0.01, tau^2 = 0.03, k = 3 -> df 1.
    m = {"k": 3, "pooled": 0.5, "se": 0.1, "tau2": 0.03}
    lo, hi = prediction_interval(m)
    half = 12.7062047 * math.sqrt(0.04)
    assert (lo, hi) == (pytest.approx(0.5 - half, abs=1e-5), pytest.approx(0.5 + half, abs=1e-5))
    assert prediction_interval(m | {"k": 2}) is None
    # Always at least as wide as the confidence interval.
    dl = dersimonian_laird([-0.49, -0.20, -0.82], [0.1, 0.065, 0.085])
    lo, hi = prediction_interval(dl)
    assert lo < dl["ci_low"] and hi > dl["ci_high"]


# --- Full fake-LLM end-to-end run (CLI wiring, real stages, real graph) -----------------------


ROOT = Path(__file__).resolve().parents[1]
QUESTION = "Does exercise reduce depressive symptoms in older adults?"
QUOTE = "Older adults aged 65 were randomized to walking or usual care"
DRAFT = {
    "title": "Exercise for depressive symptoms in older adults",
    "pico": {"population": "older adults", "intervention": "exercise",
             "comparator": "usual care", "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": ["Participants aged 60 years or older",
                           "Depressive symptoms measured"],
    "exclusion_criteria": ["Not an exercise intervention"],
    "study_designs": [], "suggested_designs": ["randomized controlled trial"],
    "design_reason": "effect question", "rationale": "r",
    "outcome_directions": [{"outcome": "depressive symptoms", "better": "lower"}],
    "databases": ["pubmed", "openalex"], "database_reason": "health question",
}
PLAN = {"concepts": [
    {"role": "population", "name": "older adults", "keywords": ["older adults"]},
    {"role": "intervention", "name": "exercise", "keywords": ["exercise"]},
    {"role": "outcome", "name": "depression", "keywords": ["depressive symptoms"]}],
    "rationale": "three blocks"}
# Per-arm results printed in the abstracts (n, mean, SD verified in the quote).
RESULT = ("After 12 weeks the GDS-15 score was {} ({}) in the exercise group (n = {}) and {} "
          "({}) in the usual care group (n = {}).")
TRIALS = {"AGREE_IN_A": (4.1, 2.0, 30, 5.6, 2.4, 30), "AGREE_IN_B": (6.0, 3.0, 40, 6.5, 3.1, 40),
          "AGREE_IN_C": (7.2, 2.8, 25, 9.0, 3.0, 25)}
MARKERS = list(TRIALS) + ["AGREE_OUT", "SPLIT_UNRES"]


def abstract(marker):
    res = RESULT.format(*TRIALS[marker][:3], *TRIALS[marker][3:]) if marker in TRIALS else ""
    return f"{QUOTE}; depressive symptoms were measured with the GDS-15. {res}"


def record(source, i, marker, doi):
    return Record(id=f"{source}:{i}", source=source, source_id=str(i), doi=doi,
                  title=f"{marker} trial of walking", abstract=abstract(marker),
                  authors=[f"Author{i}, A"], year=2015 + i, venue="Journal")


def fake_search(strategy, http, *, max_records, contact_email=None, env=None):
    pubmed = [record("pubmed", i, m, f"10.1/{i}") for i, m in enumerate(MARKERS, 1)]
    openalex = [record("openalex", 1, MARKERS[0], "10.1/1")]          # duplicate DOI
    return ({"pubmed": SearchResult(source="pubmed", query=strategy["queries"]["pubmed"],
                                    total_hits=5, records=pubmed),
             "openalex": SearchResult(source="openalex", query=strategy["queries"]["openalex"],
                                      total_hits=3, records=openalex)}, {})


def fake_fetch(papers, http, *, contact_email=None, env=None):
    out = {}
    for p in papers:
        if "AGREE_IN_A" in p["title"]:
            out[p["paper_id"]] = FullText(paper_id=p["paper_id"], status="pmc_xml", sections=[
                Section(title="Methods", text="Participants aged 60 years or older were "
                        "randomized by computer to walking or usual care. " * 4)])
        else:
            out[p["paper_id"]] = FullText(paper_id=p["paper_id"], status="abstract_only",
                                          reason="no open-access full text")
    return out


def criteria_ids(prompt):
    return list(dict.fromkeys(re.findall(r"^\s*-?\s*\[?(D|I\d+|E\d+)\]?[:.) ]", prompt, re.M)))


def verdicts(prompt, decision, criterion=None, quote=QUOTE):
    out = []
    for cid in criteria_ids(prompt) or ["D", "I1", "I2", "E1"]:
        ok = "no" if cid.startswith("E") else "yes"
        bad = "yes" if cid.startswith("E") else "no"
        met = bad if decision == "exclude" and cid == criterion else ok
        out.append({"criterion": cid, "met": met, "quote": QUOTE if met == "yes" else quote
                    if met == bad else ""})
    return {"verdicts": out, "reason": "r", "unresolved": decision == "unresolved"}


def screener(style):
    def answer(prompt):
        if "AGREE_OUT" in prompt:
            return verdicts(prompt, "exclude", "E1", quote="walking")
        if "SPLIT_UNRES" in prompt and style == "a":
            return verdicts(prompt, "exclude", "I2", quote="usual care")
        return verdicts(prompt, "include")
    return answer


def extractor(prompt):
    marker = next(m for m in TRIALS if m in prompt)
    mi, si, ni, mc, sc, nc = TRIALS[marker]
    quote = RESULT.format(mi, si, ni, mc, sc, nc)
    return {"fields": [{"key": "design", "quote": "randomized", "chunk_id": "",
                        "value": "randomized controlled trial"},
                       {"key": "O1_measure", "quote": "the GDS-15", "chunk_id": "",
                        "value": "GDS-15"}],
            "design_category": "randomized trial",
            "results": [{"outcome": "O1", "quote": quote, "chunk_id": "", "timepoint": "12 weeks",
                         "arms": [{"arm": "exercise", "n": ni, "mean": mi, "sd": si},
                                  {"arm": "usual care", "n": nc, "mean": mc, "sd": sc}]}]}


def synthesis(prompt):
    labels = sorted(set(re.findall(r"^(S\d+) \|", prompt, re.M)))
    return {"sentences": [
        {"section": "O1", "text": "Studies were compared by direction of effect.",
         "studies": labels},
        {"section": "overall", "text": "Studies were extracted from abstracts and full text.",
         "studies": labels}],
        "heterogeneity": [{"outcome": "O1", "studies": labels,
                           "text": "I2 and tau2 are given in the tables; doses varied."}]}


def coordinator(prompt):
    done = re.search(r"^Stages run \(in order\): (.*)$", prompt, re.M).group(1).split(", ")
    return {"next": next((s for s in STAGES if s not in done), "report"), "rationale": "next"}


# Part 11f: the writer's answer (no numbers, so every section passes the code checks).
WRITER = {"abstract_background": "Depressive symptoms are common in older adults.",
          "abstract_results": "The included trials compared exercise with usual care.",
          "abstract_conclusion": "Exercise may reduce depressive symptoms; the evidence is "
                                 "limited.",
          "introduction": ["Depression in later life is common.",
                           "This review asks whether exercise reduces depressive symptoms."],
          "discussion": ["The trials point in one direction.", "Most were small.",
                         "Larger trials are needed."],
          "conclusion": "Exercise may help; a reviewer should check the trials."}


def e2e_llm():
    return ScriptedLLM({"coordinator": coordinator, "planner": lambda p: DRAFT,
                     "search_strategy": lambda p: PLAN,
                     "screener_a": screener("a"), "screener_b": screener("b"),
                     "adjudicator": lambda p: verdicts(p, "unresolved"),
                     "extraction": extractor, "risk_of_bias": lambda p: {"domains": []},
                     "synthesis": synthesis, "critic": lambda p: {"issues": []},
                     "writer": lambda p: WRITER})


def e2e_config(tmp_path):
    return Config(project_root=tmp_path, protocol_template=ROOT / "protocols" / "template.yaml",
                  ft_rerank=False, screening_workers=2)


INJECT = {"search": fake_search, "fetch": fake_fetch, "embedder": FakeEmbedder(), "http": None}

REPORT_FILES = {"prisma_flow.png", "manuscript.md", "manuscript.docx", "references.bib",
                "writer.json",
                "run_summary.json", "prisma_counts.json", "protocol.json",
                *(f"tables/{t}.csv" for t in ("studies", "extraction", "results",
                                              "risk_of_bias", "screening"))}


class Answers:
    """Scripted human at the terminal; ``quit_at`` raises at that prompt (user leaves)."""

    def __init__(self, answers, quit_at=None):
        self.answers, self.quit_at, self.prompts = list(answers), quit_at, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.quit_at and self.quit_at in prompt:
            raise KeyboardInterrupt
        return self.answers.pop(0)


def check_outputs(cfg, run_id, state):
    run_dir = cfg.runs_dir / run_id
    files = state["report"]["files"]
    assert set(files) == REPORT_FILES
    assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in files.values())
    for f in ("run.json", "checkpoints.sqlite", "audit.jsonl", "state.json"):
        assert (run_dir / f).exists()
    plots = list((run_dir / "figures").glob("forest_O1_*.png"))
    assert plots and state["synthesis"]["outcomes"][0]["method"] == "meta-analysis"
    md = Path(files["manuscript.md"]).read_text()
    for head in ("## Methods", "### Information sources", "## Results", "## Limitations",
                 "## Quality assurance", "## Run summary", "## Included studies"):
        assert head in md
    assert "95% prediction interval" in md and "Eligibility never depends on results" in md
    assert "![PRISMA 2020 flow diagram](prisma_flow.png)" in md
    assert "](../figures/forest_O1_" in md
    assert state["report"]["prisma_consistent"]
    assert "@article{author2016," in Path(files["references.bib"]).read_text()
    rows = Path(files["tables/screening.csv"]).read_text().splitlines()
    assert len(rows) == 1 + 5                          # header + 5 unique records
    summary = json.loads(Path(files["run_summary.json"]).read_text())
    assert set(summary["llm"]["per_node"]) >= {"coordinator", "planner", "search_strategy",
                                               "ta_screening", "ft_screening", "extraction",
                                               "synthesis", "critic"}
    audit = [json.loads(x) for x in (run_dir / "audit.jsonl").read_text().splitlines()]
    assert audit[-1]["node"] == "report"
    # Part 11b: written live, node by node, yet the file is exactly the state's log, in order
    # (no line lost or written twice across pauses, quits and crash-resumes).
    on_disk = [{k: v for k, v in e.items() if k not in ("run_id", "written_at")} for e in audit
               if e.get("event") != "report re-rendered (--report-only)"]   # file-only line
    assert on_disk == json.loads(json.dumps(state["audit_log"], default=str))
    assert all("started_at" in e and "duration_s" in e for e in on_disk)
    assert not [e for e in audit if e.get("outcome") == "llm_error"]


def test_full_review_end_to_end_with_human_answers(tmp_path):
    cfg = e2e_config(tmp_path)
    llm = e2e_llm()
    ask = Answers(["s", "e"])   # accept the suggested RCT design; exclude the conflict paper
    run_id, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=False, ask=ask,
                            out=lambda *a: None, llm=llm, **INJECT)
    assert state["completed_stages"] == list(STAGES)
    assert state["protocol"]["study_designs"] == ["randomized controlled trial"]
    assert len(ask.prompts) == 2
    ta = state["prisma_counts"]["title_abstract"]
    assert (ta["records_screened"], ta["included"], ta["records_excluded"]) == (5, 3, 2)
    assert state["prisma_counts"]["identification"]["duplicates_removed"] == 1
    assert state["prisma_counts"]["full_text"]["studies_included"] == 3
    assert state["qa_report"]["passed"]
    check_outputs(cfg, run_id, state)


def test_auto_run_never_pauses_and_resume_of_a_finished_run(tmp_path):
    cfg = e2e_config(tmp_path)
    ask = Answers([])
    run_id, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True, ask=ask,
                            out=lambda *a: None, llm=e2e_llm(), **INJECT)
    assert ask.prompts == [] and state["completed_stages"][-1] == "report"
    # --auto never applies the design suggestion; the unresolved paper is auto-included.
    assert state["protocol"]["study_designs"] == []
    assert state["prisma_counts"]["title_abstract"]["included"] == 4
    # Finished run: only the final checkpoint is kept, and resuming still sees it.
    con = sqlite3.connect(cfg.runs_dir / run_id / "checkpoints.sqlite")
    assert con.execute("select count(*) from checkpoints").fetchone() == (1,)
    con.close()
    said = []
    _, again = cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id,
                       ask=ask, out=said.append, llm=e2e_llm(), **INJECT)
    assert "already finished" in said[0] and again["report"] == state["report"]


def test_quit_at_a_pause_then_resume_from_the_sqlite_checkpoint(tmp_path):
    cfg = e2e_config(tmp_path)
    llm = e2e_llm()
    with pytest.raises(KeyboardInterrupt):
        cli.run(cfg, question=QUESTION, topic_id=None, auto=False,
                ask=Answers([""], quit_at="[i]nclude"), out=lambda *a: None, llm=llm, **INJECT)
    run_id = next(p.name for p in cfg.runs_dir.iterdir())
    screen_calls = len([a for a, _ in llm.calls if a.startswith("screener")])
    # New process: new graph, new connection to the same checkpoint file, same LLM cache.
    ask = Answers(["i", "i"])
    run_id2, state = cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id,
                             ask=ask, out=lambda *a: None, llm=llm, **INJECT)
    assert run_id2 == run_id and state["completed_stages"][-1] == "report"
    assert state["prisma_counts"]["title_abstract"]["included"] == 4      # human included it
    # Included by the human, the paper reaches full text, where the screeners split again.
    assert len(ask.prompts) == 2 and state["prisma_counts"]["full_text"]["studies_included"] == 4
    # Resuming re-ran only the conflict node: no screener prompt was ever sent twice.
    prompts = [p for a, p in llm.calls if a.startswith("screener")]
    assert len(prompts) == len(set(prompts)) and len(prompts) > screen_calls
    check_outputs(cfg, run_id, state)


def test_main_argument_errors(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
    assert "give a question, --topic or --resume" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--resume", "x", "--auto"])
    with pytest.raises(SystemExit):
        cli.main(["other question", "--topic", "D0"])
    with pytest.raises(SystemExit):
        cli.main(["--topic", "T9"])


def test_crash_in_a_stage_then_resume_continues_from_the_last_checkpoint(tmp_path):
    cfg = e2e_config(tmp_path)
    crash = {"left": 1}

    def flaky_extractor(prompt):          # the process dies once, in the middle of extraction
        if crash["left"]:
            crash["left"] -= 1
            raise KeyboardInterrupt
        return extractor(prompt)

    llm = e2e_llm()
    llm.script["extraction"] = flaky_extractor
    cfg = cfg.model_copy(update={"screening_workers": 1})
    with pytest.raises(KeyboardInterrupt):
        cli.run(cfg, question=QUESTION, topic_id=None, auto=True, ask=Answers([]),
                out=lambda *a: None, llm=llm, **INJECT)
    run_id = next(p.name for p in cfg.runs_dir.iterdir())
    before = [a for a, _ in llm.calls]
    assert "synthesis" not in before
    said = []
    _, state = cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id,
                       ask=Answers([]), out=said.append, llm=llm, **INJECT)
    assert "Resuming run" in said[0] and "extraction" in said[0]
    assert state["completed_stages"] == list(STAGES)
    after = [a for a, _ in llm.calls][len(before):]
    assert not [a for a in after if a.startswith(("screener", "adjudicator", "planner"))]
    check_outputs(cfg, run_id, state)
