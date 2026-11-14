"""Guided flow (Part 11e): the logic behind the Home page and the step-by-step review screens.

No Streamlit here (tested directly). The screens are in ``guide.py``.

Two modes share the same five steps (question, approve plan, AI at work, results, check):
- **demo**: a replay of a finished review from a small tracked snapshot (``docs/demo/<run>``:
  run.json, audit.jsonl, report/, demo.json). No API call, nothing written outside a scratch
  folder. The replay cuts the audit log after each stage, so the screens show the same counts
  the real run showed, in the same order.
- **real**: a live run through ``ui.live`` (same runner as the Live run page and the CLI). The
  step shown is derived from the run's status, never kept apart from it.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from ..evaluation import stage_checks as SC
from ..guardrail import STAGES

ROOT = SC.CHECKS_DIR.parents[1]                 # project root (evaluation/stage_checks -> ..)
DEMO_DIR = ROOT / "docs" / "demo"

STEPS = ["Question", "Approve the plan", "AI at work", "Results", "Check accuracy",
         "Your review article"]

PLAIN = {   # stage -> what it does, in plain words
    "planner": "Write the plan (which papers count)",
    "search_strategy": "Prepare the search words",
    "retrieval": "Find papers in the databases",
    "ta_screening": "Read titles and summaries",
    "ft_screening": "Read the full papers",
    "extraction": "Copy the key facts from each paper",
    "synthesis": "Write the overall summary",
    "critic": "Double-check the summary",
    "report": "Write the final report",
}

# The checks offered in the guided flow, in order, with a plain question.
CHECKS = [
    ("ta", "Did the AI keep or throw out the right papers, from the title and summary?"),
    ("ft", "Did the AI keep or throw out the right papers, after reading them?"),
    ("extraction", "Is the fact the AI copied from the paper correct?"),
]

DEMO_FILES = ["run.json", "audit.jsonl"]
REPORT_FILES = ["prisma_flow.png", "manuscript.md", "manuscript.docx", "references.bib",
                "protocol.json", "prisma_counts.json", "run_summary.json", "evidence_map.png",
                "writer.json", "findings.json", "prisma_checklist.json"]


# --- Demo snapshot ----------------------------------------------------------------------

def demo_run_dir(base: Path | None = None) -> Path | None:
    """The demo snapshot folder (the first one under docs/demo), or None when missing."""
    base = base or DEMO_DIR
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir() if (p / "audit.jsonl").exists()
                  and (p / "demo.json").exists())
    return runs[0] if runs else None


def build_demo(run_dir: Path, out: Path | None = None) -> Path:
    """Copy what the demo needs from a finished run into docs/demo/<run_id> (tracked):
    run.json, audit.jsonl, the report files and demo.json (results + a practice disagreement,
    taken from state.json, which is too big to track)."""
    run_dir = Path(run_dir)
    out = Path(out or DEMO_DIR / run_dir.name)
    if out.exists():
        shutil.rmtree(out)
    (out / "report" / "tables").mkdir(parents=True)
    for name in DEMO_FILES:
        _copy(run_dir / name, out / name)
    # The replay shows the run as it happened: report re-renders (--report-only, added to
    # the audit later) are not part of it.
    log = out / "audit.jsonl"
    kept = [x for x in log.read_text().splitlines()
            if x.strip() and "--report-only" not in json.loads(x).get("event", "")]
    log.write_text("\n".join(kept) + "\n")
    for name in REPORT_FILES:
        if (run_dir / "report" / name).exists():
            _copy(run_dir / "report" / name, out / "report" / name)
    for f in (run_dir / "report" / "tables").glob("*.csv"):
        _copy(f, out / "report" / "tables" / f.name)
    state = json.loads((run_dir / "state.json").read_text())
    demo = {"results": results_from_state(state, run_dir),
            "protocol": state.get("protocol") or {},
            "practice": practice_conflict(state)}
    (out / "demo.json").write_text(scrub(json.dumps(demo, indent=1, ensure_ascii=False)))
    return out


def scrub(text: str) -> str:
    """Local paths out of a tracked file: the project root becomes relative, the home
    folder becomes ~ (the audit's report entry stores the absolute report folder)."""
    return text.replace(str(ROOT) + "/", "").replace(str(Path.home()), "~")


def _copy(src: Path, dst: Path) -> None:
    if src.suffix in (".json", ".jsonl", ".md", ".csv", ".bib"):
        dst.write_text(scrub(src.read_text()))
    else:
        shutil.copy2(src, dst)


def practice_conflict(state: dict) -> dict | None:
    """One real title/abstract record where the two screeners disagreed and the adjudicator
    decided: shown in the demo as a practice question (the answer changes nothing)."""
    papers = {p["paper_id"]: p for p in state.get("unique_records") or []}
    from ..state import latest
    for d in latest(state.get("ta_decisions") or []):
        a, b, adj = d.get("a") or {}, d.get("b") or {}, d.get("adjudicator") or {}
        if (d.get("resolved_by") == "adjudicator" and a.get("decision") and b.get("decision")
                and a.get("decision") != b.get("decision") and adj.get("decision")):
            p = papers.get(d["paper_id"]) or {}
            return {"title": d.get("title") or p.get("title"),
                    "abstract": SC._cut(p.get("abstract"), 1200),
                    "a": _vote(a), "b": _vote(b), "judge": _vote(adj),
                    "final": d.get("final")}
    return None


def _vote(v: dict) -> dict:
    quote = v.get("quote") or next((x.get("quote") for x in v.get("verdicts") or []
                                    if x.get("criterion") == v.get("criterion")
                                    and x.get("quote")), "")
    return {"decision": v.get("decision"), "criterion": v.get("criterion"),
            "quote": SC._cut(quote, 400)}


# --- Results ----------------------------------------------------------------------------

def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def results_from_state(state: dict, run_dir: Path) -> dict:
    """What the Results screen shows: the funnel in plain numbers, time and cost, and the
    papers kept with how they were judged and the quote behind it."""
    counts = _json(run_dir / "report" / "prisma_counts.json") or state.get("prisma_counts") or {}
    summary = _json(run_dir / "report" / "run_summary.json")
    ident, ta, ft = (counts.get("identification") or {}, counts.get("title_abstract") or {},
                     counts.get("full_text") or {})
    kept = []
    for r in SC.stage_items(state, "ft"):
        if r["decision"] != "include":
            continue
        how = ("summary only (no free full text)" if "abstract_only" in (r.get("detail") or "")
               else "full paper")
        kept.append({"title": r.get("title"), "how": how, "why": SC._cut(r.get("quote"), 600),
                     "link": r.get("link")})
    llm = (summary.get("llm") or {}).get("cost_usd") or 0.0
    emb = (summary.get("embeddings") or {}).get("cost_usd") or 0.0
    return {
        "question": state.get("question") or summary.get("question"),
        "found": ident.get("identified"), "unique": ident.get("records_screened"),
        "screened": ta.get("records_screened"), "not_screened": ta.get("not_screened"),
        "by_rule": (ta.get("title_rule") or 0) + (ta.get("arxiv_rule") or 0),
        "read": (ta.get("records_screened") or 0) - (ta.get("title_rule") or 0)
                - (ta.get("arxiv_rule") or 0),
        "ta_kept": ta.get("included"), "sought": ft.get("reports_sought"),
        "full_text_read": ft.get("reports_assessed"),
        "abstract_only": ft.get("screened_on_abstract_only"),
        "included": ft.get("studies_included"),
        "minutes": round((summary.get("seconds") or 0) / 60, 1),
        "cost_usd": round(llm + emb, 4),
        "kept": kept,
    }


def results(run_dir: Path) -> dict | None:
    """Results of a finished run: from demo.json for the demo snapshot, else from state.json."""
    demo = _json(run_dir / "demo.json")
    if demo.get("results"):
        return demo["results"]
    if not (run_dir / "state.json").exists():
        return None
    return results_from_state(json.loads((run_dir / "state.json").read_text()), run_dir)


def findings(run_dir: Path) -> dict | None:
    """Summary of findings and evidence map of a run (report/findings.json), if any."""
    data = _json(run_dir / "report" / "findings.json")
    return data or None


def report_files(run_dir: Path) -> list[Path]:
    """Report files a person would download, most useful first."""
    rep = run_dir / "report"
    order = ["manuscript.docx", "manuscript.md", "prisma_flow.png", "evidence_map.png",
             "references.bib"]
    files = [rep / n for n in order if (rep / n).exists()]
    return files + sorted((rep / "tables").glob("*.csv"))


# --- Replay -----------------------------------------------------------------------------

def replay_frames(audit: list[dict]) -> list[tuple[int, str | None]]:
    """Frames of a replay: (number of audit entries shown, stage running). Frame j shows the
    log up to the (j+1)-th stage entry, with that stage running; the last frame shows it all.
    Coordinator decisions (refusals included) and conflict entries stay with the stage that
    follows them, so the done count never goes back."""
    marks = [i for i, e in enumerate(audit) if e.get("node") in STAGES and e.get("event")]
    frames = [(i, audit[i]["node"]) for i in marks]
    frames.append((len(audit), None))
    return frames


def pause_frame(audit: list[dict]) -> int | None:
    """The frame at which the demo stops for the practice disagreement: right after
    title/abstract screening is done."""
    frames = replay_frames(audit)
    done_ta = [j for j, (cut, _) in enumerate(frames)
               if any(e.get("node") == "ta_screening" for e in audit[:cut])]
    return done_ta[0] if done_ta else None


# --- Plain-language progress ------------------------------------------------------------

def _last(audit: list[dict], node: str) -> dict:
    return next((e for e in reversed(audit) if e.get("node") == node and e.get("event")), {})


def _n(x) -> str:
    return "–" if x is None else f"{x:,}"


def step_lines(audit: list[dict], running: str | None, done: set[str]) -> list[dict]:
    """One line per stage: status (done / running / waiting) and a plain one-line result."""
    out = []
    for s in STAGES:
        e = _last(audit, s)
        status = "running" if s == running else "done" if s in done else "waiting"
        text = ""
        if status == "done":
            if s == "planner":
                text = "Plan ready"
            elif s == "search_strategy":
                text = f"Search words ready for {len(e.get('queries') or {})} databases"
            elif s == "retrieval":
                text = (f"{_n(e.get('identified'))} papers found · "
                        f"{_n(e.get('records_screened'))} after removing duplicates")
            elif s == "ta_screening":
                rule = (e.get("title_rule") or 0) + (e.get("arxiv_rule") or 0)
                read = (e.get("records_screened") or 0) - rule
                text = (f"{_n(read)} read" + (f" ({rule} removed first by a rule)" if rule
                                               else "") + f" · {_n(e.get('included'))} kept")
                if e.get("unresolved"):
                    text += f" · {e['unresolved']} undecided (settled next)"
            elif s == "ft_screening":
                text = (f"{_n(e.get('reports_sought'))} read · {_n(e.get('studies_included'))} kept "
                        f"({_n(e.get('reports_assessed'))} full papers, "
                        f"{_n(e.get('screened_on_abstract_only'))} summary only)")
            elif s == "extraction":
                text = f"Facts taken from {_n(e.get('studies'))} papers"
            elif s == "synthesis":
                text = f"Summary written ({_n(e.get('sentences'))} sentences)"
            elif s == "critic":
                text = ("Passed" if e.get("passed") else
                        f"{_n(e.get('blockers'))} problems found, summary sent back")
            elif s == "report":
                text = f"{len(e.get('files') or [])} files written"
        elif status == "running":
            text = "Working on it…"
        out.append({"stage": s, "label": PLAIN[s], "status": status, "text": text})
    return out


def done_stages(audit: list[dict]) -> set[str]:
    return {e["node"] for e in audit if e.get("node") in STAGES and e.get("event")}


# --- Real run: which step -----------------------------------------------------------------

def real_step(status: str, pending: dict | None, audit: list[dict], auto: bool) -> int:
    """Step (1-based) a live run is at, from its status alone. Finished -> 4 (results);
    waiting for the plan, or still writing it -> 2; anything else (running, waiting for a
    disagreement or the budget, stopped, error) -> 3, where the screen says what to do."""
    if status == "finished":
        return 4
    kind = (pending or {}).get("kind")
    if status == "waiting" and kind == "protocol_approval":
        return 2
    approved = auto or any(e.get("node") == "protocol_approval" for e in audit)
    if status in ("starting", "running") and not approved:
        return 2
    return 3


# --- Checks in the demo: a practice copy --------------------------------------------------

def practice_checks(run_id: str, dest: Path, source: Path | None = None) -> Path:
    """Copy the run's tracked check samples into ``dest`` with the marks and notes cleared,
    so the demo can be marked without touching evaluation/stage_checks. Existing practice
    marks are kept (a page reload does not wipe them)."""
    source = Path(source or SC.CHECKS_DIR) / run_id
    target = Path(dest) / run_id
    target.mkdir(parents=True, exist_ok=True)
    for stage, _ in CHECKS:
        src_csv, src_json = source / f"{stage}.csv", source / f"{stage}.json"
        if not src_csv.exists() or (target / f"{stage}.csv").exists():
            continue
        with src_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["mark"], r["note"] = "", ""
        with (target / f"{stage}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SC.COLS)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in SC.COLS} for r in rows)
        if src_json.exists():
            shutil.copy2(src_json, target / f"{stage}.json")
    return Path(dest)


def reset_practice(run_id: str, dest: Path) -> None:
    """Wipe the demo's practice marks, so each demo starts unmarked."""
    shutil.rmtree(Path(dest) / run_id, ignore_errors=True)


def default_cap(est: dict | None, fallback: dict | None = None) -> float | None:
    """Default budget cap: 1.5 x the high estimate of the chosen size (at least $1); for a
    size without an estimate (everything found), the largest estimate known; else none."""
    ref = est or fallback
    return float(max(1.0, round(ref["high"] * 1.5, 1))) if ref else None


def unfinished_ui_runs(runs_dir: Path) -> list[Path]:
    """Runs started from the app that have no report yet, newest first: Home offers to
    continue them even after a reload or a closed tab."""
    out = []
    for d in sorted(Path(runs_dir).glob("*"), reverse=True):
        meta = _json(d / "run.json")
        if (meta.get("started_from") == "ui"
                and not (d / "report" / "run_summary.json").exists()):
            out.append(d)
    return out


def article_parts(text: str) -> list[tuple[str, str]]:
    """The manuscript split around its images: [("md", text), ("img", file name), ...], so
    the page can show the PRISMA diagram from the report folder (Markdown alone cannot)."""
    import re
    parts, pos = [], 0
    for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
        parts.append(("md", text[pos:m.start()]))
        parts.append(("img", m.group(1)))
        pos = m.end()
    parts.append(("md", text[pos:]))
    return [(k, v) for k, v in parts if v.strip()]


def next_unmarked(rows: list[dict]) -> int:
    """Index of the first item without a mark (len(rows) when all are marked)."""
    return next((i for i, r in enumerate(rows) if not SC.normalise_mark(r.get("mark"))),
                len(rows))
