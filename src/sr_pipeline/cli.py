"""Command line: run a full systematic review.

    sr-review "Does exercise reduce depressive symptoms in older adults?"
    sr-review --topic T3                    # evaluation topic: question + gold review criteria
    sr-review "question" --auto             # never pause (protocol auto-approved, conflicts
                                            # auto-included)
    sr-review --resume 20261108-101500-a1b2c3   # continue a stopped or paused run
    sr-review "question" --config my.yaml
    sr-review --topic T3 --limit 100        # screen only the 100 best-ranked records
                                            # (default: config screen_limit = 100; 0 = all)
    sr-review --report-only 20261108-101500-a1b2c3   # rebuild report/ from the saved state
                                                     # (code only, $0, no API key needed)

Every run lives in ``runs/<run_id>/``: ``run.json`` (the arguments, so ``--resume`` rebuilds
the same graph), ``checkpoints.sqlite`` (LangGraph state after every step: a crash or a
Ctrl-C loses at most the step in progress), ``audit.jsonl``, ``index/``, ``figures/`` and
``report/`` (manuscript, PRISMA diagram, tables, BibTeX, run summary), ``state.json``.

Without ``--auto`` the run pauses twice at most: protocol approval (Enter = approve, ``s`` =
accept the suggested study designs, or a JSON answer such as
``{"edits": {"years": {"start": 2010}}}``), and unresolved screening conflicts (one answer
per paper). The answers are recorded in the audit log.
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .audit import AuditWriter, new_run_id
from .config import Config, load_config
from .evaluation.topics import load_topic
from .graph import run_config
from .pipeline import agent_models, build_review_graph
from .reporting import make_report

Ask = Callable[[str], str]


# --- Human answers ------------------------------------------------------------------------

def _protocol_text(payload: dict) -> str:
    p = payload["protocol"]
    pico = p.get("pico") or {}
    crit = ([f"  I{i} {c}" for i, c in enumerate(p.get("inclusion_criteria") or [], 1)]
            + [f"  E{i} {c}" for i, c in enumerate(p.get("exclusion_criteria") or [], 1)])
    lines = [
        "=" * 70, "PROTOCOL APPROVAL", "=" * 70,
        f"Question:     {p.get('question')}",
        f"Population:   {pico.get('population')}",
        f"Intervention: {pico.get('intervention')}",
        f"Comparator:   {pico.get('comparator')}",
        f"Outcomes:     {'; '.join(pico.get('outcomes') or [])}",
        f"Designs:      {', '.join(p.get('study_designs') or []) or 'no restriction'}",
        f"Databases:    {', '.join(p.get('databases') or [])}",
        f"Years:        {p.get('years')}   Languages: {p.get('languages')}",
        f"Criteria from: {payload.get('criteria_source') or 'Planner (LLM draft)'}",
        "Criteria:", *crit,
    ]
    if payload.get("suggested_designs"):
        lines.append(f"Suggested designs (not applied): {payload['suggested_designs']} "
                     f"({payload.get('design_reason')})")
    return "\n".join(lines)


def protocol_answer(payload: dict, ask: Ask, out=print):
    if payload.get("error"):
        out(f"\nPrevious answer refused: {payload['error']}")
    out(_protocol_text(payload))
    while True:
        raw = ask("Enter = approve, s = accept suggested designs, or a JSON answer: ").strip()
        if raw.lower() in ("", "y", "yes"):
            return True
        if raw.lower() == "s":
            return {"accept_suggested_designs": True}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            out(f"Not JSON ({exc}); try again.")


def conflicts_answer(payload: dict, ask: Ask, out=print) -> dict:
    if payload.get("error"):
        out(f"\nPrevious answer refused: {payload['error']}")
    papers = payload["papers"]
    out(f"\n{'=' * 70}\nUNRESOLVED SCREENING CONFLICTS ({payload['stage']}, "
        f"{len(papers)} papers)\n{'=' * 70}")
    answer = {}
    for p in papers:
        a, b = p.get("screener_a") or {}, p.get("screener_b") or {}
        adj = p.get("adjudicator") or {}
        out(f"\n{p['paper_id']}: {p['title']}\n  A: {a.get('decision')} "
            f"({a.get('criterion')})  B: {b.get('decision')} ({b.get('criterion')})  "
            f"adjudicator: {adj.get('decision') or adj.get('reason', 'n/a')}")
        while True:
            raw = ask("  [i]nclude / [e]xclude (Enter = include): ").strip().lower()
            if raw in ("", "i", "include"):
                answer[p["paper_id"]] = "include"
                break
            if raw in ("e", "exclude"):
                answer[p["paper_id"]] = "exclude"
                break
    return answer


def answer_interrupt(value: dict, ask: Ask, out=print):
    if value.get("kind") == "protocol_approval":
        return protocol_answer(value, ask, out)
    if value.get("kind") == "screening_conflicts":
        return conflicts_answer(value, ask, out)
    raise ValueError(f"unknown interrupt: {value.get('kind')!r}")


# --- Running ----------------------------------------------------------------------------

def open_checkpointer(run_dir: Path) -> SqliteSaver:
    # Own connection (not ``from_conn_string``'s context manager): stages run threads.
    return SqliteSaver(sqlite3.connect(run_dir / "checkpoints.sqlite",
                                       check_same_thread=False))


def compact_checkpoints(run_dir: Path, checkpoint_id: str) -> None:
    """Keep only the final checkpoint of a finished run. SqliteSaver stores the whole state
    at every step (a real run: tens of steps x a state of many MB); once the report is
    written only the last one is needed (``--resume`` then says the run is finished)."""
    con = sqlite3.connect(run_dir / "checkpoints.sqlite")
    with con:
        con.execute("DELETE FROM checkpoints WHERE checkpoint_id != ?", (checkpoint_id,))
        con.execute("DELETE FROM writes WHERE checkpoint_id != ?", (checkpoint_id,))
    con.execute("VACUUM")
    con.close()


def drive(graph, config: dict, first, audit: AuditWriter, ask: Ask, out=print) -> dict:
    """Invoke, answer every interrupt, sync the audit after each pause. Returns the state."""
    result = graph.invoke(first, config)
    while True:
        audit.sync(graph.get_state(config).values.get("audit_log", []))
        pending = result.get("__interrupt__") if isinstance(result, dict) else None
        if not pending:
            return graph.get_state(config).values
        answer = answer_interrupt(pending[0].value, ask, out)
        result = graph.invoke(Command(resume=answer), config)


def run(cfg: Config, *, question: str | None, topic_id: str | None, auto: bool,
        resume: str | None = None, screen_limit: int | None = None, ask: Ask = input,
        out=print, llm=None, **inject) -> tuple[str, dict]:
    """New run or ``resume``; returns (run_id, final state). ``screen_limit``: records
    screened at title/abstract (None = config; 0 = all), stored in run.json so a resume
    uses the same. ``llm`` / ``inject`` are for tests (scripted LLM, fake search / full
    text / embedder)."""
    if resume:
        run_id = resume
        run_dir = cfg.runs_dir / run_id
        meta = json.loads((run_dir / "run.json").read_text())
    else:
        run_id = new_run_id()
        run_dir = cfg.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": run_id, "question": question, "topic": topic_id, "auto": auto,
                "screen_limit": cfg.screen_limit if screen_limit is None else screen_limit,
                "models": agent_models(cfg),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    cfg = cfg.model_copy(update={"screen_limit": meta.get("screen_limit", cfg.screen_limit)})
    topic = load_topic(meta["topic"]) if meta.get("topic") else None
    question = topic["question"] if topic else meta["question"]

    if llm is None:
        llm = cfg.make_llm()
        inject = {"http": cfg.make_http(), "embedder": cfg.make_embedder(), **inject}
    graph = build_review_graph(cfg, llm, run_dir, auto=meta["auto"], topic=topic,
                               checkpointer=open_checkpointer(run_dir), **inject)
    config = run_config(run_id)
    audit = AuditWriter(cfg.runs_dir, run_id)

    if resume:
        snap = graph.get_state(config)
        if not snap.next:
            out(f"Run {run_id} has already finished.")
            return run_id, snap.values
        if snap.interrupts:   # paused for a human: ask again
            first = Command(resume=answer_interrupt(snap.interrupts[0].value, ask, out))
        else:                 # stopped mid-run (crash, Ctrl-C): continue from the checkpoint
            first = None
        out(f"Resuming run {run_id} at {', '.join(snap.next)}")
    else:
        first = {"question": question,
                 "run": {"run_id": run_id, "started_at": meta["started_at"]}}
        out(f"Run {run_id}: {question}\nOutputs in {run_dir}")
    state = drive(graph, config, first, audit, ask, out)
    (run_dir / "state.json").write_text(json.dumps(state, indent=2, default=str))
    snap = graph.get_state(config)
    if state.get("report") and not snap.next:
        compact_checkpoints(run_dir, snap.config["configurable"]["checkpoint_id"])
    return run_id, state


def report_only(cfg: Config, run_id: str, out=print) -> dict:
    """Part 9b: rebuild ``runs/<id>/report/`` from the run's saved ``state.json`` (code only,
    no LLM call, $0), e.g. after a fix in the reporting code. The run's facts are kept:
    models from run.json, finish time and embedding cost from the previous run summary.
    Appends one audit entry. Returns the report's file map."""
    run_dir = cfg.runs_dir / run_id
    state_file = run_dir / "state.json"
    if not state_file.exists():
        raise FileNotFoundError(f"{run_dir} has no state.json: the run has not finished "
                                f"(continue it with --resume {run_id})")
    state = json.loads(state_file.read_text())
    if not state.get("report"):
        raise ValueError(f"run {run_id} ended before the report stage: nothing to re-render")
    meta_file, prev_file = run_dir / "run.json", run_dir / "report" / "run_summary.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    prev = json.loads(prev_file.read_text()) if prev_file.exists() else None
    update = make_report(run_dir / "report", models=meta.get("models") or agent_models(cfg),
                         previous_summary=prev)(state)
    rep = update["report"]
    entry = {"node": "report", "event": "report re-rendered (--report-only)",
             "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "files": sorted(rep["files"]), "prisma_consistent": rep["prisma_consistent"]}
    with (run_dir / "audit.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    out(f"Report of run {run_id} rebuilt in {rep['folder']} (PRISMA counts "
        f"{'add up' if rep['prisma_consistent'] else 'do NOT add up: see the Results'})")
    return rep["files"]


def _final_message(run_id: str, state: dict, out=print) -> None:
    rep = state.get("report")
    if not rep:
        out(f"\nRun {run_id} ended without a report (stages run: "
            f"{', '.join(state.get('completed_stages', []))}). See audit.jsonl.")
        return
    summary = json.loads(Path(rep["files"]["run_summary.json"]).read_text())
    qa = state.get("qa_report") or {}
    pc = (state.get("prisma_counts") or {}).get("full_text") or {}
    out(f"\nRun {run_id} done: {pc.get('studies_included', 0)} studies included; critic "
        f"{'passed' if qa.get('passed') else 'not passed'} ({qa.get('blockers', 0)} "
        f"blockers, {qa.get('warnings', 0)} warnings)")
    llm = summary["llm"]
    out(f"LLM calls {llm['calls']} ({llm['cached']} cached), cost ${llm['cost_usd']:.4f}, "
        f"{(summary['seconds'] or 0) / 60:.1f} min")
    out(f"Manuscript: {rep['files']['manuscript.md']}\n            "
        f"{rep['files']['manuscript.docx']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sr-review",
                                 description="Run a PRISMA 2020 systematic review.")
    ap.add_argument("question", nargs="?", help="research question")
    ap.add_argument("--topic", help="evaluation topic of evaluation/topics.yaml (T1, T2, T3)")
    ap.add_argument("--auto", action="store_true",
                    help="never pause: approve the protocol, include unresolved conflicts")
    ap.add_argument("--config", type=Path, help="config YAML (default: config.yaml)")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue a run from its checkpoint")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="screen only the N best-ranked unique records (default: config "
                         "screen_limit, 100; 0 = all); the rest are reported as not screened")
    ap.add_argument("--report-only", metavar="RUN_ID",
                    help="rebuild runs/RUN_ID/report/ from the saved state (no LLM call)")
    args = ap.parse_args(argv)

    if args.report_only:
        if args.question or args.topic or args.auto or args.resume or args.limit is not None:
            ap.error("--report-only takes only a run id (and --config)")
        cfg = load_config(args.config)
        try:
            report_only(cfg, args.report_only)
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    if args.limit is not None and args.limit < 0:
        ap.error("--limit must be 0 (no limit) or more")
    if args.resume and (args.question or args.topic or args.auto or args.limit is not None):
        ap.error("--resume takes the question, topic, --auto and --limit of the original run")
    topic = None
    if args.topic:
        try:
            topic = load_topic(args.topic)
        except KeyError as exc:
            ap.error(str(exc))
        if args.question and args.question != topic["question"]:
            ap.error(f"--topic {args.topic} sets the question: {topic['question']!r}")
    if not (args.resume or args.question or topic):
        ap.error("give a question, --topic or --resume")

    cfg = load_config(args.config)  # also loads .env
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).",
              file=sys.stderr)
        return 1
    if args.resume and not (cfg.runs_dir / args.resume / "run.json").exists():
        print(f"No run {args.resume} in {cfg.runs_dir}", file=sys.stderr)
        return 1
    try:
        run_id, state = run(cfg, question=args.question, topic_id=args.topic,
                            auto=args.auto, resume=args.resume, screen_limit=args.limit)
    except KeyboardInterrupt:
        print("\nStopped. Continue later with: sr-review --resume <run_id> "
              "(the last finished step is checkpointed)", file=sys.stderr)
        return 130
    _final_message(run_id, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
