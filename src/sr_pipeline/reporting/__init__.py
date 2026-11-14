"""Report stage: PRISMA flow diagram, manuscript (Markdown + DOCX), evidence tables (CSV),
BibTeX, run summary, and (Part 11f) the evidence map. Code only, except one optional writer
call (Introduction, Discussion, Conclusions, abstract) checked by code (``writer.py``);
without an LLM those sections are written by code ($0).

Writes ``<out_dir>/`` (``runs/<id>/report/``):
  manuscript.md, manuscript.docx, prisma_flow.png, references.bib, run_summary.json,
  prisma_counts.json, protocol.json, writer.json, tables/{studies,extraction,results,
  risk_of_bias,screening}.csv, evidence_map.png (topics with an evidence map)
Forest plots stay where the synthesis wrote them (``runs/<id>/figures``); the manuscript
links to them.
"""

import json
from pathlib import Path

from ..agents.extraction import included_papers
from ..state import stage_round
from . import findings as F
from . import prisma_checklist as PC
from .manuscript import facts_sheet, manuscript
from .writer import write as write_prose
from .prisma import flow_counts, flow_diagram
from .summary import run_summary
from .tables import bibtex, write_tables
from .word import markdown_to_docx

__all__ = ["make_report", "flow_counts", "run_summary", "collect_findings"]


def collect_findings(state: dict, out_dir: Path, counter=None) -> dict:
    """Study table, evidence map (+ figure) and summary of findings, for topics that have
    their own extraction fields; {} otherwise (the article keeps its PICO table)."""
    p = state.get("protocol") or {}
    sp = F.spec(p)
    fx: dict = {"spec": sp}
    if not p.get("extraction_fields"):
        return fx
    fx["table"] = F.study_table(state, sp)
    em = F.evidence_map(state, sp)
    if em:
        fx["em"] = em
        fx["map_png"] = F.evidence_map_png(em, out_dir / "evidence_map.png")
    sm = F.summary(state, sp, em)
    if sm:
        fx["sof"] = F.summary_lines(sm, p)
        fx["sof_counts"] = sm
    gs = F.gaps(state, sp, em, sm)
    if gs:
        F.check_gaps(gs, counter, years=p.get("years"),
                     few=int(((sp or {}).get("check") or {}).get("few") or 25))
        fx["gaps"] = gs
    return fx


def make_report(out_dir: str | Path, *, models: dict[str, str] | None = None,
                embedder=None, previous_summary: dict | None = None, llm=None,
                counter=None):
    """``models``: agent -> model, for the Methods. ``embedder``: the run's embedder, whose
    cumulative ``usage()`` goes into the run summary (embeddings are not in the audit).
    ``llm``: for the writer (one call); None = the framing sections are written by code.
    ``counter``: OpenAlex hit counter for the gap check (``findings.openalex_counter``);
    None = gaps stay unchecked (said in the article).
    ``previous_summary`` (Part 9b, ``--report-only``): the run summary of the original
    report; its finish time and embedding cost are kept, so a re-render changes no fact."""
    out_dir = Path(out_dir)

    def report(state: dict) -> dict:
        rnd = stage_round("report", state.get("completed_stages", []))
        out_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}

        png = flow_diagram(state.get("prisma_counts") or {}, out_dir / "prisma_flow.png",
                           protocol=state.get("protocol"))
        files["prisma_flow.png"] = str(png)
        for name, path in write_tables(state, out_dir / "tables").items():
            files[f"tables/{name}"] = str(path)
        bib = out_dir / "references.bib"
        bib.write_text(bibtex(included_papers(state)))
        files["references.bib"] = str(bib)
        for name, value in (("prisma_counts.json", state.get("prisma_counts") or {}),
                            ("protocol.json", state.get("protocol") or {})):
            (out_dir / name).write_text(json.dumps(value, indent=2, default=str))
            files[name] = str(out_dir / name)

        # The writer runs before the summary: in a live run its one call is part of the
        # run (same audit entry as below), so the summary written now and any later
        # re-render agree. A re-render's own writer call stays out (writer.json only).
        fx = collect_findings(state, out_dir, counter)
        if fx.get("map_png"):
            files["evidence_map.png"] = str(fx["map_png"])
        if fx.get("sof"):     # for the app's results screen (same content as the article)
            (out_dir / "findings.json").write_text(json.dumps(
                {"summary": fx["sof"], "evidence_map": fx.get("em"),
                 "gaps": fx.get("gaps") or []}, indent=2, default=str))
            files["findings.json"] = str(out_dir / "findings.json")
        facts, facts_d = facts_sheet(state, fx)
        labels = [s["label"] for s in (state.get("synthesis") or {}).get("studies", [])]
        prose = write_prose(llm, state.get("protocol") or {}, facts, facts_d, labels)
        (out_dir / "writer.json").write_text(json.dumps({"facts": facts, **prose}, indent=2,
                                                        default=str))
        files["writer.json"] = str(out_dir / "writer.json")

        usage = embedder.usage() if embedder is not None and hasattr(embedder, "usage") \
            else None
        if previous_summary:
            usage = usage or previous_summary.get("embeddings") or None
        counted = state
        if prose["usage"] and not previous_summary:
            counted = {**state, "audit_log": [*state.get("audit_log", []),
                                              {"node": "report", **prose["usage"]}]}
        summary = run_summary(counted, embeddings=usage)
        if previous_summary:   # the run's own times, copied (not recomputed from rounded ones)
            for key in ("finished_at", "seconds", "stages_run"):
                if key in previous_summary:
                    summary[key] = previous_summary[key]
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        files["run_summary.json"] = str(out_dir / "run_summary.json")

        md = manuscript(state, base=out_dir, models=models or {}, summary=summary,
                        prisma_png=png, fx=fx, writer=prose)
        checklist = PC.check(md, state)       # PRISMA 2020, on the finished article
        md = md.rstrip("\n") + "\n\n" + "\n".join(PC.appendix(checklist)) + "\n"
        (out_dir / "prisma_checklist.json").write_text(json.dumps(
            {"summary": PC.summary(checklist), "items": checklist}, indent=2))
        files["prisma_checklist.json"] = str(out_dir / "prisma_checklist.json")
        (out_dir / "manuscript.md").write_text(md)
        files["manuscript.md"] = str(out_dir / "manuscript.md")
        files["manuscript.docx"] = str(markdown_to_docx(md, out_dir / "manuscript.docx",
                                                        base=out_dir))
        counts = flow_counts(state.get("prisma_counts") or {}, state.get("protocol"))
        return {
            "report": {"round": rnd, "folder": str(out_dir), "files": files,
                       "prisma_consistent": counts["consistent"]},
            "completed_stages": ["report"],
            "audit_log": [{"node": "report", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "report written", "folder": str(out_dir),
                           "files": sorted(files), "prisma_checks": counts["checks"],
                           "run_cost_usd": summary["llm"]["cost_usd"],
                           "llm_calls_total": summary["llm"]["calls"],
                           "seconds": summary["seconds"],
                           "writer": {k: v for k, v in prose["source"].items()},
                           **(prose["usage"] or {})}],
        }

    return report
