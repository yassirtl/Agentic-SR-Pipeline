"""Report stage: PRISMA flow diagram, manuscript (Markdown + DOCX), evidence tables (CSV),
BibTeX, run summary. Code only (no LLM call, $0).

Writes ``<out_dir>/`` (``runs/<id>/report/``):
  manuscript.md, manuscript.docx, prisma_flow.png, references.bib, run_summary.json,
  prisma_counts.json, protocol.json, tables/{studies,extraction,results,risk_of_bias,
  screening}.csv
Forest plots stay where the synthesis wrote them (``runs/<id>/figures``); the manuscript
links to them.
"""

import json
from pathlib import Path

from ..agents.extraction import included_papers
from ..state import stage_round
from .manuscript import manuscript
from .prisma import flow_counts, flow_diagram
from .summary import run_summary
from .tables import bibtex, write_tables
from .word import markdown_to_docx

__all__ = ["make_report", "flow_counts", "run_summary"]


def make_report(out_dir: str | Path, *, models: dict[str, str] | None = None,
                embedder=None):
    """``models``: agent -> model, for the Methods. ``embedder``: the run's embedder, whose
    cumulative ``usage()`` goes into the run summary (embeddings are not in the audit)."""
    out_dir = Path(out_dir)

    def report(state: dict) -> dict:
        rnd = stage_round("report", state.get("completed_stages", []))
        out_dir.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}

        png = flow_diagram(state.get("prisma_counts") or {}, out_dir / "prisma_flow.png")
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

        usage = embedder.usage() if embedder is not None and hasattr(embedder, "usage") \
            else None
        summary = run_summary(state, embeddings=usage)
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        files["run_summary.json"] = str(out_dir / "run_summary.json")

        md = manuscript(state, base=out_dir, models=models or {}, summary=summary,
                        prisma_png=png)
        (out_dir / "manuscript.md").write_text(md)
        files["manuscript.md"] = str(out_dir / "manuscript.md")
        files["manuscript.docx"] = str(markdown_to_docx(md, out_dir / "manuscript.docx",
                                                        base=out_dir))
        counts = flow_counts(state.get("prisma_counts") or {})
        return {
            "report": {"round": rnd, "folder": str(out_dir), "files": files,
                       "prisma_consistent": counts["consistent"]},
            "completed_stages": ["report"],
            "audit_log": [{"node": "report", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "report written", "folder": str(out_dir),
                           "files": sorted(files), "prisma_checks": counts["checks"],
                           "run_cost_usd": summary["llm"]["cost_usd"],
                           "llm_calls_total": summary["llm"]["calls"],
                           "seconds": summary["seconds"]}],
        }

    return report
