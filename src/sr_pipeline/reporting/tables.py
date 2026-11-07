"""Evidence tables (CSV) and the BibTeX file of the included studies.

Every table is written by code from the state, never by the LLM. Items are read through
``latest()`` (one row per paper, the most recent round), restricted to the current
records. Values keep their status: a field the extraction could not verify is written
``not reported``, and its quote column stays empty.

Files (``runs/<id>/tables/``):
- ``studies.csv``: one row per included study (label, author, year, design, n, PICO
  fields, text used, risk of bias overall);
- ``extraction.csv``: one row per (study, field): value, status, quote, chunk id;
- ``results.csv``: one row per (outcome, study): direction, metric used, metric
  conflict, Hedges' g with its CI, reason when no direction;
- ``risk_of_bias.csv``: one row per (study, domain): judgment, verified quote, reason;
- ``screening.csv``: every screened record with its title/abstract and full-text
  decision and deciding criterion (used by the Part 10 evaluation).
"""

import csv
import re
from pathlib import Path

from ..agents.extraction import included_papers
from ..state import latest

STUDY_FIELDS = ("design", "country", "setting", "n_enrolled", "n_analysed", "age", "sex",
                "population", "intervention", "intervention_dose", "comparator", "follow_up")


def _write(path: Path, rows: list[dict], columns: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in columns})
    return path


def _value(ext: dict | None, key: str) -> str | None:
    f = ((ext or {}).get("fields") or {}).get(key) or {}
    return f.get("value") if f.get("status") == "reported" else "not reported"


def _current(items: list[dict], ids: set[str]) -> dict[str, dict]:
    return {i["paper_id"]: i for i in latest(items) if i["paper_id"] in ids}


def study_rows(state: dict) -> list[dict]:
    syn = state.get("synthesis") or {}
    ids = {p["paper_id"] for p in included_papers(state)}
    ext = _current(state.get("extractions", []), ids)
    papers = {p["paper_id"]: p for p in state.get("unique_records", [])}
    rows = []
    for s in syn.get("studies", []):
        e, p = ext.get(s["paper_id"]), papers.get(s["paper_id"], {})
        rows.append({"label": s["label"], "paper_id": s["paper_id"], "author": s["author"],
                     "year": s["year"], "title": s["title"], "doi": p.get("doi"),
                     "venue": p.get("venue"),
                     "text_used": "abstract only" if s["abstract_only"] else "full text",
                     "design_category": s["design_category"],
                     "design_verified": s["design_verified"],
                     **{k: _value(e, k) for k in STUDY_FIELDS},
                     "rob_tool": s["rob_tool"], "rob_overall": s["rob_overall"]})
    return rows


def extraction_rows(state: dict) -> list[dict]:
    ids = {p["paper_id"] for p in included_papers(state)}
    labels = {s["paper_id"]: s["label"] for s in (state.get("synthesis") or {}).get(
        "studies", [])}
    rows = []
    for pid, e in _current(state.get("extractions", []), ids).items():
        for key, f in (e.get("fields") or {}).items():
            rows.append({"label": labels.get(pid, ""), "paper_id": pid, "field": key,
                         "status": f.get("status"), "value": f.get("value")
                         if f.get("status") == "reported" else "not reported",
                         "quote": f.get("quote") if f.get("status") == "reported" else "",
                         "chunk_id": f.get("chunk_id"), "flags": ";".join(f.get("flags") or [])})
    return rows


def result_rows(state: dict) -> list[dict]:
    rows = []
    for g in (state.get("synthesis") or {}).get("outcomes", []):
        for p in g["per_study"]:
            rows.append({"outcome_id": g["id"], "outcome": g["outcome"], "label": p["label"],
                         "measure": p.get("measure"), "direction": p["direction"],
                         "metric_used": p.get("metric_used"),
                         "metric_conflict": p.get("metric_conflict"),
                         "conflict": p.get("conflict"), "timepoint": p.get("timepoint"),
                         "hedges_g": p.get("g"), "ci_low": p.get("ci_low"),
                         "ci_high": p.get("ci_high"), "reason": p.get("reason")})
    return rows


def rob_rows(state: dict) -> list[dict]:
    ids = {p["paper_id"] for p in included_papers(state)}
    labels = {s["paper_id"]: s["label"] for s in (state.get("synthesis") or {}).get(
        "studies", [])}
    rows = []
    for pid, r in _current(state.get("risk_of_bias", []), ids).items():
        for d in r.get("domains") or []:
            rows.append({"label": labels.get(pid, ""), "paper_id": pid, "tool": r["tool"],
                         "domain": d.get("domain"), "name": d.get("name"),
                         "judgment": d.get("judgment"), "verified": d.get("verified"),
                         "quote": d.get("quote") if d.get("verified") else "",
                         "reason": d.get("reason"), "overall": r.get("overall")})
    return rows


def screening_rows(state: dict) -> list[dict]:
    papers = state.get("unique_records", [])
    ids = {p["paper_id"] for p in papers}
    ta = _current(state.get("ta_decisions", []), ids)
    ft = _current(state.get("ft_decisions", []), ids)
    rows = []
    for p in papers:
        t, f = ta.get(p["paper_id"]), ft.get(p["paper_id"])
        rows.append({"paper_id": p["paper_id"], "title": p.get("title"), "year": p.get("year"),
                     "doi": p.get("doi"), "ta": t and t["final"],
                     "ta_criterion": t and t.get("criterion"),
                     "ta_resolved_by": t and t.get("resolved_by"),
                     "ft": f and f["final"], "ft_criterion": f and f.get("criterion"),
                     "ft_resolved_by": f and f.get("resolved_by"),
                     "fulltext_status": f and f.get("fulltext_status")})
    return rows


def write_tables(state: dict, folder: str | Path) -> dict[str, Path]:
    folder = Path(folder)
    out = {}
    for name, rows in (("studies", study_rows(state)), ("extraction", extraction_rows(state)),
                       ("results", result_rows(state)), ("risk_of_bias", rob_rows(state)),
                       ("screening", screening_rows(state))):
        cols = list(dict.fromkeys(k for r in rows for k in r)) or ["paper_id"]
        out[f"{name}.csv"] = _write(folder / f"{name}.csv", rows, cols)
    return out


# --- BibTeX -------------------------------------------------------------------------------

def _bib_escape(text: str) -> str:
    return re.sub(r"([&%$#_{}])", r"\\\1", text or "")


def bib_key(paper: dict, used: set[str]) -> str:
    authors = paper.get("authors") or []
    last = authors[0].split(",")[0].split()[-1] if authors and authors[0].strip() else "anon"
    base = re.sub(r"[^A-Za-z]", "", last).lower() or "anon"
    base += str(paper.get("year") or "nd")
    key, n = base, 0
    while key in used:
        n += 1
        key = base + "abcdefghijklmnopqrstuvwxyz"[(n - 1) % 26] * ((n - 1) // 26 + 1)
    used.add(key)
    return key


def bibtex(papers: list[dict]) -> str:
    """One @article entry per paper (fields the databases gave; none invented)."""
    used: set[str] = set()
    entries = []
    for p in papers:
        fields = {"title": p.get("title"), "author": " and ".join(p.get("authors") or []),
                  "year": p.get("year"), "journal": p.get("venue"), "doi": p.get("doi"),
                  "pmid": p.get("pmid"), "url": p.get("url")}
        body = ",\n".join(f"  {k} = {{{_bib_escape(str(v))}}}" for k, v in fields.items()
                          if v not in (None, ""))
        entries.append(f"@article{{{bib_key(p, used)},\n{body}\n}}")
    return "\n\n".join(entries) + ("\n" if entries else "")
