"""Per-stage human check (Part 11c). No LLM, no network, no Streamlit.

For one finished run and one stage, a random sample of items (5 to 20) is drawn with a fixed
seed and stored; the user marks each item correct / wrong / unsure, with an optional note.
The result is the share of judged items marked correct, with a Wilson 95% interval.

Stages and their unit (one item = one thing the pipeline decided or wrote):

- ``search``: a unique record found by the search (all records, screened or not). The user
  says whether it is on topic: the result is the **relevant share** (search precision).
- ``ta`` / ``ft``: a record's final screening decision (latest round), with the deciding
  criterion and quote.
- ``extraction``: one **reported** field of one study (value + quote). Fields marked "not
  reported" and the outcome result blocks (arm tables) are not sampled.
- ``rob``: one risk-of-bias domain of one study (judgement + quote + reason).
- ``synthesis``: one narrative or heterogeneity sentence, shown with the evidence rows of
  the studies it cites.

Files (tracked: ``runs/`` is git-ignored, so each row keeps a copy of what was judged):
``evaluation/stage_checks/<run_id>/<stage>.csv`` (items + ``mark`` + ``note``) and
``<stage>.json`` (seed, sample size, population size, unit). A sample is never redrawn
once it has a mark. Drawing = sort the item ids, shuffle once with ``Random(seed)``, take
the first n: a larger n extends a smaller sample with the same seed.

This check is **not blind**: the user sees the pipeline's answer. Report it as "judged
correct by the author on k of n sampled items", not as accuracy against an independent
reference (the blind labels of ``labels.py`` are that).
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from ..agents.screening import criterion_labels
from ..state import latest
from .topics import TOPICS_FILE

CHECKS_DIR = TOPICS_FILE.parent / "stage_checks"     # evaluation/stage_checks (tracked)
MIN_N, MAX_N, DEFAULT_N, DEFAULT_SEED = 5, 20, 10, 10
CONTEXT_CHARS = 1500

STAGES: dict[str, dict] = {
    "search": {"name": "Search relevance", "unit": "unique record found by the search",
               "ask": "Is this record on topic for the review question?",
               "result": "relevant share"},
    "ta": {"name": "Title/abstract screening", "unit": "record screened at title/abstract "
           "(final decision, latest round)",
           "ask": "Is the include / exclude decision right, from the title and abstract?",
           "result": "decisions judged correct"},
    "ft": {"name": "Full-text screening", "unit": "record screened at full text "
           "(final decision, latest round)",
           "ask": "Is the include / exclude decision right, and does the quote support it?",
           "result": "decisions judged correct"},
    "extraction": {"name": "Extraction", "unit": "reported field of one study (value + quote)",
                   "ask": "Is the value right, and does the quote support it?",
                   "result": "values judged correct"},
    "rob": {"name": "Risk of bias", "unit": "risk-of-bias domain of one study",
            "ask": "Is the judgement right, and do the quote and reason support it?",
            "result": "judgements judged correct"},
    "synthesis": {"name": "Synthesis claims", "unit": "narrative or heterogeneity sentence",
                  "ask": "Is the claim supported by the studies it cites?",
                  "result": "claims judged supported"},
}
MARKS = ("correct", "wrong", "unsure")
MARK_ALIASES = {"c": "correct", "y": "correct", "yes": "correct", "ok": "correct",
                "w": "wrong", "n": "wrong", "no": "wrong", "u": "unsure", "?": "unsure"}
COLS = ["n", "item_id", "paper_id", "title", "decision", "detail", "quote", "context", "link",
        "mark", "note"]


# --- Pipeline output → items ------------------------------------------------------------

def deciding_verdict(d: dict) -> dict:
    """Quote (and passage id) behind a screening decision: the resolver's own quote, else
    the verdict quote of the deciding criterion."""
    src = d.get("adjudicator") if d.get("resolved_by") == "adjudicator" else d.get("a")
    src = src or d.get("a") or {}
    if src.get("quote"):
        return {"quote": src["quote"], "chunk_id": src.get("chunk_id") or ""}
    crit = d.get("criterion")
    for v in src.get("verdicts") or []:
        if v.get("criterion") == crit and v.get("quote"):
            return {"quote": v["quote"], "chunk_id": v.get("chunk_id") or ""}
    return {"quote": "", "chunk_id": ""}


def deciding_quote(d: dict) -> str:
    return deciding_verdict(d)["quote"]


def _cut(text, n: int = CONTEXT_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _link(p: dict | None) -> str:
    p = p or {}
    if p.get("doi"):
        return f"https://doi.org/{p['doi']}"
    if p.get("pmid"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/"
    return p.get("url") or ""


def _passage(chunks: dict[str, str], chunk_id: str, fallback: str = "") -> str:
    text = chunks.get(chunk_id or "")
    return _cut(text) if text else fallback


def _search_items(state: dict, chunks: dict) -> list[dict]:
    out = []
    for p in state.get("unique_records") or []:
        rank = p.get("rank")
        src = ", ".join(p.get("sources") or [])
        out.append({"item_id": p["paper_id"], "paper_id": p["paper_id"], "title": p.get("title"),
                    "decision": "found by the search"
                    + (f" (best rank {rank})" if rank is not None else ""),
                    "detail": f"{p.get('year') or ''} · {src}".strip(" ·"), "quote": "",
                    "context": _cut(p.get("abstract")) or "(no abstract)", "link": _link(p)})
    return out


def _screening_items(state: dict, chunks: dict, key: str) -> list[dict]:
    papers = {p["paper_id"]: p for p in state.get("unique_records") or []}
    names = criterion_labels(state.get("protocol") or {})
    out = []
    for d in latest(state.get(key) or []):
        if d.get("final") not in ("include", "exclude"):
            continue
        pid = d["paper_id"]
        p = papers.get(pid) or {}
        crit = d.get("criterion")
        detail = (f"criterion {crit}" + (f" ({names[crit]})" if crit in names else "")
                  if crit and crit != "all" else "all criteria met") + f" · resolved by {d.get('resolved_by')}"
        v = deciding_verdict(d)
        if key == "ft_decisions":
            status = d.get("fulltext_status") or ""
            detail += f" · text: {status}" if status else ""
            context = _passage(chunks, v["chunk_id"], _cut(p.get("abstract")))
        else:
            context = _cut(p.get("abstract")) or "(no abstract)"
        out.append({"item_id": pid, "paper_id": pid, "title": d.get("title") or p.get("title"),
                    "decision": d["final"], "detail": detail, "quote": v["quote"],
                    "context": context, "link": _link(p)})
    return out


def _extraction_items(state: dict, chunks: dict) -> list[dict]:
    papers = {p["paper_id"]: p for p in state.get("unique_records") or []}
    out = []
    for ex in latest(state.get("extractions") or []):
        pid = ex["paper_id"]
        for name, f in (ex.get("fields") or {}).items():
            if (f or {}).get("status") != "reported":
                continue
            out.append({"item_id": f"{pid}::{name}", "paper_id": pid, "title": ex.get("title"),
                        "decision": str(f.get("value")), "detail": f"field {name} · text: "
                        f"{ex.get('fulltext_status') or 'unknown'}", "quote": f.get("quote") or "",
                        "context": _passage(chunks, f.get("chunk_id"), ""),
                        "link": _link(papers.get(pid))})
    return out


def _rob_items(state: dict, chunks: dict) -> list[dict]:
    papers = {p["paper_id"]: p for p in state.get("unique_records") or []}
    out = []
    for r in latest(state.get("risk_of_bias") or []):
        pid = r["paper_id"]
        for dom in r.get("domains") or []:
            passage = _passage(chunks, dom.get("chunk_id"), "")
            reason = dom.get("reason") or ""
            out.append({"item_id": f"{pid}::{dom.get('domain')}", "paper_id": pid,
                        "title": r.get("title"), "decision": str(dom.get("judgment")),
                        "detail": f"{r.get('tool') or ''} {dom.get('domain')}: {dom.get('name') or ''}"
                        + ("" if dom.get("verified", True) else " · quote not verified"),
                        "quote": dom.get("quote") or "",
                        "context": _cut(f"Reason: {reason}" + (f" | Passage: {passage}"
                                                                if passage else "")),
                        "link": _link(papers.get(pid))})
    return out


def _synthesis_items(state: dict, chunks: dict) -> list[dict]:
    syn = state.get("synthesis") or {}
    rows = {}
    for line in (syn.get("evidence") or "").splitlines():
        label = line.split("|", 1)[0].strip()
        if "|" in line and label[:1] == "S" and label[1:].isdigit():
            rows[label] = line.strip()
    out = []
    for kind in ("narrative", "heterogeneity"):
        for i, s in enumerate(syn.get(kind) or [], 1):
            cited = s.get("studies") or []
            shown = [rows.get(c, c) for c in cited[:8]]
            more = f" … and {len(cited) - 8} more" if len(cited) > 8 else ""
            out.append({"item_id": f"{kind}:{i}", "paper_id": "", "title":
                        f"{kind} · section {s.get('section')}", "decision": s.get("text") or "",
                        "detail": f"cites {', '.join(cited) or 'no study'}", "quote": "",
                        "context": _cut(" || ".join(shown) + more, 3000), "link": ""})
    return out


def stage_items(state: dict, stage: str, chunks: dict[str, str] | None = None) -> list[dict]:
    """Every item of ``stage`` in the run (the population the sample is drawn from)."""
    chunks = chunks or {}
    if stage == "search":
        return _search_items(state, chunks)
    if stage in ("ta", "ft"):
        return _screening_items(state, chunks, f"{stage}_decisions")
    if stage == "extraction":
        return _extraction_items(state, chunks)
    if stage == "rob":
        return _rob_items(state, chunks)
    if stage == "synthesis":
        return _synthesis_items(state, chunks)
    raise KeyError(f"unknown stage {stage!r} (known: {', '.join(STAGES)})")


def load_chunks(run_dir: str | Path) -> dict[str, str]:
    """chunk_id → passage text from the run's RAG index (empty when there is none)."""
    out = {}
    try:
        lines = (Path(run_dir) / "index" / "chunks.jsonl").read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get("chunk_id"):
            out[c["chunk_id"]] = c.get("text") or ""
    return out


# --- Sampling ---------------------------------------------------------------------------

def draw(items: list[dict], n: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Seeded random sample: ids sorted, shuffled once, first n (all when fewer)."""
    by_id = {it["item_id"]: it for it in items}
    order = sorted(by_id)
    random.Random(seed).shuffle(order)
    return [dict(by_id[i], n=k, mark="", note="") for k, i in enumerate(order[:n], 1)]


def paths(run_id: str, stage: str, root: str | Path | None = None) -> tuple[Path, Path]:
    folder = Path(root or CHECKS_DIR) / run_id
    return folder / f"{stage}.csv", folder / f"{stage}.json"


def _write_rows(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def create_check(state: dict, run_id: str, stage: str, n: int = DEFAULT_N,
                 seed: int = DEFAULT_SEED, *, root: str | Path | None = None,
                 chunks: dict[str, str] | None = None) -> dict:
    """Draw and store a sample. Refuses to overwrite an existing one."""
    if stage not in STAGES:
        raise KeyError(f"unknown stage {stage!r}")
    if not MIN_N <= n <= MAX_N:
        raise ValueError(f"sample size must be {MIN_N} to {MAX_N}, got {n}")
    csv_path, meta_path = paths(run_id, stage, root)
    if csv_path.exists():
        raise FileExistsError(f"{csv_path} exists: a stored sample is never redrawn")
    items = stage_items(state, stage, chunks)
    if not items:
        raise ValueError(f"run {run_id} has no {STAGES[stage]['name'].lower()} items")
    rows = draw(items, n, seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "stage": stage, "name": STAGES[stage]["name"],
            "unit": STAGES[stage]["unit"], "question": STAGES[stage]["ask"], "seed": seed,
            "requested": n, "sampled": len(rows), "population": len(items),
            "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "blind": False}
    _write_rows(csv_path, rows)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"meta": meta, "rows": rows}


def load_check(run_id: str, stage: str, root: str | Path | None = None) -> dict | None:
    csv_path, meta_path = paths(run_id, stage, root)
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        meta = {"run_id": run_id, "stage": stage, "sampled": len(rows)}
    return {"meta": meta, "rows": rows}


def normalise_mark(raw) -> str:
    m = str(raw or "").strip().lower()
    return m if m in MARKS else MARK_ALIASES.get(m, "")


def set_mark(run_id: str, stage: str, item_id: str, mark: str | None = None,
             note: str | None = None, *, root: str | Path | None = None) -> dict:
    """Update one item's mark and/or note (None = unchanged) and save at once."""
    check = load_check(run_id, stage, root)
    if check is None:
        raise FileNotFoundError(f"no {stage} check for run {run_id}")
    for r in check["rows"]:
        if r["item_id"] == item_id:
            if mark is not None:
                m = normalise_mark(mark)
                if mark and not m:
                    raise ValueError(f"unknown mark {mark!r} (use {', '.join(MARKS)})")
                r["mark"] = m
            if note is not None:
                r["note"] = note.strip()
            break
    else:
        raise KeyError(f"{item_id} is not in the {stage} sample")
    _write_rows(paths(run_id, stage, root)[0], check["rows"])
    return check


def discard_check(run_id: str, stage: str, root: str | Path | None = None) -> None:
    """Delete a sample that has no mark yet (a marked sample is kept)."""
    check = load_check(run_id, stage, root)
    if check is None:
        return
    if any(normalise_mark(r.get("mark")) for r in check["rows"]):
        raise PermissionError("this sample has marks: it is kept")
    for p in paths(run_id, stage, root):
        p.unlink(missing_ok=True)


# --- Scoring ----------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score 95% interval for k successes out of n (None when n = 0)."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def score(rows: list[dict]) -> dict:
    """Share correct = correct / (correct + wrong); unsure and unmarked items are counted
    apart and left out of the share."""
    marks = [normalise_mark(r.get("mark")) for r in rows]
    c, w, u = (marks.count(m) for m in MARKS)
    judged = c + w
    return {"sampled": len(rows), "marked": c + w + u, "unmarked": len(rows) - c - w - u,
            "correct": c, "wrong": w, "unsure": u, "judged": judged,
            "share_correct": round(c / judged, 4) if judged else None,
            "ci95": wilson(c, judged),
            "wrong_items": [{"item_id": r["item_id"], "title": r.get("title"),
                             "decision": r.get("decision"), "note": r.get("note") or ""}
                            for r, m in zip(rows, marks) if m == "wrong"]}


def summarise(run_id: str, root: str | Path | None = None) -> dict:
    """{stage: meta + score} for every stage that has a stored sample (empty if none)."""
    out = {}
    for stage in STAGES:
        check = load_check(run_id, stage, root)
        if check is None:
            continue
        m = check["meta"]
        out[stage] = {"name": STAGES[stage]["name"], "unit": STAGES[stage]["unit"],
                      "result": STAGES[stage]["result"], "seed": m.get("seed"),
                      "population": m.get("population"), **score(check["rows"])}
    return out
