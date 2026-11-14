"""Accuracy of title/abstract screening against a human-labelled sample (Part 10).

1. ``make_sheet``: a random sample (fixed seed) of the records the run screened, written
   as ``labels.csv`` with title and abstract and an empty ``human`` column, plus
   ``criteria.md`` with the protocol's numbered criteria. The pipeline's decisions are
   **not** in the sheet: the labeller must not see them (blind labelling).
2. The user writes ``include`` or ``exclude`` (``i`` / ``e`` also accepted) in ``human``,
   judging only the title and abstract, like the first screening stage.
3. ``score_labels``: the human label is the reference; the pipeline's title/abstract
   decision is compared with it: confusion matrix, accuracy, sensitivity (human
   includes the pipeline kept: the number that matters for a review, a lost include is
   a lost study), specificity, Cohen's kappa (the screening module's function), and the
   list of disagreements with the pipeline's deciding criterion.

The sample is simple random over screened records, so with a low include rate it may
hold few includes: sensitivity then rests on few records, and the output says how many.
"""

import csv
import random
from pathlib import Path

from ..agents.screening import cohen_kappa, criteria, criterion_labels
from ..reporting.tables import screening_rows

SHEET_COLS = ["n", "paper_id", "title", "abstract", "human", "note"]
ANSWERS = {"include": "include", "i": "include", "in": "include", "yes": "include",
           "exclude": "exclude", "e": "exclude", "ex": "exclude", "out": "exclude",
           "no": "exclude"}


UNSURE = ("unsure", "u", "?", "not sure")


def sheets(root: str | Path) -> list[Path]:
    """Label sheets under ``root`` (evaluation/labels/<run_id>/labels.csv), newest first."""
    return sorted(Path(root).glob("*/labels.csv"), reverse=True)


def sheet_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def set_label(path: str | Path, paper_id: str, human: str, note: str | None = None) -> None:
    """Write one label (include / exclude / unsure) and note, at once (atomic replace)."""
    human = human.strip().lower()
    if human not in ANSWERS and human not in UNSURE:
        raise ValueError(f"unknown label {human!r}")
    path = Path(path)
    rows = sheet_rows(path)
    for r in rows:
        if r["paper_id"] == paper_id:
            r["human"] = ANSWERS.get(human, "unsure")
            if note is not None:
                r["note"] = note.strip()
            break
    else:
        raise KeyError(f"{paper_id} is not in {path}")
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def sample_records(state: dict, n: int = 30, seed: int = 10) -> list[dict]:
    screened = [r for r in screening_rows(state) if r["ta"] in ("include", "exclude")]
    rng = random.Random(seed)
    picked = rng.sample(screened, min(n, len(screened)))
    abstracts = {p["paper_id"]: p.get("abstract") or "" for p in state.get("unique_records", [])}
    return [{"n": i, "paper_id": r["paper_id"], "title": r["title"],
             "abstract": abstracts.get(r["paper_id"], ""), "human": "", "note": ""}
            for i, r in enumerate(picked, 1)]


def criteria_text(protocol: dict) -> str:
    lines = [f"# Screening criteria: {protocol.get('question', '')}", "",
             "Label each record from its title and abstract only, as a first-stage "
             "screener: `include` unless the record clearly fails a criterion (when in "
             "doubt, include). Criteria marked *full text only* cannot exclude here.", ""]
    full_text = set(protocol.get("full_text_criteria") or [])
    for cid, text in criteria(protocol):
        tag = " *(full text only)*" if text in full_text else ""
        lines.append(f"- **{cid}**: {text}{tag}")
    return "\n".join(lines) + "\n"


def make_sheet(state: dict, folder: str | Path, n: int = 30, seed: int = 10) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sheet = folder / "labels.csv"
    if sheet.exists():
        raise FileExistsError(f"{sheet} exists: labels are never overwritten")
    rows = sample_records(state, n, seed)
    with sheet.open("w", newline="") as f:
        w = csv.DictWriter(f, SHEET_COLS)
        w.writeheader()
        w.writerows(rows)
    (folder / "criteria.md").write_text(criteria_text(state.get("protocol") or {}))
    return {"sheet": sheet, "criteria": folder / "criteria.md", "records": len(rows),
            "seed": seed}


def read_labels(path: str | Path) -> tuple[dict[str, str], list[str]]:
    """({paper_id: include|exclude}, problems). Blank rows are skipped (not labelled)."""
    labels, problems = {}, []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("human") or "").strip().lower()
            if not raw or raw in UNSURE:      # not labelled, or seen and left undecided
                continue
            if raw not in ANSWERS:
                problems.append(f"row {r.get('n')}: unknown label {raw!r}")
                continue
            labels[r["paper_id"]] = ANSWERS[raw]
    return labels, problems


def score_labels(state: dict, labels: dict[str, str]) -> dict:
    rows = {r["paper_id"]: r for r in screening_rows(state)}
    names = criterion_labels(state.get("protocol") or {})
    pairs, missing, disagreements = [], [], []
    for pid, human in labels.items():
        r = rows.get(pid)
        if not r or r["ta"] not in ("include", "exclude"):
            missing.append(pid)
            continue
        pairs.append((human, r["ta"]))
        if human != r["ta"]:
            c = r.get("ta_criterion")
            disagreements.append({"paper_id": pid, "title": r["title"], "human": human,
                                  "pipeline": r["ta"], "criterion": c,
                                  "label": names.get(c) if c else None,
                                  "resolved_by": r.get("ta_resolved_by")})
    tp = sum(h == "include" and p == "include" for h, p in pairs)
    fn = sum(h == "include" and p == "exclude" for h, p in pairs)
    fp = sum(h == "exclude" and p == "include" for h, p in pairs)
    tn = sum(h == "exclude" and p == "exclude" for h, p in pairs)
    n = len(pairs)

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {"labelled": n, "human_includes": tp + fn, "human_excludes": fp + tn,
            "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
            "accuracy": ratio(tp + tn, n), "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp),
            "kappa": cohen_kappa([h for h, _ in pairs], [p for _, p in pairs]),
            "disagreements": disagreements, "not_in_run": missing}
