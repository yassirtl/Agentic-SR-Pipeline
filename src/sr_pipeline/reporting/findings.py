"""Findings for the article (Part 11f): topic-aware study table, evidence map and summary of
findings. Code only, from the extraction already in the state ($0, reproducible).

Driven by the topic's ``evidence_map`` (``evaluation/topics.yaml``, or the protocol when it
carries one): two extraction fields as the map's rows and columns, keyword groups for each,
the study-table columns and which of them are numeric. A topic without it (e.g. a clinical
question) gets none of this, and the article keeps its PICO table.

Rules (stated in the article):
- a value goes to the first group whose keyword it contains (whole word, case-insensitive,
  plural allowed); a field "not reported" is classified from the study title instead; no
  keyword -> "Other", nothing at all -> "Not reported": every study is counted once;
- a numeric column shows a value only when it contains a digit (words such as "low power"
  are not a measurement);
- accuracy figures are listed as reported, with no ranking: datasets, train/test splits
  and class sets differ between studies, so the figures are not comparable.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..agents.extraction import included_papers
from ..state import latest

OTHER, NONE = "Other", "Not reported"
HEADERS = {"model": "Model", "platform": "Device", "dataset": "Dataset",
           "partition": "Train/test split", "classification_metrics": "Reported metrics",
           "latency": "Latency", "power_energy": "Power / energy", "memory": "Memory",
           "parameters": "Parameters", "operations": "Operations", "clock": "Clock",
           "arithmetic": "Arithmetic", "classes": "Classes"}
CELL = 90                                  # characters per table cell in the article
_ACC = re.compile(r"accura\w*[^0-9%\n]{0,25}?(\d{1,3}(?:\.\d+)?)\s?%", re.I)


def spec(protocol: dict) -> dict | None:
    """The topic's evidence-map settings, or None."""
    if protocol.get("evidence_map"):
        return protocol["evidence_map"]
    topic = protocol.get("topic")
    if not topic:
        return None
    try:
        from ..evaluation.topics import load_topic
        return load_topic(str(topic)).get("evidence_map") or None
    except (KeyError, FileNotFoundError, ValueError, OSError):
        return None


def _value(fields: dict, key: str) -> str | None:
    f = (fields or {}).get(key) or {}
    v = (f.get("value") or "").strip() if f.get("status") == "reported" else ""
    return v or None


def has_digit(text: str | None) -> bool:
    return bool(text and re.search(r"\d", text))


def _pattern(word: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"s?(?![A-Za-z0-9])", re.I)


def classify(text: str | None, groups: dict[str, list[str]]) -> str | None:
    """First group with a keyword in ``text`` (None when no keyword matches)."""
    if not text:
        return None
    for name, words in groups.items():
        if any(_pattern(str(w)).search(text) for w in words or []):
            return name
    return None


def _cut(text: str | None, n: int = CELL) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def studies(state: dict) -> list[dict]:
    """Included studies in synthesis order: label, author, year, title, abstract_only and
    the latest extraction fields."""
    ids = {p["paper_id"] for p in included_papers(state)}
    ext = {e["paper_id"]: e for e in latest(state.get("extractions") or [])
           if e["paper_id"] in ids}
    out = []
    for s in (state.get("synthesis") or {}).get("studies", []):
        out.append({**s, "fields": (ext.get(s["paper_id"]) or {}).get("fields") or {}})
    return out


# --- Study table -----------------------------------------------------------------------

def table_columns(sp: dict | None, protocol: dict) -> list[str]:
    fields = list((protocol.get("extraction_fields") or {}).keys())
    cols = [c for c in ((sp or {}).get("table") or {}).get("columns") or [] if c in fields]
    return cols or fields[:6]


def study_table(state: dict, sp: dict | None) -> tuple[list[dict], list[tuple[str, str]]]:
    """Rows and (key, header) columns of the article's study table for a topic with its own
    extraction fields. Numeric columns keep only values with a digit."""
    protocol = state.get("protocol") or {}
    cols = table_columns(sp, protocol)
    numeric = set(((sp or {}).get("table") or {}).get("numeric") or [])
    rows = []
    for s in studies(state):
        r = {"who": f"{s['label']} {s['author']} {s['year'] or ''}".strip()}
        for c in cols:
            v = _value(s["fields"], c)
            if c in numeric and not has_digit(v):
                v = None
            r[c] = _cut(v) if v else "–"
        r["used"] = "abstract only †" if s.get("abstract_only") else "full text"
        rows.append(r)
    columns = [("who", "Study")] + [(c, HEADERS.get(c, c.replace("_", " ").capitalize()))
                                    for c in cols] + [("used", "Text used")]
    return rows, columns


# --- Evidence map ------------------------------------------------------------------------

def _axis(s: dict, axis: dict) -> tuple[str, str]:
    """(group, source) of one study on one axis; source = value / title / none."""
    groups = axis.get("groups") or {}
    v = _value(s["fields"], axis["field"])
    if v:
        return classify(v, groups) or OTHER, "value"
    g = classify(s.get("title"), groups)
    return (g, "title") if g else (NONE, "none")


def evidence_map(state: dict, sp: dict | None) -> dict | None:
    """Rows x columns grid of study labels. Every included study is in exactly one cell."""
    if not sp or not sp.get("rows") or not sp.get("cols"):
        return None
    rax, cax = sp["rows"], sp["cols"]
    order_r = list(rax.get("groups") or {}) + [OTHER, NONE]
    order_c = list(cax.get("groups") or {}) + [OTHER, NONE]
    cells: dict[tuple[str, str], list[str]] = {}
    sources = {"rows": Counter(), "cols": Counter()}
    for s in studies(state):
        r, rs = _axis(s, rax)
        c, cs = _axis(s, cax)
        sources["rows"][rs] += 1
        sources["cols"][cs] += 1
        cells.setdefault((r, c), []).append(s["label"])
    used_r = [r for r in order_r if any(k[0] == r for k in cells)]
    used_c = [c for c in order_c if any(k[1] == c for k in cells)]
    return {"row_label": rax.get("label") or rax["field"], "col_label":
            cax.get("label") or cax["field"], "rows": used_r, "cols": used_c,
            "cells": {f"{r}|{c}": v for (r, c), v in cells.items()},
            "n": sum(len(v) for v in cells.values()),
            "sources": {k: dict(v) for k, v in sources.items()},
            "row_field": rax["field"], "col_field": cax["field"]}


def evidence_map_rows(em: dict) -> tuple[list[dict], list[tuple[str, str]]]:
    """Markdown table rows: one per row group, a count (labels) per column, a total."""
    rows = []
    for r in em["rows"]:
        row = {"g": r}
        total = 0
        for i, c in enumerate(em["cols"]):
            labels = em["cells"].get(f"{r}|{c}") or []
            total += len(labels)
            row[f"c{i}"] = f"{len(labels)} ({', '.join(labels)})" if labels else ""
        row["t"] = str(total)
        rows.append(row)
    total = {"g": "**Total**", "t": f"**{em['n']}**"}
    for i, c in enumerate(em["cols"]):
        total[f"c{i}"] = str(sum(len(em["cells"].get(f"{r}|{c}") or []) for r in em["rows"]))
    rows.append(total)
    cols = [("g", f"{em['row_label']} \\ {em['col_label']}")] + [
        (f"c{i}", c) for i, c in enumerate(em["cols"])] + [("t", "Total")]
    return rows, cols


def evidence_map_png(em: dict, path: str | Path) -> Path:
    """Heat map of the counts (the article's evidence map figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = [[len(em["cells"].get(f"{r}|{c}") or []) for c in em["cols"]] for r in em["rows"]]
    w = 1.6 + 1.25 * len(em["cols"])
    h = 1.2 + 0.55 * len(em["rows"])
    fig, ax = plt.subplots(figsize=(max(7.0, w), max(3.0, h)), dpi=150)
    ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, max(map(max, grid))))
    ax.set_xticks(range(len(em["cols"])))
    ax.set_xticklabels([_wrap(c) for c in em["cols"]], fontsize=8)
    ax.set_yticks(range(len(em["rows"])))
    ax.set_yticklabels(em["rows"], fontsize=8)
    ax.xaxis.tick_top()
    top = max(map(max, grid)) or 1
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > top / 2 else "#111")
    ax.set_xlabel(em["col_label"], fontsize=9)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel(em["row_label"], fontsize=9)
    ax.set_title(f"Evidence map: {em['n']} included studies", fontsize=10, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _wrap(text: str, n: int = 14) -> str:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])


# --- Summary of findings ---------------------------------------------------------------

def accuracy_values(text: str | None) -> list[float]:
    """Accuracy percentages as printed ("Accuracy: 98.18%", "an accuracy of 97 %")."""
    return [float(m) for m in _ACC.findall(text or "") if 0 < float(m) <= 100]


def summary(state: dict, sp: dict | None, em: dict | None) -> dict | None:
    """Counts behind the summary of findings (no ranking, no pooled accuracy)."""
    protocol = state.get("protocol") or {}
    if not protocol.get("extraction_fields"):
        return None
    ss = studies(state)
    numeric = set(((sp or {}).get("table") or {}).get("numeric") or [])
    reported = {}
    for key in protocol["extraction_fields"]:
        vals = [_value(s["fields"], key) for s in ss]
        reported[key] = sum(1 for v in vals if v and (key not in numeric or has_digit(v)))
    datasets = Counter()
    for s in ss:
        v = _value(s["fields"], "dataset")
        if v:
            datasets[classify(v, (sp or {}).get("datasets") or {}) or OTHER] += 1
    split = Counter()
    for s in ss:
        v = (_value(s["fields"], "partition") or "").lower()
        if "inter" in v:
            split["inter-patient"] += 1
        elif "intra" in v or "random" in v:
            split["intra-patient / random"] += 1
        elif v:
            split["other wording"] += 1
    acc = {s["label"]: accuracy_values(_value(s["fields"], "classification_metrics"))
           for s in ss}
    acc = {k: v for k, v in acc.items() if v}
    allv = [x for v in acc.values() for x in v]
    rob = Counter(s.get("rob_overall") or "not assessed" for s in ss)
    return {"n": len(ss), "abstract_only": sum(1 for s in ss if s.get("abstract_only")),
            "reported": reported, "datasets": dict(datasets.most_common()),
            "split": dict(split), "accuracy_studies": sorted(acc, key=_label_key),
            "accuracy_min": min(allv) if allv else None,
            "accuracy_max": max(allv) if allv else None, "rob": dict(rob.most_common()),
            "map_top": _top_cells(em)}


def _label_key(label: str) -> int:
    m = re.search(r"\d+", label)
    return int(m.group()) if m else 0


def _top_cells(em: dict | None, k: int = 3) -> list[tuple[str, str, int]]:
    if not em:
        return []
    cells = [(key.split("|")[0], key.split("|")[1], len(v)) for key, v in em["cells"].items()
             if NONE not in key and OTHER not in key]
    return sorted(cells, key=lambda x: -x[2])[:k]


def _g(x: float) -> str:
    return f"{x:g}"


def summary_lines(sm: dict, protocol: dict) -> list[str]:
    """Bullet points of the summary of findings (also the writer's facts)."""
    n = sm["n"]
    rep = sm["reported"]
    out = [f"{n} studies were included; {sm['abstract_only']} of them were assessed on the "
           "abstract only (no open-access full text)."]
    if sm["map_top"]:
        out.append("Most frequent model and device combinations: " + "; ".join(
            f"{r} on {c} ({k} studies)" for r, c, k in sm["map_top"]) + ".")
    if sm["datasets"]:
        out.append(f"Dataset named in {sum(sm['datasets'].values())} studies: " + ", ".join(
            f"{name} {k}" for name, k in sm["datasets"].items()) + ".")
    if sm["split"]:
        out.append("Train/test split stated in " + str(sum(sm["split"].values()))
                   + " studies: " + ", ".join(f"{k} {v}" for k, v in sm["split"].items())
                   + ".")
    if sm["accuracy_studies"]:
        out.append(f"An accuracy figure is reported by {len(sm['accuracy_studies'])} studies "
                   f"({', '.join(sm['accuracy_studies'])}); the figures range from "
                   f"{_g(sm['accuracy_min'])}% to {_g(sm['accuracy_max'])}%. They are not "
                   "comparable: datasets, train/test splits and class sets differ.")
    eff = [(HEADERS.get(k, k), rep[k]) for k in ("latency", "power_energy", "memory",
                                                 "parameters", "operations") if k in rep]
    if eff:
        out.append("Efficiency measured with a number: " + ", ".join(
            f"{name.lower()} {k}" for name, k in eff) + f" of {n} studies.")
    if sm["rob"]:
        out.append("Risk of bias (overall): " + ", ".join(f"{k} {v}" for k, v in
                                                          sm["rob"].items()) + ".")
    return out
