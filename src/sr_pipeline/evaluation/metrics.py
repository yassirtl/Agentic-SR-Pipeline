"""Evaluation metrics of one run (Part 10), pluggable.

The course asks for four criteria; each metric belongs to one of them:

- **completeness**: ``gold_recall`` (stage-by-stage funnel, ``recall.py``),
  ``extraction_completeness`` (fields reported with a verified quote; only the
  topic's own fields when the protocol declares them),
  ``prisma_consistency`` (the diagram's arithmetic checks), ``critic``;
- **accuracy**: ``gold_precision`` (lower bound), ``human_labels`` (screening vs the
  user's labelled sample, ``labels.py``);
- **inter-agent consistency**: ``screener_agreement`` (kappa A vs B per stage,
  adjudication share), ``coordinator`` (guardrail refusals, budget);
- **time savings**: ``time_cost`` (run time, LLM calls, cost; vs a manual estimate
  when ``evaluation/manual_effort.yaml`` is filled).

A metric is a function ``(ctx) -> dict | None`` registered with ``@metric(name,
criterion)``; ``None`` = not applicable (e.g. no gold list, no labels). A topic picks
its metrics with ``metrics: [...]`` in ``topics.yaml`` (default: all), so a field
can add its own (e.g. a diagnostic-review metric on 2×2 tables) without touching
the others.

``ctx`` = {state, summary (run_summary.json), run (run.json), topic, gold_rows,
labels (dict or None), manual (manual_effort.yaml or None)}.
"""

from collections import Counter
from pathlib import Path

import yaml

from ..agents.extraction import included_papers
from ..reporting.prisma import flow_counts
from ..state import latest
from .labels import score_labels
from .recall import gold_funnel

MANUAL_FILE = Path(__file__).resolve().parents[3] / "evaluation" / "manual_effort.yaml"
METRICS: dict = {}  # name -> (criterion, function)
CRITERIA = ("completeness", "accuracy", "consistency", "time")


def metric(name: str, criterion: str):
    assert criterion in CRITERIA

    def register(fn):
        METRICS[name] = (criterion, fn)
        return fn
    return register


def _ratio(a, b):
    return round(a / b, 4) if b else None


# --- completeness -------------------------------------------------------------------

@metric("gold_recall", "completeness")
def gold_recall(ctx: dict) -> dict | None:
    if not ctx.get("gold_rows"):
        return None
    f = gold_funnel(ctx["state"], ctx["gold_rows"])
    return {k: f[k] for k in ("counts", "recall", "screen_limit", "lost_by_stage", "studies")}


@metric("extraction_completeness", "completeness")
def extraction_completeness(ctx: dict) -> dict | None:
    """Share of fields reported with a verified quote. When the protocol declares its own
    ``extraction_fields`` (engineering topics T2/T3), only those count: the generic
    clinical fields (country, age, sex, comparator, follow-up...) do not apply to a
    hardware study, so counting them would measure the topic, not the extraction. The
    count over every field is kept under ``all_fields``."""
    state = ctx["state"]
    ids = {p["paper_id"] for p in included_papers(state)}
    exts = [e for e in latest(state.get("extractions") or []) if e["paper_id"] in ids]
    if not exts:
        return None
    topic_keys = set((state.get("protocol") or {}).get("extraction_fields") or {})

    def count(keys):
        status = Counter(f.get("status") for e in exts
                         for k, f in (e.get("fields") or {}).items()
                         if keys is None or k in keys)
        by_text = {}
        for e in exts:
            key = "abstract only" if "abstract_only" in (e.get("flags") or []) else "full text"
            c = by_text.setdefault(key, [0, 0])
            for k, f in (e.get("fields") or {}).items():
                if keys is None or k in keys:
                    c[0] += f.get("status") == "reported"
                    c[1] += 1
        total = sum(status.values())
        return {"fields": total, "reported": status.get("reported", 0),
                "share_reported": _ratio(status.get("reported", 0), total),
                "by_text": {k: {"reported": a, "fields": b, "share": _ratio(a, b)}
                            for k, (a, b) in by_text.items()}}

    scoped = count(topic_keys or None)
    out = {"studies": len(exts), "scope": "topic fields" if topic_keys else "all fields",
           **scoped}
    if topic_keys:
        out["topic_fields"] = sorted(topic_keys)
        a = count(None)
        out["all_fields"] = {"reported": a["reported"], "fields": a["fields"],
                             "share": a["share_reported"]}
    return out


@metric("prisma_consistency", "completeness")
def prisma_consistency(ctx: dict) -> dict | None:
    prisma = ctx["state"].get("prisma_counts")
    if not prisma:
        return None
    f = flow_counts(prisma, ctx["state"].get("protocol"))
    return {"consistent": f["consistent"], "checks": f["checks"]}


@metric("critic", "completeness")
def critic(ctx: dict) -> dict | None:
    qa = ctx["state"].get("qa_report")
    if not qa:
        return None
    def n(v):  # the QA report stores counts (lists in older drafts)
        return len(v) if isinstance(v, list) else (v or 0)
    return {"passed": qa.get("passed"), "blockers": n(qa.get("blockers")),
            "warnings": n(qa.get("warnings")), "loops_used": qa.get("loops_used")}


# --- accuracy -----------------------------------------------------------------------

@metric("gold_precision", "accuracy")
def gold_precision(ctx: dict) -> dict | None:
    if not ctx.get("gold_rows"):
        return None
    f = gold_funnel(ctx["state"], ctx["gold_rows"])
    return {"included": f["included"], "gold_included": f["counts"]["included"],
            "precision_lower_bound": f["precision_lower_bound"],
            "note": "lower bound: an include that is not in the gold list may still be eligible"}


@metric("human_labels", "accuracy")
def human_labels(ctx: dict) -> dict | None:
    if not ctx.get("labels"):
        return None
    return score_labels(ctx["state"], ctx["labels"])


# --- inter-agent consistency ----------------------------------------------------------

@metric("screener_agreement", "consistency")
def screener_agreement(ctx: dict) -> dict | None:
    prisma = ctx["state"].get("prisma_counts") or {}
    out = {}
    for key, name in (("title_abstract", "ta"), ("full_text", "ft")):
        c = prisma.get(key) or {}
        if not c:
            continue
        by = c.get("resolved_by") or {}
        decided = sum(by.values())
        out[name] = {"pairs": c.get("pairs"), "kappa": c.get("kappa"),
                     "agreement": c.get("agreement"), "resolved_by": by,
                     "adjudicated_share": _ratio(by.get("adjudicator", 0), decided),
                     "quotes_not_found": c.get("quotes_not_found"),
                     "llm_errors": c.get("llm_errors")}
    return out or None


@metric("coordinator", "consistency")
def coordinator(ctx: dict) -> dict | None:
    s = ctx.get("summary") or {}
    if not s:
        return None
    co = s.get("coordinator") or {}
    return {"steps": s.get("steps_taken"), "decisions": co.get("decisions"),
            "refusals": co.get("refusals"), "budget_exhausted": s.get("budget_exhausted"),
            "stages_run": s.get("stages_run")}


# --- time savings ---------------------------------------------------------------------

def manual_estimate(manual: dict | None, counts: dict) -> dict | None:
    """Minutes a manual team would need, from per-item minutes in manual_effort.yaml
    times this run's counts. None when any needed rate is missing (never guessed)."""
    if not manual:
        return None
    rates = manual.get("minutes_per_item") or {}
    reviewers = manual.get("reviewers") or 1
    needed = {"title_abstract": counts["screened"], "full_text": counts["full_text"],
              "extraction": counts["included"], "risk_of_bias": counts["included"]}
    if any(rates.get(k) is None for k in needed):
        return None
    per_task = {k: round(rates[k] * n * (reviewers if k in ("title_abstract", "full_text")
                                         else 1), 1) for k, n in needed.items()}
    fixed = manual.get("fixed_hours") or {}
    if any(v is None for v in fixed.values()):
        return None
    total_h = sum(per_task.values()) / 60 + sum(fixed.values())
    return {"minutes_per_task": per_task, "fixed_hours": fixed,
            "reviewers": reviewers, "hours": round(total_h, 1),
            "source": manual.get("source")}


@metric("time_cost", "time")
def time_cost(ctx: dict) -> dict | None:
    s = ctx.get("summary") or {}
    if not s:
        return None
    llm = s.get("llm") or {}
    emb = s.get("embeddings") or {}
    cost = (llm.get("cost_usd") or 0) + (emb.get("cost_usd") or 0)
    prisma = ctx["state"].get("prisma_counts") or {}
    ta, ft = prisma.get("title_abstract") or {}, prisma.get("full_text") or {}
    counts = {"screened": ta.get("records_screened") or 0,
              "full_text": ft.get("reports_sought") or 0,
              "included": len(ctx["state"].get("included") or [])}
    out = {"seconds": s.get("seconds"), "hours": round((s.get("seconds") or 0) / 3600, 2),
           "seconds_note": "wall clock from start to finish: includes human pauses and the "
                           "time between a crash and its --resume",
           "llm_calls": llm.get("calls"), "llm_calls_cached": llm.get("cached"),
           "cost_usd": round(cost, 4), "cost_is_lower_bound": llm.get("cost_is_lower_bound"),
           "cost_per_screened_record": _ratio(cost, counts["screened"]), "counts": counts}
    manual = manual_estimate(ctx.get("manual"), counts)
    if manual:
        out["manual"] = manual
        out["time_saved_share"] = round(1 - out["hours"] / manual["hours"], 3) \
            if manual["hours"] else None
    elif ctx.get("manual"):
        out["manual"] = "manual_effort.yaml has empty rates: no manual estimate computed"
    if ctx.get("manual") and ctx["manual"].get("benchmark"):
        out["benchmark"] = ctx["manual"]["benchmark"]
    return out


# --- running ----------------------------------------------------------------------------

def load_manual(path: str | Path | None = None) -> dict | None:
    path = Path(path or MANUAL_FILE)
    return yaml.safe_load(path.read_text()) if path.exists() else None


def run_metrics(ctx: dict, names: list[str] | None = None) -> dict:
    """{criterion: {metric: result}} for the chosen metrics (all by default)."""
    unknown = [n for n in names or [] if n not in METRICS]
    if unknown:
        raise KeyError(f"unknown metrics {unknown} (known: {', '.join(METRICS)})")
    out = {c: {} for c in CRITERIA}
    for name in names or list(METRICS):
        criterion, fn = METRICS[name]
        result = fn(ctx)
        if result is not None:
            out[criterion][name] = result
    return out
