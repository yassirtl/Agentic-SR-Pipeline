"""Data layer of the run viewer and the cost estimator (Part 11a).

Reads finished runs from disk only: no LLM, database or embedding call. Every number
shown by the viewer comes from a file the pipeline wrote:

- ``report/run_summary.json``: time, steps, LLM calls, tokens and billed cost per stage;
- ``report/prisma_counts.json``: PRISMA counts per stage;
- ``audit.jsonl``: coordinator decisions (proposal, outcome, guardrail reason, rationale);
- ``state.json``: completed stages (stage status) and screening quotes;
- ``report/tables/*.csv`` and ``evaluation/evaluation.json``: tables and gold funnel.

Billed vs fresh cost: ``cost_usd`` in the summary is what was **billed** (a call answered
from the LLM disk cache costs $0), but token counts include cached calls. The viewer shows
the billed cost (it must match ``run_summary.json``); the estimator re-prices all tokens with
``llm.PRICES`` (a new topic starts with an empty cache, so every call is paid).

Time per stage is not logged by the pipeline (audit entries are written together at the end
of a run), so the viewer shows the total time only and the estimator apportions time by
LLM calls (seconds per non-cached call, measured per run).
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..guardrail import STAGES, is_fresh
from ..llm import PRICES, compute_cost
from ..state import latest

STAGE_LABELS: dict[str, str] = {
    "planner": "Protocol",
    "search_strategy": "Search strategy",
    "retrieval": "Retrieval + dedup",
    "ta_screening": "Title/abstract screening",
    "ft_screening": "Full-text screening",
    "extraction": "Extraction + risk of bias",
    "synthesis": "Synthesis",
    "critic": "Critic (QA)",
    "report": "Report",
}

STAGE_AGENTS: dict[str, str] = {  # the agent whose model prices the stage's tokens
    "coordinator": "coordinator",
    "planner": "planner",
    "search_strategy": "search_strategy",
    "ta_screening": "screener_a",
    "ft_screening": "screener_a",
    "extraction": "extraction",
    "synthesis": "synthesis",
    "critic": "critic",
}

DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"


# --- Reading files -------------------------------------------------------------------

def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_audit(run_dir: Path) -> list[dict]:
    rows = []
    try:
        lines = (run_dir / "audit.jsonl").read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def read_state(run_dir: Path) -> dict | None:
    return _json(run_dir / "state.json")


# --- Run list --------------------------------------------------------------------------

@dataclass
class RunInfo:
    run_id: str
    path: Path
    status: str                      # finished / budget exhausted / incomplete
    topic: str | None = None
    question: str | None = None
    started_at: str | None = None
    screen_limit: int | None = None
    screened: int | None = None
    included: int | None = None
    cost_usd: float | None = None
    seconds: float | None = None
    steps: int | None = None
    models: dict = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in ("finished", "budget exhausted")


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "audit.jsonl").exists() or (path / "run.json").exists())


def run_info(path: Path) -> RunInfo:
    meta = _json(path / "run.json") or {}
    summary = _json(path / "report" / "run_summary.json")
    prisma = _json(path / "report" / "prisma_counts.json") or {}
    if summary is None:
        status = "incomplete"
    elif summary.get("budget_exhausted"):
        status = "budget exhausted"
    else:
        status = "finished"
    s = summary or {}
    ta = prisma.get("title_abstract") or {}
    ft = prisma.get("full_text") or {}
    return RunInfo(
        run_id=path.name, path=path, status=status,
        topic=meta.get("topic"),
        question=s.get("question") or meta.get("question"),
        started_at=s.get("started_at") or meta.get("started_at"),
        screen_limit=meta.get("screen_limit"),
        screened=ta.get("records_screened"),
        included=ft.get("studies_included"),
        cost_usd=(s.get("llm") or {}).get("cost_usd"),
        seconds=s.get("seconds"),
        steps=s.get("steps_taken"),
        models=meta.get("models") or {},
    )


def discover_runs(runs_dir: Path) -> list[RunInfo]:
    """Every run folder under ``runs_dir``, newest first. Non-run folders are skipped."""
    if not runs_dir.is_dir():
        return []
    runs = [run_info(p) for p in sorted(runs_dir.iterdir()) if is_run_dir(p)]
    return sorted(runs, key=lambda r: r.run_id, reverse=True)


# --- One run ---------------------------------------------------------------------------

@dataclass
class Run:
    info: RunInfo
    summary: dict
    prisma: dict
    audit: list[dict]
    state: dict | None
    evaluation: dict | None

    @property
    def path(self) -> Path:
        return self.info.path

    @property
    def completed(self) -> list[str]:
        if self.state and "completed_stages" in self.state:
            return list(self.state["completed_stages"])
        done = list(self.summary.get("stages_run") or [])
        if (self.path / "report").is_dir() and self.summary:
            done.append("report")       # the summary is written before report adds itself
        return done

    def table(self, name: str) -> list[dict]:
        return read_csv(self.path / "report" / "tables" / f"{name}.csv")


def load_run(path: Path, *, with_state: bool = True) -> Run:
    return Run(
        info=run_info(path),
        summary=_json(path / "report" / "run_summary.json") or {},
        prisma=_json(path / "report" / "prisma_counts.json") or {},
        audit=read_audit(path),
        state=read_state(path) if with_state else None,
        evaluation=_json(path / "evaluation" / "evaluation.json"),
    )


# --- Stage cards -----------------------------------------------------------------------

@dataclass
class StageCard:
    stage: str
    label: str
    status: str            # done / stale / not run
    runs: int
    refused: int           # times the coordinator proposed it and the guardrail refused
    count_in: str
    count_out: str
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def _n(value) -> str:
    return "–" if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def _stage_counts(run: Run) -> dict[str, tuple[str, str]]:
    p = run.prisma
    ident = p.get("identification") or {}
    ta = p.get("title_abstract") or {}
    ft = p.get("full_text") or {}
    proto = (run.state or {}).get("protocol") or _json(run.path / "report" / "protocol.json") or {}
    n_crit = len(proto.get("inclusion_criteria") or []) + len(proto.get("exclusion_criteria") or [])
    dbs = ident.get("databases") or {}
    by_node = {}
    for e in run.audit:
        if e.get("event"):
            by_node[e.get("node")] = e      # last entry per stage
    ext = by_node.get("extraction") or {}
    syn = by_node.get("synthesis") or {}
    cri = by_node.get("critic") or {}
    rep = by_node.get("report") or {}
    qa = (run.state or {}).get("qa_report") or {}
    passed = cri.get("passed", qa.get("passed"))
    files = rep.get("files") or ((run.state or {}).get("report") or {}).get("files") or {}
    return {
        "planner": ("question", f"{n_crit} criteria" if n_crit else "protocol"),
        "search_strategy": (f"{len(dbs)} databases" if dbs else "protocol",
                            f"{_n(ident.get('identified'))} hits"),
        "retrieval": (f"{_n(ident.get('identified'))} identified",
                      f"{_n(ident.get('records_screened'))} unique"),
        "ta_screening": (f"{_n(ta.get('records_screened'))} screened",
                         f"{_n(ta.get('included'))} included"),
        "ft_screening": (f"{_n(ft.get('reports_sought'))} sought",
                         f"{_n(ft.get('studies_included'))} included"),
        "extraction": (f"{_n(ft.get('studies_included'))} studies",
                       f"{_n(ext.get('studies'))} extracted"),
        "synthesis": (f"{_n(syn.get('studies'))} studies",
                      f"{_n(syn.get('sentences'))} sentences"),
        "critic": (f"{_n(cri.get('blockers'))} blockers · {_n(cri.get('warnings'))} warnings",
                   "passed" if passed else "–" if passed is None else "not passed"),
        "report": ("all stages fresh" if rep else "–",
                   f"{len(files)} files" if files else "–"),
    }


def stage_cards(run: Run) -> list[StageCard]:
    completed = run.completed
    per_node = ((run.summary.get("llm") or {}).get("per_node")) or {}
    refusals: dict[str, int] = {}
    for e in coordinator_steps(run):
        if e["outcome"] == "refused":
            refusals[e["proposed"]] = refusals.get(e["proposed"], 0) + 1
    counts = _stage_counts(run)
    cards = []
    for s in STAGES:
        runs = completed.count(s)
        status = "done" if is_fresh(s, completed) else "stale" if runs else "not run"
        u = per_node.get(s) or {}
        c_in, c_out = counts.get(s, ("–", "–"))
        cards.append(StageCard(
            stage=s, label=STAGE_LABELS.get(s, s), status=status, runs=runs,
            refused=refusals.get(s, 0), count_in=c_in, count_out=c_out,
            calls=u.get("calls", 0), cached=u.get("cached", 0),
            input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
            cost_usd=u.get("cost_usd", 0.0)))
    return cards


def coordinator_steps(run: Run) -> list[dict]:
    """One row per coordinator decision, in step order."""
    rows = []
    for e in run.audit:
        if e.get("node") != "coordinator" or "outcome" not in e:
            continue
        rows.append({
            "step": e.get("step"),
            "proposed": e.get("proposed"),
            "outcome": e.get("outcome"),
            "ran": e.get("substituted") or e.get("proposed"),
            "reason": e.get("reason") or e.get("error") or "",
            "rationale": e.get("rationale") or "",
            "cached": bool(e.get("cached")),
        })
    return rows


# --- Screening -------------------------------------------------------------------------

def _deciding_quote(d: dict) -> str:
    """Quote behind the final decision: the resolver's own quote, else the verdict quote
    of the deciding criterion."""
    src = d.get("adjudicator") if d.get("resolved_by") == "adjudicator" else d.get("a")
    src = src or d.get("a") or {}
    if src.get("quote"):
        return src["quote"]
    crit = d.get("criterion")
    for v in src.get("verdicts") or []:
        if v.get("criterion") == crit and v.get("quote"):
            return v["quote"]
    return ""


def screening_rows(run: Run, stage: str) -> list[dict]:
    """Latest decision per record for ``stage`` ("ta" or "ft")."""
    key = "ta_decisions" if stage == "ta" else "ft_decisions"
    items = latest(((run.state or {}).get(key)) or [])
    rows = []
    for d in items:
        a, b = d.get("a") or {}, d.get("b") or {}
        rows.append({
            "paper_id": d.get("paper_id"),
            "title": d.get("title"),
            "decision": d.get("final"),
            "criterion": d.get("criterion"),
            "resolved_by": d.get("resolved_by"),
            "screener_a": a.get("decision"),
            "screener_b": b.get("decision"),
            "quote": _deciding_quote(d),
            **({"full_text": d.get("fulltext_status")} if stage == "ft" else {}),
        })
    return rows


# --- Cost estimator --------------------------------------------------------------------

# What each stage's cost scales with.
DRIVERS = {
    "coordinator": "run", "planner": "run", "search_strategy": "run",
    "ta_screening": "screened", "ft_screening": "sought",
    "extraction": "included", "synthesis": "included", "critic": "included",
}
DRIVER_TEXT = {"run": "per run", "screened": "per record screened",
               "sought": "per full-text report sought", "included": "per included study"}


@dataclass
class RunRates:
    run_id: str
    topic: str | None
    screen_limit: int | None
    screened: int
    sought: int
    included: int
    seconds: float | None
    fresh_calls: int                         # calls not answered by the cache
    per_unit: dict[str, dict]                # stage -> calls / input / output per unit
    embed_tokens_per_sought: float | None    # None when every embedding came from cache
    billed_usd: float

    @property
    def ta_rate(self) -> float:
        return self.sought / self.screened if self.screened else 0.0

    @property
    def ft_rate(self) -> float:
        return self.included / self.sought if self.sought else 0.0

    @property
    def sec_per_call(self) -> float | None:
        if not self.seconds or not self.fresh_calls:
            return None
        return self.seconds / self.fresh_calls


def run_rates(run: Run) -> RunRates | None:
    """Per-unit rates measured on one finished run (None if the run did not finish)."""
    if not run.summary or not run.prisma:
        return None
    ta = run.prisma.get("title_abstract") or {}
    ft = run.prisma.get("full_text") or {}
    units = {"run": 1, "screened": ta.get("records_screened") or 0,
             "sought": ft.get("reports_sought") or 0,
             "included": ft.get("studies_included") or 0}
    llm = run.summary.get("llm") or {}
    per_unit = {}
    for stage, driver in DRIVERS.items():
        u = (llm.get("per_node") or {}).get(stage)
        if not u or not units[driver]:
            continue
        k = units[driver]
        per_unit[stage] = {"calls": u["calls"] / k, "input": u["input_tokens"] / k,
                           "output": u["output_tokens"] / k}
    emb = run.summary.get("embeddings") or {}
    embedded = emb.get("texts_embedded") or 0
    embed_rate = None
    if embedded and emb.get("input_tokens") and units["sought"]:
        per_text = emb["input_tokens"] / embedded
        embed_rate = per_text * (embedded + (emb.get("texts_cached") or 0)) / units["sought"]
    return RunRates(
        run_id=run.info.run_id, topic=run.info.topic, screen_limit=run.info.screen_limit,
        screened=units["screened"], sought=units["sought"], included=units["included"],
        seconds=run.summary.get("seconds"),
        fresh_calls=(llm.get("calls") or 0) - (llm.get("cached") or 0),
        per_unit=per_unit, embed_tokens_per_sought=embed_rate,
        billed_usd=llm.get("cost_usd") or 0.0)


@dataclass
class Range:
    low: float
    mid: float
    high: float

    def __add__(self, other: "Range") -> "Range":
        return Range(self.low + other.low, self.mid + other.mid, self.high + other.high)


def _range(values: list[float]) -> Range:
    if not values:
        return Range(0.0, 0.0, 0.0)
    return Range(min(values), statistics.median(values), max(values))


def _price(model: str, inp: float, out: float) -> float:
    return compute_cost(model, int(round(inp)), int(round(out))) or 0.0


@dataclass
class StageEstimate:
    stage: str
    driver: str
    model: str
    calls: Range
    cost: Range
    minutes: Range


def project(r: RunRates, *, limit: int, models: dict[str, str] | None = None,
            embed_fallback: float | None = None) -> dict:
    """What a fresh run screening ``limit`` records would cost if it behaved like ``r``.

    Volumes follow r's own include rates (screened -> sought -> included), calls and tokens
    follow r's per-unit rates, prices follow the chosen models, time follows r's seconds per
    non-cached call. Assumes the search finds at least ``limit`` records.
    """
    models = models or {}
    volume = {"run": 1.0, "screened": float(limit), "sought": limit * r.ta_rate,
              "included": limit * r.ta_rate * r.ft_rate}
    spc = r.sec_per_call or 0.0
    stages = {}
    for stage, driver in DRIVERS.items():
        m = r.per_unit.get(stage)
        if not m:
            continue
        model = models.get(stage, DEFAULT_MODEL)
        calls = m["calls"] * volume[driver]
        stages[stage] = {"calls": calls,
                         "cost": _price(model, m["input"], m["output"]) * volume[driver],
                         "minutes": calls * spc / 60}
    embed_rate = r.embed_tokens_per_sought or embed_fallback
    embed = _price(EMBED_MODEL, embed_rate, 0) * volume["sought"] if embed_rate else 0.0
    return {"run_id": r.run_id, "volume": volume, "stages": stages, "embeddings": embed,
            "cost": sum(x["cost"] for x in stages.values()) + embed,
            "calls": sum(x["calls"] for x in stages.values()),
            "minutes": sum(x["minutes"] for x in stages.values())}


def estimate(rates: list[RunRates], *, limit: int,
             models: dict[str, str] | None = None) -> dict:
    """Estimated calls, $ and minutes per stage for a fresh run screening ``limit`` records.

    Each source run is projected to ``limit`` on its own (``project``); every figure is then
    the low / median / high over those projections, so a range is always a scenario one of
    the finished runs actually lived through, never a mix of extremes from different runs.
    """
    if not rates:
        raise ValueError("no source runs selected")
    models = models or {}
    measured_embed = [r.embed_tokens_per_sought for r in rates if r.embed_tokens_per_sought]
    fallback = statistics.median(measured_embed) if measured_embed else None
    projections = [project(r, limit=limit, models=models, embed_fallback=fallback)
                   for r in rates]
    stages = []
    for stage, driver in DRIVERS.items():
        rows = [p["stages"][stage] for p in projections if stage in p["stages"]]
        if not rows:
            continue
        stages.append(StageEstimate(
            stage=stage, driver=driver, model=models.get(stage, DEFAULT_MODEL),
            calls=_range([x["calls"] for x in rows]),
            cost=_range([x["cost"] for x in rows]),
            minutes=_range([x["minutes"] for x in rows])))
    return {
        "stages": stages, "projections": projections,
        "volume": {k: _range([p["volume"][k] for p in projections])
                   for k in ("screened", "sought", "included")},
        "embeddings": _range([p["embeddings"] for p in projections]),
        "embed_runs": len(measured_embed),
        "cost": _range([p["cost"] for p in projections]),
        "calls": _range([p["calls"] for p in projections]),
        "minutes": _range([p["minutes"] for p in projections]),
    }


def priced_models() -> list[str]:
    return [m for m, (_, out) in PRICES.items() if out > 0]


def fresh_cost(run: Run) -> float:
    """What the run's LLM tokens would cost with an empty cache (embeddings excluded)."""
    models = run.info.models or {}
    total = 0.0
    for stage, u in (((run.summary.get("llm") or {}).get("per_node")) or {}).items():
        model = models.get(STAGE_AGENTS.get(stage, stage), DEFAULT_MODEL)
        total += compute_cost(model, u["input_tokens"], u["output_tokens"]) or 0.0
    return total
