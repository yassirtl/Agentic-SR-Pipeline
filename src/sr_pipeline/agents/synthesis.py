"""Synthesis: code does the maths, the LLM writes the text, code checks the text.

**Input**: the latest extraction and risk-of-bias items (``latest()``) of the papers in
``included``. Studies get short labels ``S1..Sn`` (order of ``included``); the LLM cites
labels, never DOIs.

**Per outcome (by code), path chosen by the data:**

- *continuous*: a study contributes an effect when its verified result has, at one
  timepoint, exactly one intervention arm and one control arm (control = arm name with
  control / usual care / waiting list / placebo / ...), each with verified n, mean and SD.
  Effect = Hedges' g (intervention minus control). With >= ``MIN_POOL`` (3) such studies:
  random-effects meta-analysis (DerSimonian-Laird, I^2, tau^2) and a forest plot. With
  fewer, no pooling: the outcome goes to narrative synthesis. "Comparable" = same outcome
  (the review's construct; the SMD puts different instruments on one scale) and same
  comparison type, which screening already fixed (every include passed the protocol's
  comparator).
- *diagnostic*: studies with a verified, complete and consistent 2x2 table (Part 8 step 0;
  the largest table if several): sensitivity and specificity with Wilson 95% CIs, paired
  forest plot; pooled (univariate logit random effects, a stated limitation versus the
  bivariate model) only with >= 3 studies.
- *everything else*: narrative synthesis following SWiM (Campbell 2020): grouping by
  outcome, standardised metric = direction of effect, synthesis method = vote counting by
  direction (with a sign test), heterogeneity described, risk of bias next to each study.
  A p value alone never gives a direction ("not extractable" instead).

**Direction of effect (Part 8b)**, with the protocol's ``outcome_directions`` (lower or
higher is better):

- only between-group results count: an estimate marked within-group (one arm before vs
  after) is shown but never used; any other one (including one the LLM labels
  between-group) counts only if its quote names a control arm and another arm
  (``estimate_role``);
- one metric per study, the first available in the protocol's ``direction_metric_order``
  (default: between-group difference with a stated contrast order > post-intervention
  means of both arms > change from baseline of both arms, printed or derived by code from
  verified baseline and post means). ``metric_used`` is recorded; ``metric_conflict`` when
  another available metric points the opposite way. The text must name conflicts and give
  a heterogeneity statement per outcome (checked by the critic). Pooling (Hedges' g) still
  uses post-intervention means only.

**Text** (``agent="synthesis"``): the LLM gets the evidence tables (formatted by code) and
returns sentences, each with the labels of the studies it is about. Code checks every
sentence (``check_narrative``): cited labels exist; every number is printed in the tables
shown (rounding allowed; small integers up to the number of studies count as study
counts); a sentence without a citation is flagged. Flags are not fixed here: the Critic
turns them into issues. Abstract-only studies carry a dagger in the rendered citations.

**Loop-back**: when the Critic found blockers in the previous round, they are added to the
prompt. This matters: the LLM cache is keyed on the prompt, so without the feedback a
re-run would return the same text.
"""

import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from ..forest import forest_plot, paired_forest
from ..llm import LLM
from ..meta import ci, dersimonian_laird, hedges_g, pooled_proportion, sign_test, wilson
from ..protocol import DEFAULT_METRIC_ORDER, better_direction, metric_order
from ..state import ReviewState, latest, stage_round
from .extraction import included_papers, numbers, outcomes
from .screening import _sum_usage

MIN_POOL = 3
DAGGER = "†"

CONTROL_ARM = re.compile(
    r"control|usual|wait|placebo|sham|standard care|attention|comparison|no[ -]exercise|"
    r"no[ -]intervention|no[ -]treatment|\bcon\b|\bcg\b|\bctrl\b", re.I)


# --- Studies -----------------------------------------------------------------------------

def first_author(paper: dict) -> str:
    authors = paper.get("authors") or []
    if not authors:
        return ""
    a = authors[0].strip()
    return a.split(",")[0].strip() if "," in a else a.split()[-1]


def study_list(state: ReviewState) -> list[dict]:
    """Included studies with labels, extraction and risk of bias (latest round, current
    includes only)."""
    papers = included_papers(state)
    ids = {p["paper_id"] for p in papers}
    ext = {e["paper_id"]: e for e in latest(state.get("extractions", []))
           if e["paper_id"] in ids}
    rob = {r["paper_id"]: r for r in latest(state.get("risk_of_bias", []))
           if r["paper_id"] in ids}
    out = []
    for i, p in enumerate(papers, 1):
        e, r = ext.get(p["paper_id"]), rob.get(p["paper_id"])
        out.append({
            "label": f"S{i}", "paper_id": p["paper_id"], "title": p.get("title") or "",
            "year": p.get("year"), "author": first_author(p),
            "abstract_only": bool(e and "abstract_only" in e["flags"]),
            "extracted": e is not None, "extraction": e,
            "design_category": e.get("design_category") if e else None,
            "design_verified": bool(e and e.get("design_category_verified")),
            "rob_tool": r["tool"] if r else None, "rob_overall": r.get("overall") if r else None,
            "rob_any_high": r.get("any_high") if r else None,
        })
    return out


# --- Continuous outcomes -------------------------------------------------------------------

# A timepoint that is the baseline, not a result (guard: the extraction prompt asks for
# baseline means inside the arm entry, but a model may still list a "baseline" arm).
BASELINE_TP = re.compile(r"baseline|\bpre(?:[- ]?(?:intervention|test|treatment))?\b|"
                         r"\bbefore\b", re.I)
# Difference measures: their sign gives a direction once the contrast order is known.
# Ratios (OR, RR, HR) and p values never do here.
DIFFERENCE = re.compile(r"differ|\bsmd\b|\bmd\b|hedges|cohen", re.I)
ARM_STOP = {"group", "groups", "arm", "the", "and", "with", "program", "programme"}

METRIC_NAMES = {"between_group_estimate": "between-group estimate",
                "post_means": "post-intervention means",
                "change_means": "change from baseline"}
OPPOSITE = {"favours intervention": "favours control",
            "favours control": "favours intervention"}


def arm_pair(arms: list[dict]) -> tuple[dict, dict, str] | None:
    """(intervention arm, control arm, timepoint): the first non-baseline timepoint with
    exactly one control arm and one other arm. None for multi-arm or unlabelled results."""
    by_tp: dict[str, list[dict]] = {}
    for a in arms:
        by_tp.setdefault((a.get("timepoint") or "").strip(), []).append(a)
    for tp, group in by_tp.items():
        if tp and BASELINE_TP.search(tp) and not re.search(r"post|after|follow", tp, re.I):
            continue
        ctrl = [a for a in group if CONTROL_ARM.search(a.get("arm") or "")]
        inter = [a for a in group if a not in ctrl]
        if len(ctrl) == 1 and len(inter) == 1:
            return inter[0], ctrl[0], tp
    return None


def names_arm(quote: str, arm: str | None) -> bool:
    """The quote names this arm: one of its words (3+ letters, not 'group'...) appears as a
    whole word."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", arm or "") if w.lower() not in ARM_STOP]
    return any(re.search(rf"\b{re.escape(w)}\b", quote or "", re.I) for w in words)


def estimate_role(est: dict, quote: str, arms: list[dict]) -> tuple[str, bool]:
    """(comparison, counts for direction). Part 8b fix 1: a within-group result (one arm
    before/after) never counts. Any other result counts only if its quote names a control
    arm and another arm. This applies to 'between_group' too: seen live, gpt-4o-mini
    labels every estimate between_group (8/8), e.g. "participants in the exercise group
    had decreased GDS-15 scores" (no control arm named). An unverified label is kept as
    claimed and shown, but the estimate does not count."""
    comp = est.get("comparison") or "unknown"
    if comp == "within_group":
        return comp, False
    ctrl = [a for a in arms if CONTROL_ARM.search(a.get("arm") or "")]
    other = [a for a in arms if a not in ctrl]
    both = (any(names_arm(quote, a.get("arm")) for a in ctrl)
            and any(names_arm(quote, a.get("arm")) for a in other))
    return comp, both


def _direction(diff: float, better: str) -> str | None:
    """Direction of an intervention-minus-control difference; None if the scale's good
    end is unknown."""
    if better not in ("lower", "higher"):
        return None
    if diff == 0:
        return "no difference"
    good = diff < 0 if better == "lower" else diff > 0
    return "favours intervention" if good else "favours control"


def direction_metrics(res: dict, pair: tuple | None, better: str,
                      estimates: list[dict]) -> dict[str, dict]:
    """Every metric that can give this study a direction (Part 8b fix 2), each with its
    direction or the reason it cannot: {metric: {"direction", "detail"} | {"reason"}}."""
    out: dict[str, dict] = {}
    # (1) Between-group effect estimate: counted, a difference measure, a value, and the
    # paper's contrast order (a bare "mean difference -2.1" has no known sign convention).
    usable = [e for e in estimates if e["counts"] and e.get("value") is not None
              and DIFFERENCE.search(e.get("measure") or "")]
    signed = [e for e in usable if e.get("contrast")]
    if signed:
        e = signed[0]
        diff = e["value"] if e["contrast"] == "intervention minus control" else -e["value"]
        out["between_group_estimate"] = {"direction": _direction(diff, better),
                                         "detail": f"{e['measure']} {e['value']:g} "
                                                   f"({e['contrast']})"}
    else:
        out["between_group_estimate"] = {"reason": (
            "difference without a stated contrast order" if usable else
            "no between-group difference with a value")}
    if pair is None:
        why = "no single intervention-control arm pair"
        out["post_means"] = {"reason": why}
        out["change_means"] = {"reason": why}
        return out
    a_i, a_c, _ = pair
    # (2) Post-intervention means of both arms.
    if a_i.get("mean") is not None and a_c.get("mean") is not None:
        out["post_means"] = {"direction": _direction(a_i["mean"] - a_c["mean"], better),
                             "detail": f"{a_i['mean']:g} vs {a_c['mean']:g}"}
    else:
        out["post_means"] = {"reason": "per-arm means not verified"}
    # (3) Change from baseline of both arms: printed, else derived by code from the
    # verified baseline and post means of the same arm (code arithmetic, labelled).
    changes = []
    for a in (a_i, a_c):
        if a.get("change_mean") is not None:
            changes.append((a["change_mean"], "printed"))
        elif a.get("baseline_mean") is not None and a.get("mean") is not None:
            changes.append((round(a["mean"] - a["baseline_mean"], 4), "derived"))
    if len(changes) == 2:
        (ci_, si), (cc, sc) = changes
        how = "printed" if si == sc == "printed" else "derived from baseline and post means"
        out["change_means"] = {"direction": _direction(ci_ - cc, better),
                               "detail": f"{ci_:g} vs {cc:g} ({how})",
                               "intervention": ci_, "control": cc}
    else:
        out["change_means"] = {"reason": "change from baseline not available in both arms"}
    return out


def study_outcome(study: dict, oid: str, better: str,
                  order: list[str] | tuple[str, ...] = tuple(DEFAULT_METRIC_ORDER)) -> dict:
    """One study's result for one outcome: direction (first available metric in the
    protocol's order, conflict flagged) and, if possible, Hedges' g (post means)."""
    out = {"label": study["label"], "direction": "not extractable", "g": None,
           "metric_used": None, "metric_conflict": False}
    e = study["extraction"]
    if e is None:
        return out | {"reason": "not extracted"}
    res = next((r for r in e.get("results") or [] if r["outcome"] == oid
                and r["status"] == "reported"), None)
    measure = (e["fields"].get(f"{oid}_measure") or {}).get("value")
    out["measure"] = measure
    if res is None:
        return out | {"reason": "no verified result"}
    arms = res.get("arms") or []
    ests = []
    for x in res.get("estimates") or []:
        if x.get("value") is None and not x.get("p_value"):
            continue
        comp, counts = estimate_role(x, res.get("quote") or "", arms)
        ests.append({**x, "comparison": comp, "counts": counts})
    out["estimates"] = ests
    pair = arm_pair(arms)
    if pair is not None:
        a_i, a_c, tp = pair
        keys = ("arm", "n", "mean", "sd", "baseline_mean", "change_mean", "change_sd")
        out |= {"timepoint": tp, "intervention": {k: a_i.get(k) for k in keys},
                "control": {k: a_c.get(k) for k in keys}}
    metrics = direction_metrics(res, pair, better, ests)
    out["metrics"] = metrics
    if better not in ("lower", "higher"):
        return out | {"reason": "direction of the scale unknown (protocol outcome_directions)"}
    available = [m for m in order if metrics.get(m, {}).get("direction")]
    if not available:
        return out | {"reason": "; ".join(f"{METRIC_NAMES[m]}: {metrics[m]['reason']}"
                                          for m in order if "reason" in metrics[m])}
    used = available[0]
    out |= {"direction": metrics[used]["direction"], "metric_used": used}
    against = [m for m in available[1:]
               if metrics[m]["direction"] == OPPOSITE.get(out["direction"])]
    if against:
        out["metric_conflict"] = True
        out["conflict"] = (f"{METRIC_NAMES[used]} {out['direction']}, but "
                           + ", ".join(f"{METRIC_NAMES[m]} {metrics[m]['direction']}"
                                       for m in against)
                           + f"; protocol order uses {METRIC_NAMES[used]}")
    if pair is None:
        return out
    a_i, a_c, _ = pair
    if a_i.get("mean") is None or a_c.get("mean") is None:
        return out
    nums = [a.get(k) for a in (a_i, a_c) for k in ("n", "sd")]
    if any(x is None for x in nums):
        return out | {"reason": "n or SD not verified: no effect size"}
    try:
        g, v = hedges_g(a_i["n"], a_i["mean"], a_i["sd"], a_c["n"], a_c["mean"], a_c["sd"])
    except ValueError as exc:
        return out | {"reason": f"effect size not computable: {exc}"}
    lo, hi = ci(g, v)
    return out | {"g": g, "var": v, "ci_low": lo, "ci_high": hi}


def synthesise_outcome(oid: str, outcome: str, better: str, studies: list[dict],
                       min_pool: int = MIN_POOL,
                       order: list[str] | tuple[str, ...] = tuple(DEFAULT_METRIC_ORDER)) -> dict:
    per = [study_outcome(s, oid, better, order) for s in studies]
    effects = [p for p in per if p["g"] is not None]
    votes = Counter(p["direction"] for p in per)
    fav_i, fav_c = votes["favours intervention"], votes["favours control"]
    # Strict majority of the studies with a direction; a study against it is shown to the
    # critic (warning: check the source; never corrected here).
    majority = ("favours intervention" if fav_i > fav_c else
                "favours control" if fav_c > fav_i else None)
    group = {"id": oid, "outcome": outcome, "better": better, "per_study": per,
             "metric_order": list(order),
             "vote_count": {"favours intervention": fav_i, "favours control": fav_c,
                            "no difference": votes["no difference"],
                            "not extractable": votes["not extractable"],
                            "sign_test_p": sign_test(fav_i, fav_c)},
             "metrics_used": dict(Counter(p["metric_used"] for p in per
                                          if p["metric_used"])),
             "conflicts": [p["label"] for p in per if p["metric_conflict"]],
             "majority": majority,
             "against_majority": [p["label"] for p in per if majority
                                  and p["direction"] == OPPOSITE[majority]],
             "effects": len(effects), "meta": None, "plot": None}
    if len(effects) >= min_pool:
        m = dersimonian_laird([p["g"] for p in effects], [p["var"] for p in effects])
        m["studies"] = [p["label"] for p in effects]
        group |= {"method": "meta-analysis", "meta": m,
                  "reason": f"{len(effects)} studies with per-arm n, mean and SD "
                            f"(>= {min_pool}): random-effects SMD"}
    else:
        group |= {"method": "narrative (SWiM)",
                  "reason": f"{len(effects)} study(ies) with per-arm n, mean and SD "
                            f"(< {min_pool} needed for a meta-analysis): vote counting by "
                            "direction of effect"}
    return group


# --- Diagnostic accuracy -------------------------------------------------------------------

def usable_two_by_two(tables: list[dict] | None) -> tuple[dict | None, str]:
    ok = [t for t in tables or [] if t.get("status") == "reported" and t.get("complete")
          and t.get("consistent") is not False and t["tp"] + t["fn"] > 0
          and t["tn"] + t["fp"] > 0]
    if ok:
        return max(ok, key=lambda t: t["tp"] + t["fp"] + t["fn"] + t["tn"]), ""
    if not tables:
        return None, "no 2x2 table reported"
    if any(t.get("consistent") is False for t in tables):
        return None, "2x2 cells disagree with the printed sensitivity/specificity"
    return None, "2x2 table incomplete (cells not printed, never derived)"


def synthesise_diagnostic(studies: list[dict], min_pool: int = MIN_POOL) -> dict | None:
    assessed = [s for s in studies if s["extraction"] and
                s["extraction"].get("diagnostic") is not None]
    if not assessed:
        return None
    per = []
    for s in assessed:
        tables = s["extraction"]["diagnostic"]
        t, why = usable_two_by_two(tables)
        if t is None:
            per.append({"label": s["label"], "reason": why, "tables": len(tables)})
            continue
        per.append({"label": s["label"], "index_test": t["index_test"],
                    "reference_standard": t["reference_standard"],
                    "threshold": t["threshold"], "tables": len(tables),
                    **{k: t[k] for k in ("tp", "fp", "fn", "tn")},
                    "sens": wilson(t["tp"], t["tp"] + t["fn"]),
                    "spec": wilson(t["tn"], t["tn"] + t["fp"])})
    usable = [p for p in per if "sens" in p]
    out = {"id": "DTA", "per_study": per, "studies_with_2x2": len(usable), "pooled": None,
           "plot": None}
    if len(usable) >= min_pool:
        ps = pooled_proportion([p["tp"] for p in usable], [p["tp"] + p["fn"] for p in usable])
        pp = pooled_proportion([p["tn"] for p in usable], [p["tn"] + p["fp"] for p in usable])
        out |= {"method": "univariate random-effects (logit), sensitivity and specificity "
                          "pooled separately",
                "pooled": {"sens": ps, "spec": pp, "studies": [p["label"] for p in usable]}}
    else:
        out["method"] = (f"per-study estimates only ({len(usable)} usable 2x2 table(s), "
                         f"< {min_pool} needed for pooling)")
    return out


# --- Evidence tables shown to the LLM ------------------------------------------------------

def _f(x: float | None, d: int = 2) -> str:
    return "NR" if x is None else f"{x:.{d}f}"


def _v(x: float | None) -> str:
    """An extracted number as printed (no rounding), NR if not verified."""
    return "NR" if x is None else f"{x:g}"


def _p(p: float | None) -> str:
    if p is None:
        return "NR"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _pct(t: tuple[float, float, float]) -> str:
    return f"{100 * t[0]:.1f}% (95% CI {100 * t[1]:.1f} to {100 * t[2]:.1f})"


def _field(e: dict | None, key: str, n: int = 80) -> str:
    v = ((e or {}).get("fields", {}).get(key) or {}).get("value") or "not reported"
    return v if len(v) <= n else v[:n - 3] + "..."


def _arm(a: dict) -> str:
    out = f"{a['arm']} n={_v(a['n'])} mean {_v(a['mean'])} SD {_v(a['sd'])}"
    if a.get("baseline_mean") is not None:
        out += f" (baseline mean {_v(a['baseline_mean'])})"
    if a.get("change_mean") is not None:
        out += f" (change {_v(a['change_mean'])}" + (
            f" SD {_v(a['change_sd'])})" if a.get("change_sd") is not None else ")")
    return out


def heterogeneity_line(g: dict) -> str:
    """Material for the heterogeneity statement (SWiM item 6), computed by code: the
    instruments used and the spread of directions; I2 / tau2 are on the meta-analysis
    line when there is one."""
    instruments: dict[str, list[str]] = {}
    for p in g["per_study"]:
        if p.get("measure"):
            instruments.setdefault(p["measure"].strip().rstrip("."), []).append(p["label"])
    dirs: dict[str, list[str]] = {}
    for p in g["per_study"]:
        dirs.setdefault(p["direction"], []).append(p["label"])
    return ("Heterogeneity material: instruments: "
            + ("; ".join(f"{m} ({', '.join(ls)})" for m, ls in instruments.items())
               or "not reported")
            + ". Directions: " + "; ".join(f"{d} {', '.join(ls)}" for d, ls in dirs.items())
            + "." + (f" Against the majority direction: {', '.join(g['against_majority'])}."
                     if g["against_majority"] else ""))


def evidence_text(studies: list[dict], groups: list[dict], dta: dict | None) -> str:
    rows = []
    for s in studies:
        e = s["extraction"]
        rob = (f"{s['rob_tool']}: {s['rob_overall']}" if s["rob_overall"]
               else f"{s['rob_tool'] or 'none'}" + (": any domain high" if s["rob_any_high"]
                                                    else ""))
        rows.append(" | ".join([
            s["label"], f"{s['author']} {s['year'] or ''}".strip() or "-",
            s["design_category"] or "not extracted", _field(e, "n_enrolled", 50),
            _field(e, "intervention"), _field(e, "intervention_dose"),
            _field(e, "comparator", 60),
            "abstract only" if s["abstract_only"] else "full text", f"risk of bias {rob}"]))
    # Counts the text may need are computed here, so the LLM copies them instead of
    # counting (seen live: "one study rated as high risk" when the table showed two).
    by_level: dict[str, list[str]] = {}
    for s in studies:
        level = (f"{s['rob_tool']} {s['rob_overall']}" if s["rob_overall"] else
                 f"{s['rob_tool'] or 'no tool'} " + ("any domain high" if s["rob_any_high"]
                                                     else "no overall judgment"))
        by_level.setdefault(level, []).append(s["label"])
    abstract = [s["label"] for s in studies if s["abstract_only"]]
    summary = [f"Number of studies: {len(studies)}.",
               "Risk of bias (overall): " + "; ".join(
                   f"{lvl} {len(ls)} ({', '.join(ls)})" for lvl, ls in by_level.items()) + ".",
               f"Text used: full text {len(studies) - len(abstract)}, abstract only "
               f"{len(abstract)}" + (f" ({', '.join(abstract)})" if abstract else "") + "."]
    lines = ["Studies (label | first author year | design | n enrolled | intervention | "
             "dose | comparator | text used | risk of bias):", *rows, *summary, ""]
    for g in groups:
        lines.append(f"Outcome {g['id']}: {g['outcome']} (better = {g['better']} values)")
        lines.append(f"Method: {g['method']}. {g['reason']}.")
        for p in g["per_study"]:
            bits = [f"{p['label']}: measure {p.get('measure') or 'not reported'}"]
            if p.get("intervention"):
                i, c = p["intervention"], p["control"]
                bits.append(f"{_arm(i)} vs {_arm(c)}"
                            + (f" ({p['timepoint']})" if p.get("timepoint") else ""))
            for x in p.get("estimates") or []:
                est = f"{x['measure']}"
                if x.get("value") is not None:
                    est += f" {x['value']}"
                    if x.get("ci_low") is not None and x.get("ci_high") is not None:
                        est += f" (95% CI {x['ci_low']} to {x['ci_high']})"
                if x.get("p_value"):
                    est += f", p {x['p_value']}"
                role = {"between_group": "between-group" if x["counts"] else
                        "between-group (claimed; quote does not name both arms)",
                        "within_group": "within-group (one arm before/after)"}.get(
                    x["comparison"], "comparison not stated")
                bits.append(f"reported {role} {est}" + ("" if x["counts"] else
                            " [not a between-group result: not used for direction]"))
            if p["g"] is not None:
                bits.append(f"Hedges' g {_f(p['g'])} (95% CI {_f(p['ci_low'])} to "
                            f"{_f(p['ci_high'])})")
            shown = [f"{METRIC_NAMES[m]} {d['direction']} ({d['detail']})"
                     for m, d in (p.get("metrics") or {}).items() if d.get("direction")]
            if shown:
                bits.append("metrics: " + ", ".join(shown))
            bits.append(f"direction: {p['direction']}"
                        + (f" (from {METRIC_NAMES[p['metric_used']]})"
                           if p.get("metric_used") else "")
                        + (f" ({p['reason']})" if p.get("reason") else ""))
            if p.get("metric_conflict"):
                bits.append(f"METRIC CONFLICT: {p['conflict']}")
            lines.append("  " + "; ".join(bits))
        v = g["vote_count"]
        lines.append(f"Direction metric order (protocol): "
                     + " > ".join(METRIC_NAMES[m] for m in g["metric_order"]) + "; used: "
                     + (", ".join(f"{METRIC_NAMES[m]} {k}" for m, k in g["metrics_used"].items())
                        or "none") + ".")
        lines.append(f"Vote count by direction of effect: favours intervention "
                     f"{v['favours intervention']}, favours control {v['favours control']}, "
                     f"no difference {v['no difference']}, not extractable "
                     f"{v['not extractable']}; sign test p {_p(v['sign_test_p'])}.")
        if g["conflicts"]:
            lines.append(f"Metric conflicts (direction depends on the metric): "
                         f"{', '.join(g['conflicts'])}.")
        lines.append(heterogeneity_line(g))
        if g["meta"]:
            m = g["meta"]
            lines.append(f"Random-effects meta-analysis ({m['k']} studies: "
                         f"{', '.join(m['studies'])}): SMD (Hedges' g) {_f(m['pooled'])} "
                         f"(95% CI {_f(m['ci_low'])} to {_f(m['ci_high'])}), p {_p(m['p'])}; "
                         f"heterogeneity I2 {100 * m['i2']:.0f}%, tau2 {_f(m['tau2'], 3)}, "
                         f"Q {_f(m['q'])} on {m['df']} df.")
        lines.append("")
    if dta:
        lines.append(f"Diagnostic accuracy: {dta['method']}.")
        for p in dta["per_study"]:
            if "sens" not in p:
                lines.append(f"  {p['label']}: no usable 2x2 ({p['reason']})")
                continue
            lines.append(f"  {p['label']}: {p['index_test']} vs {p['reference_standard']}"
                         + (f", threshold {p['threshold']}" if p["threshold"] else "")
                         + f"; TP {p['tp']}, FP {p['fp']}, FN {p['fn']}, TN {p['tn']}; "
                         f"sensitivity {_pct(p['sens'])}; specificity {_pct(p['spec'])}")
        if dta["pooled"]:
            ps, pp = dta["pooled"]["sens"], dta["pooled"]["spec"]
            lines.append(f"Pooled ({ps['k']} studies): sensitivity "
                         f"{_pct((ps['pooled'], ps['ci_low'], ps['ci_high']))}, I2 "
                         f"{100 * ps['i2']:.0f}%; specificity "
                         f"{_pct((pp['pooled'], pp['ci_low'], pp['ci_high']))}, I2 "
                         f"{100 * pp['i2']:.0f}%.")
        lines.append("")
    return "\n".join(lines)


# --- Narrative (LLM) and its check -----------------------------------------------------------

class Sentence(BaseModel):
    section: str = Field(description="Outcome id ('O1', 'O2', ...) or 'overall'.")
    text: str = Field(description="One sentence. Do not write study labels in it.")
    studies: list[str] = Field(description="Labels of the studies this sentence is about, "
                               "e.g. ['S1', 'S4']; every study it summarises.")


class Heterogeneity(BaseModel):
    outcome: str = Field(description="Outcome id ('O1', 'O2', ...).")
    text: str = Field(description="One or two sentences: I2 / tau2 when pooled; otherwise "
                      "the spread of directions and how the studies differ (population, "
                      "intervention dose, instrument, risk of bias). No study labels in it.")
    studies: list[str] = Field(description="Labels of the studies it is about.")


class Narrative(BaseModel):
    sentences: list[Sentence] = Field(description="The synthesis, section by section, in "
                                      "reading order.")
    heterogeneity: list[Heterogeneity] = Field(
        default_factory=list, description="Required: one entry per outcome (SWiM item 6).")


SWIM_RULES = (
    "Write the synthesis following SWiM (Synthesis Without Meta-analysis, Campbell 2020) "
    "and PRISMA 2020:\n"
    "- One section per outcome (section = its id), then a section 'overall' (strength and "
    "limitations of the evidence).\n"
    "- For each outcome say: how studies were grouped, the metric (standardised mean "
    "difference, or direction of effect and which result gave it), the synthesis method "
    "(meta-analysis, or vote counting by direction of effect), the result and the risk "
    "of bias of the studies behind the result.\n"
    "- Heterogeneity goes in the separate 'heterogeneity' list, one entry per outcome "
    "(required): I2 / tau2 when a meta-analysis was done; otherwise the spread of "
    "directions and how the studies differ (population, intervention dose, instrument, "
    "risk of bias), from the 'Heterogeneity material' line and the study table.\n"
    "- Only between-group results (intervention arm vs control arm) say anything about "
    "the effect. Results marked within-group or 'not used for direction' (one arm "
    "before vs after) must never be presented as an effect of the intervention.\n"
    "- For every study marked METRIC CONFLICT, write a sentence citing it that says its "
    "direction depends on the metric (e.g. post-intervention means vs change from "
    "baseline point in opposite directions) and which one the protocol order used.\n"
    "- Vote counting counts directions only. Never call an intervention effective, or a "
    "result significant, from individual p values; only the meta-analysis p value or the "
    "sign test p value printed below may be described as significant (p < 0.05).\n"
    "- Use only numbers printed in the tables below; never compute new ones.\n"
    "- Every sentence lists the studies it is about in 'studies' (labels only, never in "
    "the text); a summary sentence lists every study it summarises. Cite every study at "
    "least once.\n"
    "- Studies marked 'abstract only' were extracted from the abstract: say so in the "
    "overall section.\n"
    "- Do not judge study quality yourself: use the risk-of-bias column.\n"
)


def required_items(groups: list[dict]) -> list[str]:
    """Items code will check in this draft, spelled out per outcome and study (seen live:
    gpt-4o-mini skipped the general METRIC CONFLICT rule in 3 rounds out of 3)."""
    items = []
    for g in groups:
        m = g.get("meta")
        items.append(f"'heterogeneity' entry for {g['id']}"
                     + (f" giving I2 {100 * m['i2']:.0f}% and tau2 {m['tau2']:.3f}" if m
                        else " (spread of directions, how the studies differ)"))
        for p in g["per_study"]:
            if p.get("metric_conflict"):
                items.append(f"a sentence in section {g['id']} with studies ['{p['label']}'] "
                             f"saying its direction depends on the metric: {p['conflict']}")
    return items


def narrative_prompt(protocol: dict, evidence: str, feedback: list[str],
                     required: list[str] = ()) -> str:
    fix = ""
    if required:
        fix += ("\nRequired in this draft (checked by code):\n"
                + "\n".join(f"- {r}" for r in required) + "\n")
    if feedback:
        fix += ("\nA critic found these problems in the previous draft. Fix every one:\n"
                + "\n".join(f"- {f}" for f in feedback) + "\n")
    outs = "; ".join(f"{oid}: {o}" for oid, o in outcomes(protocol)) or "(none)"
    return (
        "You write the results synthesis of a PRISMA 2020 systematic review.\n"
        f"Review question: {protocol.get('question', '')}\nOutcomes: {outs}\n\n"
        f"{SWIM_RULES}{fix}\n{evidence}"
    )


_LABEL = re.compile(r"\bS\d+\b")
_NUM_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def number_supported(token: str, allowed: list[float]) -> bool:
    """A number written in the text is supported when it is printed in the tables, or is a
    rounding of a printed number (same or fewer decimals; integers must match exactly)."""
    x, d = float(token), _decimals(token)
    for a in allowed:
        if d == 0 and abs(a - x) < 1e-9:
            return True
        if d > 0 and abs(round(a, d) - x) < 1e-9:
            return True
    return False


def check_narrative(sentences: list[dict], labels: list[str], evidence: str,
                    sections: list[str]) -> list[dict]:
    """Sentences with flags: unknown_study, no_citation, number_not_supported,
    unknown_section."""
    # Label digits (S13) are not printed numbers; study counts 0..n are allowed (what they
    # count is for the critic to judge: a count is not checkable by matching numbers).
    allowed = (numbers(_LABEL.sub(" ", evidence)) + [float(i) for i in range(len(labels) + 1)]
               + [95.0])
    known = set(labels)
    out = []
    for s in sentences:
        text = (s.get("text") or "").strip()
        cites = [c.strip().upper() for c in s.get("studies") or []]
        cites += [c for c in _LABEL.findall(text) if c not in cites]
        cites = list(dict.fromkeys(cites))
        flags = []
        if s.get("section") not in sections:
            flags.append("unknown_section")
        bad = [c for c in cites if c not in known]
        if bad:
            flags.append("unknown_study: " + ", ".join(bad))
        if not [c for c in cites if c in known]:
            flags.append("no_citation")
        stripped = _LABEL.sub(" ", text)
        # Hyphenated instrument names (GDS-15, PHQ-9, SF-12) are names, not claims, but
        # their numbers are in the tables anyway; check everything written in digits.
        tokens = _NUM_TOKEN.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", stripped))
        unsupported = [t for t in tokens if not number_supported(t, allowed)]
        words = [n for n in numbers(re.sub(r"\d", " ", stripped))
                 if not number_supported(f"{n:g}", allowed)]
        unsupported += [f"{n:g}" for n in words]
        if unsupported:
            flags.append("number_not_supported: " + ", ".join(unsupported))
        out.append({"section": s.get("section"), "text": text,
                    "studies": [c for c in cites if c in known], "flags": flags})
    return out


def _cite(s: dict, abstract: set[str]) -> str:
    if not s["studies"]:
        return s["text"]
    return (s["text"].rstrip(".") + " ["
            + ", ".join(c + (DAGGER if c in abstract else "") for c in s["studies"]) + "].")


def render(protocol: dict, sentences: list[dict], studies: list[dict],
           heterogeneity: list[dict] = ()) -> str:
    """Markdown: one heading per section, citations after each sentence, abstract-only
    studies marked with a dagger, then the outcome's heterogeneity statement."""
    abstract = {s["label"] for s in studies if s["abstract_only"]}
    names = dict(outcomes(protocol)) | {"overall": "Overall"}
    parts = []
    for sec in list(names) + sorted({s["section"] for s in [*sentences, *heterogeneity]}
                                    - set(names)):
        sents = [s for s in sentences if s["section"] == sec]
        het = [h for h in heterogeneity if h["section"] == sec]
        if not sents and not het:
            continue
        head = f"{sec}: {names[sec]}" if sec in names and sec != "overall" else names.get(
            sec, sec)
        body = " ".join(_cite(s, abstract) for s in sents)
        if het:
            body += ("\n\n" if body else "") + "*Heterogeneity.* " + " ".join(
                _cite(h, abstract) for h in het)
        parts.append(f"### {head}\n\n{body}")
    if abstract:
        parts.append(f"{DAGGER} Data extracted from the abstract only (no open-access full "
                     "text).")
    return "\n\n".join(parts)


# --- Stage --------------------------------------------------------------------------------

def critic_feedback(state: ReviewState, rnd: int) -> list[str]:
    """Blocker issues of the critic that judged the previous synthesis round."""
    qa = state.get("qa_report") or {}
    if rnd <= 1 or qa.get("synthesis_round") != rnd - 1:
        return []
    return [f"{i['kind']}: {i['detail']}" for i in qa.get("issues", [])
            if i.get("severity") == "blocker"]


def make_synthesis(llm: LLM, *, out_dir: str | Path, min_pool: int = MIN_POOL):
    """Synthesis stage; forest plots are written to ``out_dir`` (``runs/<id>/figures``)."""
    out_dir = Path(out_dir)

    def synthesis(state: ReviewState) -> dict:
        protocol = state["protocol"]
        completed = state.get("completed_stages", [])
        rnd = stage_round("synthesis", completed)
        studies = study_list(state)
        order = metric_order(protocol)
        groups = [synthesise_outcome(oid, o, better_direction(protocol, o), studies,
                                     min_pool, order) for oid, o in outcomes(protocol)]
        dta = synthesise_diagnostic(studies, min_pool)

        author = {s["label"]: f"{s['label']} {s['author']} {s['year'] or ''}".strip()
                  for s in studies}
        for g in groups:
            if g["meta"]:
                eff = [p for p in g["per_study"] if p["g"] is not None]
                rows = [{"label": author[p["label"]], "est": p["g"], "lo": p["ci_low"],
                         "hi": p["ci_high"], "weight": w}
                        for p, w in zip(eff, g["meta"]["weights"])]
                m = g["meta"]
                g["plot"] = str(forest_plot(
                    rows, {"est": m["pooled"], "lo": m["ci_low"], "hi": m["ci_high"],
                           "label": f"Random effects (I2 {100 * m['i2']:.0f}%)"},
                    out_dir / f"forest_{g['id']}_r{rnd}.png",
                    xlabel="Hedges' g (intervention - control)", title=g["outcome"]))
        if dta and dta["studies_with_2x2"]:
            rows = [{"label": author[p["label"]], "sens": p["sens"], "spec": p["spec"]}
                    for p in dta["per_study"] if "sens" in p]
            pooled = None
            if dta["pooled"]:
                ps, pp = dta["pooled"]["sens"], dta["pooled"]["spec"]
                pooled = {"sens": (ps["pooled"], ps["ci_low"], ps["ci_high"]),
                          "spec": (pp["pooled"], pp["ci_low"], pp["ci_high"])}
            dta["plot"] = str(paired_forest(rows, pooled, out_dir / f"forest_DTA_r{rnd}.png",
                                            title="Diagnostic accuracy"))

        evidence = evidence_text(studies, groups, dta)
        feedback = critic_feedback(state, rnd)
        labels = [s["label"] for s in studies]
        sections = [oid for oid, _ in outcomes(protocol)] + ["overall"]
        usage, flags = [], []
        try:
            ans, u = llm.call_with_usage(Narrative, narrative_prompt(
                protocol, evidence, feedback, required_items(groups)), agent="synthesis")
            usage.append(u)
            sentences = check_narrative([s.model_dump() for s in ans.sentences], labels,
                                        evidence, sections)
            heterogeneity = check_narrative(
                [{"section": h.outcome.strip().upper(), "text": h.text,
                  "studies": h.studies} for h in ans.heterogeneity],
                labels, evidence, sections[:-1])
        except Exception as exc:  # never crash the run: the critic will see no text
            flags.append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
            sentences, heterogeneity = [], []
        checks = Counter(f.split(":")[0] for s in [*sentences, *heterogeneity]
                         for f in s["flags"])
        result = {
            "round": rnd,
            "studies": [{k: v for k, v in s.items() if k != "extraction"} for s in studies],
            "outcomes": groups, "diagnostic": dta, "evidence": evidence,
            "narrative": sentences, "heterogeneity": heterogeneity,
            "text": render(protocol, sentences, studies, heterogeneity),
            "checks": dict(checks), "flags": flags, "feedback": feedback,
        }
        return {
            "synthesis": result,
            "completed_stages": ["synthesis"],
            "audit_log": [{"node": "synthesis", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "synthesis", "studies": len(studies),
                           "methods": {g["id"]: g["method"] for g in groups}
                           | ({"DTA": dta["method"]} if dta else {}),
                           "effects": {g["id"]: g["effects"] for g in groups},
                           "metrics_used": {g["id"]: g["metrics_used"] for g in groups},
                           "metric_conflicts": {g["id"]: g["conflicts"] for g in groups},
                           "heterogeneity": len(heterogeneity),
                           "sentences": len(sentences), "checks": dict(checks),
                           "feedback_items": len(feedback), "flags": flags,
                           "plots": [g["plot"] for g in groups if g["plot"]]
                           + ([dta["plot"]] if dta and dta["plot"] else []),
                           **_sum_usage(usage)}],
        }

    return synthesis


def describe_synthesis(state: ReviewState) -> list[str]:
    """Line for the Coordinator summary."""
    s = state.get("synthesis")
    if not s:
        return []
    methods = ", ".join(f"{g['id']} {g['method']}" for g in s["outcomes"])
    return [f"Synthesis (round {s['round']}): {len(s['studies'])} studies; {methods}; "
            f"{len(s['narrative'])} sentences, checks {s['checks'] or 'clean'}"]
