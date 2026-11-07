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
  Direction comes only from verified per-arm means and the protocol's
  ``outcome_directions`` (lower or higher is better); a p value alone never gives a
  direction ("not extractable" instead).

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
from ..protocol import better_direction
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

def arm_pair(arms: list[dict]) -> tuple[dict, dict, str] | None:
    """(intervention arm, control arm, timepoint): the first timepoint with exactly one
    control arm and one other arm. None for multi-arm or unlabelled results."""
    by_tp: dict[str, list[dict]] = {}
    for a in arms:
        by_tp.setdefault((a.get("timepoint") or "").strip(), []).append(a)
    for tp, group in by_tp.items():
        ctrl = [a for a in group if CONTROL_ARM.search(a.get("arm") or "")]
        inter = [a for a in group if a not in ctrl]
        if len(ctrl) == 1 and len(inter) == 1:
            return inter[0], ctrl[0], tp
    return None


def study_outcome(study: dict, oid: str, better: str) -> dict:
    """One study's result for one outcome: direction and, if possible, Hedges' g."""
    out = {"label": study["label"], "direction": "not extractable", "g": None}
    e = study["extraction"]
    if e is None:
        return out | {"reason": "not extracted"}
    res = next((r for r in e.get("results") or [] if r["outcome"] == oid
                and r["status"] == "reported"), None)
    measure = (e["fields"].get(f"{oid}_measure") or {}).get("value")
    out["measure"] = measure
    if res is None:
        return out | {"reason": "no verified result"}
    out["estimates"] = [x for x in res.get("estimates") or []
                        if x.get("value") is not None or x.get("p_value")]
    pair = arm_pair(res.get("arms") or [])
    if pair is None:
        return out | {"reason": "no single intervention-control arm pair"}
    a_i, a_c, tp = pair
    out |= {"timepoint": tp, "intervention": {k: a_i.get(k) for k in ("arm", "n", "mean", "sd")},
            "control": {k: a_c.get(k) for k in ("arm", "n", "mean", "sd")}}
    if a_i.get("mean") is None or a_c.get("mean") is None:
        return out | {"reason": "per-arm means not verified"}
    diff = a_i["mean"] - a_c["mean"]
    if better == "unknown":
        out["reason"] = "direction of the scale unknown (protocol outcome_directions)"
    elif diff == 0:
        out["direction"] = "no difference"
    else:
        good = diff < 0 if better == "lower" else diff > 0
        out["direction"] = "favours intervention" if good else "favours control"
    nums = [a.get(k) for a in (a_i, a_c) for k in ("n", "sd")]
    if any(x is None for x in nums):
        return out | {"reason": out.get("reason") or "n or SD not verified: no effect size"}
    try:
        g, v = hedges_g(a_i["n"], a_i["mean"], a_i["sd"], a_c["n"], a_c["mean"], a_c["sd"])
    except ValueError as exc:
        return out | {"reason": f"effect size not computable: {exc}"}
    lo, hi = ci(g, v)
    return out | {"g": g, "var": v, "ci_low": lo, "ci_high": hi}


def synthesise_outcome(oid: str, outcome: str, better: str, studies: list[dict],
                       min_pool: int = MIN_POOL) -> dict:
    per = [study_outcome(s, oid, better) for s in studies]
    effects = [p for p in per if p["g"] is not None]
    votes = Counter(p["direction"] for p in per)
    fav_i, fav_c = votes["favours intervention"], votes["favours control"]
    group = {"id": oid, "outcome": outcome, "better": better, "per_study": per,
             "vote_count": {"favours intervention": fav_i, "favours control": fav_c,
                            "no difference": votes["no difference"],
                            "not extractable": votes["not extractable"],
                            "sign_test_p": sign_test(fav_i, fav_c)},
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
                bits.append(f"{i['arm']} n={_v(i['n'])} mean {_v(i['mean'])} SD "
                            f"{_v(i['sd'])} vs {c['arm']} n={_v(c['n'])} mean "
                            f"{_v(c['mean'])} SD {_v(c['sd'])}"
                            + (f" ({p['timepoint']})" if p.get("timepoint") else ""))
            for x in p.get("estimates") or []:
                est = f"{x['measure']}"
                if x.get("value") is not None:
                    est += f" {x['value']}"
                    if x.get("ci_low") is not None and x.get("ci_high") is not None:
                        est += f" (95% CI {x['ci_low']} to {x['ci_high']})"
                if x.get("p_value"):
                    est += f", p {x['p_value']}"
                bits.append(f"reported {est}")
            if p["g"] is not None:
                bits.append(f"Hedges' g {_f(p['g'])} (95% CI {_f(p['ci_low'])} to "
                            f"{_f(p['ci_high'])})")
            bits.append(f"direction: {p['direction']}"
                        + (f" ({p['reason']})" if p.get("reason") else ""))
            lines.append("  " + "; ".join(bits))
        v = g["vote_count"]
        lines.append(f"Vote count by direction of effect: favours intervention "
                     f"{v['favours intervention']}, favours control {v['favours control']}, "
                     f"no difference {v['no difference']}, not extractable "
                     f"{v['not extractable']}; sign test p {_p(v['sign_test_p'])}.")
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


class Narrative(BaseModel):
    sentences: list[Sentence] = Field(description="The synthesis, section by section, in "
                                      "reading order.")


SWIM_RULES = (
    "Write the synthesis following SWiM (Synthesis Without Meta-analysis, Campbell 2020) "
    "and PRISMA 2020:\n"
    "- One section per outcome (section = its id), then a section 'overall' (strength and "
    "limitations of the evidence).\n"
    "- For each outcome say: how studies were grouped, the metric (standardised mean "
    "difference, or direction of effect), the synthesis method (meta-analysis, or vote "
    "counting by direction of effect), the result, the heterogeneity (I2 / tau2, or how "
    "the studies differ: instruments, interventions, doses, populations) and the risk of "
    "bias of the studies behind the result.\n"
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


def narrative_prompt(protocol: dict, evidence: str, feedback: list[str]) -> str:
    fix = ""
    if feedback:
        fix = ("\nA critic found these problems in the previous draft. Fix every one:\n"
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


def render(protocol: dict, sentences: list[dict], studies: list[dict]) -> str:
    """Markdown: one heading per section, citations after each sentence, abstract-only
    studies marked with a dagger."""
    abstract = {s["label"] for s in studies if s["abstract_only"]}
    names = dict(outcomes(protocol)) | {"overall": "Overall"}
    parts = []
    for sec in list(names) + sorted({s["section"] for s in sentences} - set(names)):
        sents = [s for s in sentences if s["section"] == sec]
        if not sents:
            continue
        head = f"{sec}: {names[sec]}" if sec in names and sec != "overall" else names.get(
            sec, sec)
        body = " ".join(
            s["text"].rstrip(".") + " ["
            + ", ".join(c + (DAGGER if c in abstract else "") for c in s["studies"]) + "]."
            if s["studies"] else s["text"] for s in sents)
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
        groups = [synthesise_outcome(oid, o, better_direction(protocol, o), studies,
                                     min_pool) for oid, o in outcomes(protocol)]
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
            ans, u = llm.call_with_usage(Narrative, narrative_prompt(protocol, evidence,
                                                                     feedback),
                                         agent="synthesis")
            usage.append(u)
            sentences = check_narrative([s.model_dump() for s in ans.sentences], labels,
                                        evidence, sections)
        except Exception as exc:  # never crash the run: the critic will see no text
            flags.append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
            sentences = []
        checks = Counter(f.split(":")[0] for s in sentences for f in s["flags"])
        result = {
            "round": rnd,
            "studies": [{k: v for k, v in s.items() if k != "extraction"} for s in studies],
            "outcomes": groups, "diagnostic": dta, "evidence": evidence,
            "narrative": sentences, "text": render(protocol, sentences, studies),
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
