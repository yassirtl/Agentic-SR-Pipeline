"""Manuscript (Markdown), assembled by code from the state. No LLM call here: every number
comes from the stages' outputs, and the only LLM-written prose is the synthesis narrative,
which the synthesis checks and the critic have already judged.

Content rules (Part 9):
- a pooled result is printed with its 95% CI and, when k >= 3, its 95% prediction
  interval (``meta.prediction_interval``);
- a pooled result with I2 >= 75% is printed with the words "not interpretable as a single
  effect" (Cochrane Handbook 10.10: substantial heterogeneity);
- Methods list the screening changes made during development (frozen after Part 6d, see
  ``SCREENING_CHANGES``) and state what is reproducible: decisions are reproducible from
  the LLM cache; a fresh run without the cache can differ.
"""

import os
from collections import Counter
from pathlib import Path

from ..agents.extraction import outcomes
from ..agents.screening import criteria
from ..agents.synthesis import DAGGER, MIN_POOL
from ..agents.critic import MAX_LOOPS
from ..meta import expit, prediction_interval
from ..protocol import metric_order
from .prisma import flow_counts
from .summary import summary_markdown
from .tables import rob_rows, study_rows

I2_LIMIT = 0.75
NOT_INTERPRETABLE = "not interpretable as a single effect"

# Screening tuning log (docs/PLAN.md, "Screening freeze"): every change was made while
# looking at the T3 development sample (first 50 records of runs/20261107-160209-aec820).
SCREENING_CHANGES = [
    ("6", "Judge what the study did; background sentences are not evidence; year and "
          "language criteria never decide"),
    ("6", "Screener B excludes only on a clearly broken criterion; the adjudicator never "
          "excludes or leaves unresolved for missing information at title/abstract"),
    ("6", "At full text without a full text: no exclusion for what the abstract does not "
          "report"),
    ("6", "Eligibility never depends on results"),
    ("6", "Design criterion D = primary study (not reviews, meta-analyses, protocols)"),
    ("6b", "Evaluation runs use the gold review's criteria (topics.yaml), not the Planner's "
           "draft"),
    ("6b", "One verdict per criterion; deciding criterion chosen by code (D, I1.., E1..); a "
           "failure needs a verified quote"),
    ("6b", "Design guard: the adjudicator cannot include a verified D failure without a "
           "verified D 'yes'"),
    ("6b", "D as one positive sentence, same yes/no/unclear definition for both screeners, "
           "quote before the verdict, 'pools other studies -> D no'"),
    ("6b", "Output cap of 4,096 tokens per call (one call looped to 16,384)"),
    ("6b", "Contradicting-quote check: a D 'no' quoting a listed design without a "
           "non-primary word becomes unclear"),
    ("6c", "Criteria tagged full text never decide at title/abstract"),
    ("6c", "Multi-arm trials judged on the eligible arm pair (comparison field)"),
    ("6c", "Exclusions on different criteria go to the adjudicator; positive test sentences "
           "for exclusion criteria"),
    ("6d", "Title rule for D (protocol, systematic review, meta-analysis... where the title "
           "names the record type)"),
    ("6d", "Eligibility-list quotes: an exclusion 'yes' quoted from the study's own "
           "eligibility sentence becomes unclear"),
    ("6d", "Adjudicator model kept at gpt-4o-mini (gpt-4.1-mini tested: changes were worse)"),
]


# --- Small formatters -----------------------------------------------------------------------

def _f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def _pct(x):
    return f"{100 * x:.0f}%"


def _md_cell(x) -> str:
    return str("" if x is None else x).replace("|", "/").replace("\n", " ")


def md_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None._"
    head = "| " + " | ".join(h for _, h in columns) + " |"
    sep = "|" + "---|" * len(columns)
    body = ["| " + " | ".join(_md_cell(r.get(k)) for k, _ in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def pooled_sentence(m: dict, *, measure: str = "SMD (Hedges' g)", brief: bool = False) -> str:
    """Pooled continuous result with CI, prediction interval and the I2 rule."""
    pi = prediction_interval(m)
    out = (f"{measure} {_f(m['pooled'])} (95% CI {_f(m['ci_low'])} to {_f(m['ci_high'])}"
           + (f"; 95% prediction interval {_f(pi[0])} to {_f(pi[1])}" if pi else "")
           + f"), p {_f(m['p'], 3)}, k = {m['k']}; I2 {_pct(m['i2'])}, tau2 {_f(m['tau2'], 3)}.")
    if m["i2"] >= I2_LIMIT and brief:
        return out + f" With I2 >= 75% it is **{NOT_INTERPRETABLE}**."
    if m["i2"] >= I2_LIMIT:
        out += (f" With I2 >= 75% the pooled estimate is **{NOT_INTERPRETABLE}** (Cochrane "
                "Handbook 10.10); it is shown for transparency, the studies are described "
                "one by one below.")
    if pi and m["k"] == 3:
        out += (" With 3 studies the prediction interval has 1 degree of freedom, hence its "
                "width.")
    return out


def pooled_proportion_sentence(name: str, m: dict) -> str:
    """Pooled sensitivity or specificity (logit scale), with PI back-transformed."""
    pi = prediction_interval({**m, "pooled": m["logit_pooled"]})
    out = (f"{name} {_pct(m['pooled'])} (95% CI {_pct(m['ci_low'])} to {_pct(m['ci_high'])}"
           + (f"; 95% prediction interval {_pct(expit(pi[0]))} to {_pct(expit(pi[1]))}"
              if pi else "") + f"), k = {m['k']}, I2 {_pct(m['i2'])}.")
    if m["i2"] >= I2_LIMIT:
        out += f" With I2 >= 75% this pooled value is **{NOT_INTERPRETABLE}**."
    return out


def _rel(path: str | Path, base: Path) -> str:
    return os.path.relpath(Path(path), base)


def _cached_share(audit: list[dict]) -> tuple[int, int]:
    calls = cached = 0
    for e in audit:
        if "llm_calls" in e:
            calls += e["llm_calls"] or 0
            cached += e.get("llm_calls_cached") or 0
        elif "model" in e and "input_tokens" in e:
            calls += 1
            cached += int(bool(e.get("cached")))
    return calls, cached


# --- Sections -------------------------------------------------------------------------------

def _methods(state: dict, models: dict[str, str]) -> list[str]:
    p = state["protocol"]
    pico = p.get("pico") or {}
    strat = state.get("search_strategy") or {}
    ident = (state.get("prisma_counts") or {}).get("identification") or {}
    years = p.get("years") or {}
    out = ["## Methods", "", "### Protocol and eligibility criteria", "",
           f"Review question: {p.get('question', '')}", "",
           md_table([{"k": "Population", "v": pico.get("population")},
                     {"k": "Intervention / exposure", "v": pico.get("intervention")},
                     {"k": "Comparator", "v": pico.get("comparator")},
                     {"k": "Outcomes", "v": "; ".join(pico.get("outcomes") or [])}],
                    [("k", "PICO"), ("v", "")]), "",
           f"Study designs: {', '.join(p.get('study_designs') or []) or 'no restriction'}. "
           f"Years: {years.get('start') or 'any'} to {years.get('end_date') or years.get('end') or 'any'}. "
           f"Languages: {', '.join(p.get('languages') or []) or 'any'}. "
           f"Criteria source: {p.get('criteria_source') or 'Planner (LLM draft), approved'}.",
           ""]
    ft_only = set(p.get("full_text_criteria") or [])
    out += [f"- **{cid}** {text}" + (" *(full text only)*" if text in ft_only else "")
            for cid, text in criteria(p)]
    out += ["", "The protocol was drafted by the Planner agent and fixed before the search "
            "(human approval step unless the run used `--auto`)."]

    out += ["", "### Information sources and search strategy", "",
            f"Databases: {', '.join(p.get('databases') or [])}"
            + (f" ({p['database_reason']})" if p.get("database_reason") else "") + ". "
            "The LLM proposes the search concepts; code builds each database's Boolean query "
            "and design filter, checks the hit counts before any download, and refines at "
            "most twice.", ""]
    dbs = ident.get("databases") or {}
    rows = [{"db": db, "query": q, "searched": (dbs.get(db) or {}).get("searched_at"),
             "hits": (dbs.get(db) or {}).get("identified"),
             "dl": (dbs.get(db) or {}).get("downloaded"),
             "nr": (dbs.get(db) or {}).get("not_retrieved")}
            for db, q in (strat.get("queries") or {}).items()]
    out.append(md_table(rows, [("db", "Database"), ("searched", "Searched (UTC)"),
                               ("hits", "Hits"), ("dl", "Downloaded"),
                               ("nr", "Not downloaded"), ("query", "Query")]))
    attempts = strat.get("attempts") or []
    if len(attempts) > 1:
        out += ["", "Search attempts (hit counts before download; the chosen attempt was "
                f"{strat.get('chosen_attempt')}):", ""]
        out.append(md_table([{"a": a["attempt"],
                              "c": ", ".join(f"{d} {n}" for d, n in a["counts"].items()),
                              "s": ", ".join(f"{d} {s}" for d, s in a["status"].items()),
                              "r": a.get("reason")} for a in attempts],
                            [("a", "Attempt"), ("c", "Hits"), ("s", "Status"),
                             ("r", "Reason")]))

    out += ["", "### Selection process", "",
            "Records were deduplicated by code (record id, DOI, PMID, PMCID, then fuzzy title "
            "match with guards). Two LLM screeners (A strict, B lenient) judged every record "
            "criterion by criterion, each failure backed by a quote that code verifies in "
            "the text; disagreements went to an LLM adjudicator, unresolved records to a "
            "human (or were included with `--auto`). Full texts (open access only: PMC XML, "
            "open PDFs) were indexed (FAISS + BM25, LLM re-ranking) and the screeners read "
            "the best passages per criterion; papers without a full text were assessed on "
            "the abstract and flagged.", "",
            "Screening rules were tuned on a development sample of the T3 topic and then "
            "frozen (no change after seeing the held-out topics, except reported bug fixes). "
            "Changes made during tuning:", ""]
    out.append(md_table([{"p": pt, "c": c} for pt, c in SCREENING_CHANGES],
                        [("p", "Part"), ("c", "Change")]))

    order = ", ".join(x.replace("_", " ") for x in metric_order(p))
    out += ["", "### Data collection and risk of bias", "",
            "Fields were derived from the protocol (PICO, design, sample size, outcomes and "
            "results). A value was kept only with a quote found in the passages shown and "
            "every number of the value printed in that quote; otherwise it is `not "
            "reported` (values are never computed or invented). Risk of bias tool chosen by "
            "code from the verified design: RoB 2 (randomised trials, first protocol "
            "outcome), ROBINS-I (non-randomised studies), QUADAS-2 (diagnostic accuracy), "
            "else a 5-item checklist; a domain judgment needs a verified quote.", "",
            "### Synthesis methods", "",
            f"Continuous outcomes were pooled (random effects, DerSimonian-Laird; Hedges' g "
            f"from post-intervention means) only with at least {MIN_POOL} studies reporting "
            "verified per-arm n, mean and SD; the 95% prediction interval is given when "
            "k >= 3, and a pooled result with I2 >= 75% is reported as not interpretable "
            "as a single effect. Otherwise the synthesis follows SWiM (Campbell 2020): "
            "grouping by outcome, direction of effect as the standardised metric, vote "
            "counting with a sign test, heterogeneity described. Only between-group results "
            f"give a direction; the metric order was pre-specified ({order}), and a study "
            "whose metrics disagree is flagged. Diagnostic studies: sensitivity and "
            "specificity with Wilson 95% CIs; pooled (univariate logit random effects) "
            f"with at least {MIN_POOL} studies. A critic agent checked the text (code checks "
            f"plus an LLM pass); blockers sent it back to synthesis at most {MAX_LOOPS} "
            "times. An LLM objection counted as a blocker only when code confirmed the "
            "wrong value."]

    calls, cached = _cached_share(state.get("audit_log", []))
    used = sorted({f"{a}: {m}" for a, m in models.items()})
    out += ["", "### Automation and reproducibility", "",
            "Agents and models: " + ("; ".join(used) or "default model for every agent")
            + ". Temperature 0. Every LLM answer is cached on disk, keyed on (model, output "
            "schema, prompt): re-running the review with the same cache reproduces every "
            f"decision at no cost. In this run {cached} of {calls} LLM calls came from the "
            "cache. A fresh run without the cache can differ: the model is not "
            "deterministic across calls (seen during development: a re-extraction lost one "
            "study's table result and gained another's). Database answers are cached the "
            "same way; the search dates are those listed above."]
    return out


def _results(state: dict, base: Path, prisma_png: Path | None) -> list[str]:
    syn = state.get("synthesis") or {}
    pc = state.get("prisma_counts") or {}
    c = flow_counts(pc)
    ta, ft = pc.get("title_abstract") or {}, pc.get("full_text") or {}
    out = ["## Results", "", "### Study selection", ""]
    out.append(f"{c['identification']['n']} records were identified, "
               f"{c['removed']['lines'][1].strip()} and "
               f"{c['removed']['lines'][2].split(';')[0].strip()}) were removed before "
               f"screening, {c['screened']['n']} records were screened on title and "
               f"abstract and {c['ta_excluded']['n']} excluded. {c['sought']['n']} reports "
               f"were sought; {c['not_retrieved']['n']} had no open-access full text and "
               f"were assessed on the abstract. {c['ft_excluded']['n']} were excluded and "
               f"**{c['included']['n']} studies were included**. Screener agreement: "
               f"title/abstract kappa {ta.get('kappa', 'n/a')}, full text kappa "
               f"{ft.get('kappa', 'n/a')}.")
    if not c["consistent"]:
        bad = [k for k, ok in c["checks"].items() if not ok]
        out.append(f"\n**Count check failed:** {'; '.join(bad)}.")
    if prisma_png:
        out += ["", f"![PRISMA 2020 flow diagram]({_rel(prisma_png, base)})"]

    studies = study_rows(state)
    for r in studies:
        r["who"] = f"{r['label']} {r['author']} {r['year'] or ''}".strip()
        r["used"] = r["text_used"] + (f" {DAGGER}" if r["text_used"] == "abstract only" else "")
    out += ["", "### Study characteristics", "",
            md_table(studies, [("who", "Study"), ("design", "Design"),
                               ("country", "Country"), ("n_enrolled", "n"),
                               ("population", "Population"),
                               ("intervention", "Intervention"),
                               ("comparator", "Comparator"), ("used", "Text used")])]

    by_study: dict[str, list[str]] = {}
    for r in rob_rows(state):
        by_study.setdefault(r["label"], []).append(f"{r['domain']} {r['judgment']}")
    rob = [{"who": s["who"], "tool": s["rob_tool"], "overall": s["rob_overall"],
            "domains": "; ".join(by_study.get(s["label"], []))} for s in studies]
    levels = Counter(s["rob_overall"] for s in studies)
    out += ["", "### Risk of bias", "",
            "Overall: " + (", ".join(f"{k} {v}" for k, v in levels.most_common()) or "none")
            + ".", "", md_table(rob, [("who", "Study"), ("tool", "Tool"),
                                      ("overall", "Overall"), ("domains", "Domains")])]

    out += ["", "### Synthesis of results", ""]
    for g in syn.get("outcomes", []):
        out += [f"**{g['id']}: {g['outcome']}** (better = {g['better']} values). Method: "
                f"{g['method']} ({g['reason']}). Vote count: "
                + ", ".join(f"{k} {v if v is not None else 'n/a'}" if k != "sign_test_p"
                            else f"sign test p {_f(v, 3)}"
                            for k, v in g["vote_count"].items()) + ".", ""]
        if g.get("meta"):
            out += ["Pooled result: " + pooled_sentence(g["meta"]), ""]
        if g.get("plot"):
            out += [f"![Forest plot {g['id']}]({_rel(g['plot'], base)})", ""]
        if g.get("conflicts"):
            out += [f"Metric conflicts (direction depends on the metric): "
                    f"{', '.join(g['conflicts'])}.", ""]
    dta = syn.get("diagnostic")
    if dta:
        out += [f"**Diagnostic accuracy**: {dta['studies_with_2x2']} studies with a usable "
                "2x2 table.", ""]
        if dta.get("pooled"):
            out += [pooled_proportion_sentence("Sensitivity", dta["pooled"]["sens"]) + " "
                    + pooled_proportion_sentence("Specificity", dta["pooled"]["spec"]), ""]
        if dta.get("plot"):
            out += [f"![Paired forest plot]({_rel(dta['plot'], base)})", ""]
    text = (syn.get("text") or "_No synthesis text._").replace("\n### ", "\n#### ")
    if text.startswith("### "):
        text = "#" + text
    out += ["Narrative synthesis (written by the synthesis agent, every number and study "
            "label checked by code):", "", text]
    return out


def _qa(state: dict) -> list[str]:
    qa = state.get("qa_report") or {}
    if not qa:
        return ["## Quality assurance", "", "_No critic report._"]
    issues = qa.get("issues", [])
    out = ["## Quality assurance (critic)", "",
           f"Critic round {qa.get('round')} on synthesis round {qa.get('synthesis_round')}: "
           f"{'passed' if qa.get('passed') else 'not passed'}, {qa.get('blockers', 0)} "
           f"blockers, {qa.get('warnings', 0)} warnings; loops used {qa.get('loops_used')}/"
           f"{qa.get('max_loops')}." + (f" {qa['note']}." if qa.get("note") else ""), ""]
    for sev in ("blocker", "warning"):
        items = [i for i in issues if i["severity"] == sev]
        if not items:
            continue
        out += [f"{sev.capitalize()}s:", ""]
        for i in items:
            line = f"- `{i['kind']}`: {i['detail']}"
            if i["kind"] == "llm_unsupported_claim":
                line += (f" (LLM objection; evidence quote "
                         f"{'found' if i.get('evidence_verified') else 'not found'} in the "
                         f"tables; code check: {i.get('check', 'n/a')})")
            out.append(line)
        out.append("")
    return out


def _limitations(state: dict) -> list[str]:
    syn = state.get("synthesis") or {}
    qa = state.get("qa_report") or {}
    studies = syn.get("studies", [])
    abstract = [s["label"] for s in studies if s["abstract_only"]]
    p = state.get("protocol") or {}
    out = ["## Limitations", "",
           f"- Databases searched: {', '.join(p.get('databases') or [])} only (free "
           "sources); reviews that also search Embase, Cochrane or Scopus may find studies "
           "these do not index."]
    if abstract:
        out.append(f"- {len(abstract)} of {len(studies)} included studies had no open-access "
                   f"full text and were screened and extracted from the abstract only "
                   f"({', '.join(abstract)}, marked {DAGGER}).")
    for g in syn.get("outcomes", []):
        m = g.get("meta")
        if m and m["i2"] >= I2_LIMIT:
            out.append(f"- {g['id']}: I2 {_pct(m['i2'])} from {m['k']} studies; the pooled "
                       f"SMD is {NOT_INTERPRETABLE}.")
        if g.get("conflicts"):
            out.append(f"- {g['id']}: the direction of {', '.join(g['conflicts'])} depends on "
                       "the metric (post-intervention values vs change from baseline).")
        if g.get("against_majority"):
            out.append(f"- {g['id']}: {', '.join(g['against_majority'])} point(s) against the "
                       "majority direction; values were not corrected, a human should check "
                       "the source.")
    left = [i for i in qa.get("issues", []) if i["severity"] == "blocker"]
    if left:
        out.append(f"- {len(left)} critic blocker(s) remained after the loop budget: "
                   + "; ".join(i["kind"] for i in left) + ".")
    out += ["- Extraction values need a verified quote: correct values paraphrased by the "
            "model are refused and reported as `not reported` (conservative, lowers "
            "completeness).",
            "- Risk of bias (RoB 2) was assessed for the first protocol outcome only; "
            "diagnostic pooling is univariate (not the bivariate model).",
            "- Screening, extraction and synthesis are done by LLM agents with code checks; "
            "a human checked only the steps where the run paused for approval."]
    return out


def _references(state: dict) -> list[str]:
    papers = {p["paper_id"]: p for p in state.get("unique_records", [])}
    out = ["## Included studies", ""]
    for s in (state.get("synthesis") or {}).get("studies", []):
        p = papers.get(s["paper_id"], {})
        authors = p.get("authors") or []
        who = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
        out.append(f"- **{s['label']}** {who or s['author']} ({s['year'] or 'n.d.'}). "
                   f"{s['title'].rstrip('.')}. {p.get('venue') or ''}"
                   + (f" doi:{p['doi']}" if p.get("doi") else ""))
    return out


def manuscript(state: dict, *, base: Path, models: dict[str, str], summary: dict,
               prisma_png: Path | None = None) -> str:
    """Full Markdown text. ``base`` = folder of the .md file (image paths are relative)."""
    p = state.get("protocol") or {}
    syn = state.get("synthesis") or {}
    pc = state.get("prisma_counts") or {}
    c = flow_counts(pc)
    methods = "; ".join(f"{g['id']} {g['method']}" for g in syn.get("outcomes", []))
    main = next((g for g in syn.get("outcomes", []) if g.get("meta")), None)
    qa = state.get("qa_report") or {}
    abstract = [
        f"**Question.** {p.get('question', '')}",
        f"**Methods.** PRISMA 2020 review run by LLM agents with code checks; databases "
        f"{', '.join(p.get('databases') or [])}; designs "
        f"{', '.join(p.get('study_designs') or []) or 'not restricted'}.",
        f"**Results.** {c['identification']['n']} records identified, {c['screened']['n']} "
        f"screened, {c['included']['n']} studies included. Synthesis: {methods or 'none'}."
        + (f" {main['id']}: " + pooled_sentence(main["meta"], brief=True) if main else ""),
        f"**Quality check.** Critic {'passed' if qa.get('passed') else 'not passed'} "
        f"({qa.get('blockers', 0)} blockers, {qa.get('warnings', 0)} warnings).",
    ]
    oids = ", ".join(o for _, o in outcomes(p))
    parts = [f"# {p.get('title') or p.get('question', 'Systematic review')}", "",
             f"*Systematic review report, run {summary.get('run_id') or ''}. Outcomes: "
             f"{oids}.*", "", "## Abstract", "", "\n\n".join(abstract), "",
             *_methods(state, models), "", *_results(state, base, prisma_png), "",
             *_qa(state), "", *_limitations(state), "", "## Run summary", "",
             summary_markdown(summary), "", *_references(state), ""]
    return "\n".join(parts)
