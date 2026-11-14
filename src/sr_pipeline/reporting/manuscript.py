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
from ..agents.synthesis import DAGGER, I2_LIMIT, MIN_POOL, NOT_INTERPRETABLE
from ..agents.critic import MAX_LOOPS
from ..meta import expit, prediction_interval
from ..protocol import metric_order
from .prisma import flow_counts
from .summary import summary_markdown
from .tables import rob_rows, study_rows

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

def _methods(state: dict, models: dict[str, str], findings_spec: dict | None = None) -> list[str]:
    p = state["protocol"]
    pico = p.get("pico") or {}
    strat = state.get("search_strategy") or {}
    ident = (state.get("prisma_counts") or {}).get("identification") or {}
    years = p.get("years") or {}
    out = ["## Methods", "", "### Protocol and eligibility criteria", "",
           f"Review question: {p.get('question', '')}", ""]
    if not p.get("extraction_fields"):        # a clinical question: the PICO table
        out += [md_table([{"k": "Population", "v": pico.get("population")},
                          {"k": "Intervention / exposure", "v": pico.get("intervention")},
                          {"k": "Comparator", "v": pico.get("comparator")},
                          {"k": "Outcomes", "v": "; ".join(pico.get("outcomes") or [])}],
                         [("k", "PICO"), ("v", "")]), ""]
    else:
        out += ["Outcomes of interest: " + "; ".join(pico.get("outcomes") or []) + ".", ""]
    out += [f"Study designs: {', '.join(p.get('study_designs') or []) or 'no restriction'}. "
            f"Years: {_years(years)}. "
            f"Languages: {', '.join(p.get('languages') or []) or 'any'}. Eligibility "
            "criteria (the study must meet every inclusion criterion):", ""]
    ft_only = set(p.get("full_text_criteria") or [])
    out += [f"- **{cid}** {text}" + (" *(full text only)*" if text in ft_only else "")
            for cid, text in criteria(p)]
    out += ["", "The protocol (question, criteria, outcomes, databases) was drafted by an "
            "LLM planner and fixed before the search; it is published with this review "
            "(`protocol.json`). Where the criteria came from is given in Appendix A."]

    out += ["", "### Information sources and search strategy", "",
            f"Databases: {_dbs(p)}"
            + (f" ({p['database_reason'].strip().rstrip('.')})" if p.get("database_reason")
               else "") + ". "
            "The LLM proposes the search concepts; code builds each database's Boolean query "
            "and design filter, checks the hit counts before any download, and refines at "
            "most twice.", ""]
    dbs = ident.get("databases") or {}
    rows = [{"db": DB_NAMES.get(db, db), "query": q,
             "searched": (dbs.get(db) or {}).get("searched_at"),
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

    ta = (state.get("prisma_counts") or {}).get("title_abstract") or {}
    limit = ta.get("screen_limit")
    out += ["", "### Selection process", "",
            "Records were deduplicated by code (record id, DOI, PMID, PMCID, then fuzzy title "
            "match with guards). "
            + (f"Only the {limit} best-ranked unique records were screened (screening "
               "limit; rank = best position in any database's relevance order, so every "
               f"database contributes its top hits); the other {ta.get('not_screened', 0)} "
               "are reported in the flow diagram as not screened. " if limit else "")
            + "Automation tool (PRISMA 2020 'records marked as ineligible by automation "
            "tools'): before any LLM call, a code rule excludes on the design criterion D a "
            "record whose title names its own type as a review, meta-analysis, protocol, "
            "editorial or commentary (at the start of the title or after a colon or dash, "
            "e.g. '...: a systematic review'), and, when a criterion requires peer review, "
            "a record with an arXiv DOI (10.48550) is excluded on it; these records are "
            "counted before screening, "
            "not as screening exclusions. Two LLM screeners (A strict, B lenient) judged every record "
            "criterion by criterion, each failure backed by a quote that code verifies in "
            "the text; disagreements went to an LLM adjudicator, unresolved records to a "
            "human (or were included with `--auto`). Full texts (open access only: PMC XML, "
            "open PDFs) were indexed (FAISS + BM25, LLM re-ranking) and the screeners read "
            "the best passages per criterion; papers without a full text were assessed on "
            "the abstract and flagged. The screening rules were tuned on a separate "
            "development topic and frozen before this review (Appendix A).", ""]
    fields = p.get("extraction_fields") or {}
    if fields:
        out += ["### Data items", "",
                "For every included study the following items were extracted, each with "
                "the sentence that supports it:", ""]
        out += [f"- **{k.replace('_', ' ')}**: {v}" for k, v in fields.items()]

    order = ", ".join(x.replace("_", " ") for x in metric_order(p))
    out += ["", "### Data collection and risk of bias", "",
            "Fields were derived from the protocol (PICO, design, sample size, outcomes and "
            "results). A value was kept only with a quote found in the passages shown and "
            "every number of the value printed in that quote; otherwise it is `not "
            "reported` (values are never computed or invented). "
            + (f"Risk of bias was assessed for every study with the review's own instrument "
               f"({p.get('checklist_name') or 'review-specific checklist'}: "
               + "; ".join(f"{d['id']} {d['name']}" for d in p['checklist_domains'])
               + "); a domain without a verified quote is scored unclear."
               if p.get("checklist_domains") else
               "Risk of bias tool chosen by code from the verified design: RoB 2 "
               "(randomised trials, first protocol outcome), ROBINS-I (non-randomised "
               "studies), QUADAS-2 (diagnostic accuracy), else a 5-item checklist; a domain "
               "judgment needs a verified quote."), "",
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

    if p.get("extraction_fields") and findings_spec:
        out += ["", "Because the studies report different measures on different datasets, "
                "no effect was pooled across them. The studies were mapped by "
                f"{findings_spec['rows'].get('label', 'row').lower()} and "
                f"{findings_spec['cols'].get('label', 'column').lower()} (evidence map): each "
                "value was put in the first group whose keyword it contains, a study whose "
                "value was not reported was classified from its title, and the rest are "
                "shown as Other or Not reported. Numeric items (e.g. latency, power) are "
                "shown only when the extracted value contains a number."]
    models_used = sorted(set(models.values())) or ["the default model"]
    out += ["", "### Use of automation tools", "",
            "Every step after the protocol was carried out by LLM agents "
            f"({', '.join(models_used)}, temperature 0) under code checks: the search, "
            "screening (two screeners and an adjudicator), extraction, risk of bias, the "
            "synthesis and a critic. A person approved the protocol and settled the "
            "disagreements the adjudicator could not, unless the run was fully automatic. "
            "Agents, models, cache use and cost are listed in Appendix A."]
    return out


DB_NAMES = {"openalex": "OpenAlex", "pubmed": "PubMed", "arxiv": "arXiv"}


def _dbs(p: dict) -> str:
    return ", ".join(DB_NAMES.get(d, d) for d in p.get("databases") or [])


def _years(years: dict) -> str:
    start = years.get("start")
    end = years.get("end_date") or years.get("end")
    text = (f"{start} to {end}" if start and end else f"{start} onward" if start
            else f"up to {end}" if end else "no restriction")
    return text + (f" (last {years['recent']} years, the template default)"
                   if years.get("recent") else "")


def _reproducibility(state: dict, models: dict[str, str]) -> list[str]:
    calls, cached = _cached_share(state.get("audit_log", []))
    used = sorted({f"{a}: {m}" for a, m in models.items()})
    return ["Agents and models: " + ("; ".join(used) or "default model for every agent")
            + ". Temperature 0. Every LLM answer is cached on disk, keyed on (model, output "
            "schema, prompt): re-running the review with the same cache reproduces every "
            f"decision at no cost. In this run {cached} of {calls} LLM calls came from the "
            "cache. A fresh run without the cache can differ: the model is not "
            "deterministic across calls (seen during development: a re-extraction lost one "
            "study's table result and gained another's). Database answers are cached the "
            "same way; the search dates are those listed in the Methods."]


def _results(state: dict, base: Path, prisma_png: Path | None, fx: dict | None = None) -> list[str]:
    fx = fx or {}
    syn = state.get("synthesis") or {}
    pc = state.get("prisma_counts") or {}
    c = flow_counts(pc, state.get("protocol"))
    n = c["numbers"]
    ta, ft = pc.get("title_abstract") or {}, pc.get("full_text") or {}
    out = ["## Results", "", "### Study selection", ""]
    removed = [f"{n['duplicates']} duplicates",
               f"{n['automation']} records marked as ineligible by automation tools "
               "(title rule, arXiv rule)",
               f"{n['not_downloaded']} records not downloaded (search ceiling)"]
    if n["screen_limit"] or n["not_screened"]:
        removed.append(f"{n['not_screened']} records above the screening limit of "
                       f"{n['screen_limit']} (not screened)")
    out.append(f"{n['identified']} records were identified; "
               f"{', '.join(removed[:-1])} and {removed[-1]} were removed before "
               f"screening. {n['screened']} records were screened on title and "
               f"abstract and {n['ta_excluded']} excluded. {n['sought']} reports "
               f"were sought; {n['not_retrieved']} had no open-access full text and "
               f"were assessed on the abstract. {n['ft_excluded']} were excluded and "
               f"**{n['included']} studies were included**. Screener agreement: "
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
    out += ["", "### Study characteristics", ""]
    if fx.get("table"):
        rows, cols = fx["table"]
        out += ["Values as extracted, each backed by a sentence of the paper (see "
                "`tables/extraction.csv`); – = not reported (numeric items: no number "
                "given).", "", md_table(rows, cols)]
    else:
        out.append(md_table(studies, [("who", "Study"), ("design", "Design"),
                                      ("country", "Country"), ("n_enrolled", "n"),
                                      ("population", "Population"),
                                      ("intervention", "Intervention"),
                                      ("comparator", "Comparator"), ("used", "Text used")]))

    em = fx.get("em")
    if em:
        from .findings import evidence_map_rows
        src_r, src_c = em["sources"]["rows"], em["sources"]["cols"]
        rows, cols = evidence_map_rows(em)
        out += ["", "### Evidence map", "",
                f"Each of the {em['n']} included studies is counted once, by "
                f"{em['row_label'].lower()} and {em['col_label'].lower()}. "
                f"{em['row_label']}: from the extracted value for {src_r.get('value', 0)} "
                f"studies, from the title for {src_r.get('title', 0)}; "
                f"{em['col_label'].lower()}: from the value for {src_c.get('value', 0)}, "
                f"from the title for {src_c.get('title', 0)}."]
        if fx.get("map_png"):
            out += ["", f"![Evidence map]({_rel(fx['map_png'], base)})"]
        out += ["", md_table(rows, cols)]
    if fx.get("sof"):
        out += ["", "### Summary of findings", ""] + [f"- {x}" for x in fx["sof"]]
    if fx.get("gaps"):
        from .findings import check_text
        out += ["", "### Evidence gaps", "",
                "Found by code in the included studies: combinations of model and device "
                "never seen together, groups with one or two studies, items rarely measured "
                "or rarely done well, and dependence on one dataset. A combination or group "
                "gap was checked by a hit count in OpenAlex (title and abstract, the review's "
                "years); a large count means the gap may be one of this sample only.", "",
                md_table([{"id": g["id"], "s": g["statement"], "e": g["evidence"],
                           "c": check_text(g), "cl": ", ".join(g["closest"]) or "–"}
                          for g in fx["gaps"]],
                         [("id", "Gap"), ("s", "What is missing"), ("e", "Evidence"),
                          ("c", "Check"), ("cl", "Closest studies")])]

    by_study: dict[str, list[str]] = {}
    per_domain: dict[str, Counter] = {}
    for r in rob_rows(state):
        by_study.setdefault(r["label"], []).append(f"{r['domain']} {r['judgment']}")
        per_domain.setdefault(r["domain"], Counter())[r["judgment"]] += 1
    levels = Counter(s["rob_overall"] for s in studies)
    names = {d["id"]: d["name"] for d in (state.get("protocol") or {}).get(
        "checklist_domains") or []}
    out += ["", "### Risk of bias", "",
            "Overall: " + (", ".join(f"{k} {v}" for k, v in levels.most_common()) or "none")
            + ". Per-study judgements are in Appendix B."]
    if per_domain:
        out += ["", md_table([{"d": f"{d} {names.get(d, '')}".strip(),
                               "low": c.get("low", 0), "unclear": c.get("unclear", 0),
                               "high": c.get("high", 0)} for d, c in per_domain.items()],
                             [("d", "Domain"), ("low", "Low"), ("unclear", "Unclear"),
                              ("high", "High")])]

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
    out = ["### Limitations", "",
           f"- Databases searched: {_dbs(p)} only (free "
           "sources); reviews that also search Embase, Cochrane or Scopus may find studies "
           "these do not index."]
    ta = (state.get("prisma_counts") or {}).get("title_abstract") or {}
    if ta.get("not_screened"):
        out.append(f"- Screening limit: only the {ta['screen_limit']} best-ranked of "
                   f"{ta['records_available']} unique records were screened; relevant "
                   f"studies among the other {ta['not_screened']} were not considered.")
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
            ("- Risk of bias used the review's own instrument; a domain without a verified "
             "quote is scored unclear, so 'unclear' often means 'not reported in the text "
             "read'." if p.get("checklist_domains") else
             "- Risk of bias (RoB 2) was assessed for the first protocol outcome only; "
             "diagnostic pooling is univariate (not the bivariate model)."),
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


def _rob_per_study(state: dict) -> list[str]:
    studies = study_rows(state)
    by_study: dict[str, list[str]] = {}
    for r in rob_rows(state):
        by_study.setdefault(r["label"], []).append(f"{r['domain']} {r['judgment']}")
    rob = [{"who": f"{s['label']} {s['author']} {s['year'] or ''}".strip(),
            "tool": s["rob_tool"], "overall": s["rob_overall"],
            "domains": "; ".join(by_study.get(s["label"], []))} for s in studies]
    return ["## Appendix B. Risk of bias per study", "",
            md_table(rob, [("who", "Study"), ("tool", "Tool"), ("overall", "Overall"),
                           ("domains", "Domains")])]


def _appendix(state: dict, models: dict[str, str], summary: dict, writer: dict | None) -> list[str]:
    p = state.get("protocol") or {}
    out = ["## Appendix A. How this review was produced", "",
           f"Run `{summary.get('run_id') or ''}`. Criteria source: "
           f"{p.get('criteria_source') or 'Planner (LLM draft), approved'}. The protocol "
           "was approved by a person unless the run used `--auto`.", "",
           "### Screening rules", "",
           "The screening rules were tuned on a development topic (exercise and depression) "
           "and then frozen (no change after seeing the evaluation topics, except reported "
           "bug fixes). Changes made during tuning:", "",
           md_table([{"p": pt, "c": c} for pt, c in SCREENING_CHANGES],
                    [("p", "Part"), ("c", "Change")]), "",
           "### Automation and reproducibility", "", *_reproducibility(state, models)]
    if writer:
        ai = [k.replace("_", " ") for k, v in writer["source"].items() if v == "ai"]
        code = [k.replace("_", " ") for k, v in writer["source"].items() if v == "code"]
        line = ("Drafted by the writer agent from a facts sheet built by code, every number "
                "and study label checked: " + (", ".join(ai) or "none") + ". Written by code: "
                + (", ".join(code) or "none") + ".")
        if writer.get("flags"):
            line += " Sections replaced by the code text after a failed check: " + "; ".join(
                f"{k.replace('_', ' ')} ({'; '.join(v)})" for k, v in writer["flags"].items()) + "."
        if writer.get("error"):
            line += f" Writer call failed: {writer['error']}."
        out += ["", "### Writing", "", line]
    qa = _qa(state)
    qa[0] = "### " + qa[0].lstrip("# ")
    out += ["", *qa, "", "### Run summary", "", summary_markdown(summary)]
    return out


def _other_information() -> list[str]:
    return ["## Other information", "",
            "- **Registration and protocol.** Not registered. The protocol is published with "
            "the review (`protocol.json`); where its criteria came from is in Appendix A.",
            "- **Support and competing interests.** [to be completed by the authors]",
            "- **Availability of data.** Evidence tables, extracted values with their "
            "quotes, screening decisions and the BibTeX file are published with the review "
            "(`tables/`, `references.bib`).",
            "- **Use of AI.** This draft was produced by LLM agents under code checks; every "
            "number in the text comes from the tables. A reviewer must check it before use."]


def _prior_reviews(p: dict) -> str | None:
    """What the topic says about existing reviews (topics.yaml ``review``), if anything."""
    if not p.get("topic"):
        return None
    try:
        from ..evaluation.topics import load_topic
        review = load_topic(str(p["topic"])).get("review")
    except (KeyError, FileNotFoundError, ValueError, OSError):
        return None
    if not review:
        return None
    import re
    return re.split(r",\s*(?:doi|PMC)", str(review).strip())[0]


def facts_sheet(state: dict, fx: dict | None = None) -> tuple[str, dict]:
    """What the writer may use (text) and the numbers the code text needs (dict). Built
    by code from the state: counts, summary of findings or synthesis, risk of bias,
    limitations. Nothing else reaches the writer."""
    fx = fx or {}
    p = state.get("protocol") or {}
    pc = state.get("prisma_counts") or {}
    n = flow_counts(pc, p)["numbers"]
    syn = state.get("synthesis") or {}
    studies = syn.get("studies", [])
    lines = [f"Question: {p.get('question', '')}",
             f"Databases: {_dbs(p)}; years: "
             f"{_years(p.get('years') or {})}.",
             f"Records identified: {n['identified']}; duplicates removed: "
             f"{n['duplicates']}; screened on title and abstract: {n['screened']}"
             + (f" (the {n['screen_limit']} most relevant; {n['not_screened']} less "
                "relevant records were not screened)" if n.get("not_screened") else "")
             + f"; full texts sought: {n['sought']}; included studies: {n['included']}.",
             f"Included studies judged on the abstract only (no open-access full text): "
             f"{sum(1 for s in studies if s.get('abstract_only'))} of {len(studies)}."]
    prior = _prior_reviews(p)
    if prior:
        lines.append(f"Prior systematic reviews on this question: {prior}.")
    if fx.get("sof"):
        lines += ["Summary of findings:"] + [f"- {x}" for x in fx["sof"]]
    gaps = fx.get("gaps") or []
    if gaps:
        from .findings import check_text
        lines.append("Evidence gaps (found by code; 'propose' = write a proposal for it):")
        for g in gaps:
            g["propose"] = not (g.get("check_result") or {}).get("confirmed") is False
            lines.append(f"- {g['id']} [{'propose' if g['propose'] else 'not confirmed'}] "
                         f"{g['statement']}. Evidence: {g['evidence']}. Check: "
                         f"{check_text(g)}. Closest included studies: "
                         f"{', '.join(g['closest']) or 'none'}.")
    else:
        for g in syn.get("outcomes", []):
            vc = ", ".join(f"{k} {v}" for k, v in g["vote_count"].items()
                           if k != "sign_test_p" and v is not None)
            lines.append(f"Outcome {g['outcome']}: {g['method']}; vote count {vc}."
                         + (" Pooled: " + pooled_sentence(g["meta"], brief=True)
                            if g.get("meta") else ""))
    levels = Counter(s.get("rob_overall") for s in studies)
    lines.append("Risk of bias, overall: " + ", ".join(f"{k} {v}" for k, v in
                                                      levels.most_common()) + ".")
    lines += ["Limitations of this review:"] + [x for x in _limitations(state)[2:] if x]
    d = {"identified": n["identified"], "screened": n["screened"], "included": n["included"],
         "gaps": gaps,
         "headline": next((x for x in fx.get("sof") or [] if "accuracy" in x.lower()), "")}
    return "\n".join(lines), d


def _paras(x) -> list[str]:
    items = x if isinstance(x, list) else [x]
    out = []
    for t in items:
        if t:
            out += [t, ""]
    return out[:-1] if out else []


def manuscript(state: dict, *, base: Path, models: dict[str, str], summary: dict,
               prisma_png: Path | None = None, fx: dict | None = None,
               writer: dict | None = None) -> str:
    """Full Markdown text. ``base`` = folder of the .md file (image paths are relative).
    ``fx``: findings (study table, evidence map, summary of findings; ``findings.py``);
    ``writer``: the writer's sections (``writer.py``), code text when absent."""
    fx = fx or {}
    p = state.get("protocol") or {}
    syn = state.get("synthesis") or {}
    pc = state.get("prisma_counts") or {}
    c = flow_counts(pc, p)
    methods = "; ".join(f"{g['id']} {g['method']}" for g in syn.get("outcomes", []))
    main = next((g for g in syn.get("outcomes", []) if g.get("meta")), None)
    sec = (writer or {}).get("sections") or {}
    name = p.get("checklist_name") or "review's own instrument"
    tool = ("the " + name[:1].lower() + name[1:]) if p.get(
        "checklist_domains") else "RoB 2, ROBINS-I, QUADAS-2 or a checklist, by design"
    synth = ("narrative synthesis with an evidence map (no pooling)"
             if p.get("extraction_fields") else f"synthesis by outcome ({methods or 'none'})")
    abstract = [
        f"**Background.** {sec.get('abstract_background') or p.get('question', '')}",
        f"**Methods.** We searched {_dbs(p)} "
        f"({_years(p.get('years') or {})}). LLM agents screened titles, abstracts and "
        "full texts against pre-specified criteria (two screeners and an adjudicator, every "
        "exclusion backed by a verified quote), extracted data with verified quotes and "
        f"assessed risk of bias with {tool}; {synth}.",
        f"**Results.** {sec.get('abstract_results') or ''}".rstrip()
        + ("" if sec.get("abstract_results") else
           f" {c['identification']['n']} records identified, {c['screened']['n']} screened, "
           f"{c['included']['n']} studies included."
           + (f" {main['id']}: " + pooled_sentence(main["meta"], brief=True) if main else "")),
        f"**Conclusions.** {sec.get('abstract_conclusion') or ''}".rstrip(),
    ]
    abstract = [a for a in abstract if not a.endswith("**")]
    ai = [k for k, v in ((writer or {}).get("source") or {}).items() if v == "ai"]
    byline = ("*Draft systematic review produced by an agentic AI pipeline (PRISMA 2020 "
              "structure). A reviewer must check it before use."
              + (" The Introduction, Discussion, Conclusions and parts of the abstract were "
                 "drafted by an AI writer from the review's own tables; every number was "
                 "checked by code (Appendix A).*" if ai else "*"))
    intro = sec.get("introduction") or []
    disc = sec.get("discussion") or []
    concl = sec.get("conclusion") or ""
    parts = [f"# {p.get('title') or p.get('question', 'Systematic review')}", "", byline, "",
             "## Abstract", "", "\n\n".join(abstract), ""]
    if intro:
        parts += ["## Introduction", "", *_paras(intro), ""]
    parts += [*_methods(state, models, fx.get("spec")), "",
              *_results(state, base, prisma_png, fx), ""]
    parts += ["## Discussion", "", *(_paras(disc) + [""] if disc else []),
              *_limitations(state), ""]
    agenda = sec.get("research_agenda") or []
    if agenda:
        parts += ["## Research agenda: proposed approaches", "",
                  "*Proposals for future studies, one per evidence gap that holds after the "
                  "check; they follow from the gaps above and are not results of this review.*",
                  "", *_paras(agenda), ""]
    if concl:
        parts += ["## Conclusions", "", concl, ""]
    parts += [*_other_information(), "", *_references(state), "",
              *_appendix(state, models, summary, writer), "", *_rob_per_study(state), ""]
    return "\n".join(parts)
