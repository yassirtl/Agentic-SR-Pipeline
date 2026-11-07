"""Screening: title/abstract, then full text. Dual screening, adjudication, human for the rest.

PRISMA 2020 asks how many reviewers screened each record and how disagreements
were resolved (item 8). This module copies the human method:

1. **Two screeners with opposite instructions** read every record:
   - A (strict, agent ``screener_a``) includes only when the text *shows* every
     inclusion criterion is met;
   - B (lenient, agent ``screener_b``) excludes only when the text *clearly*
     breaks a criterion; missing information is never a reason to exclude.
   When they agree the record is decided. They disagree mostly when the text is
   silent on a criterion: exactly the records worth a second look.
2. **Adjudicator** (agent ``adjudicator``): reads the record and both opinions,
   decides include / exclude, or answers ``unresolved``.
3. **Human**: unresolved records wait in a separate graph node
   (``ta_conflicts`` / ``ft_conflicts``) that pauses the run with LangGraph
   ``interrupt``. In ``--auto`` mode they are included instead
   (``auto-include``): a record wrongly kept is caught at the next stage, a
   record wrongly excluded is lost for good (recall over precision).

Every decision names the criterion it rests on and quotes the text; code checks
that the quote really is in the text (``quote_found``), so invented evidence is
flagged. **No exclusion without evidence** (code, not prompt): two exclusions
that quote nothing found in the text go to the adjudicator, and an adjudicator
exclusion without such a quote becomes ``unresolved``. Agreement between A and
B is measured with Cohen's kappa.

**Full text** (``ft_screening``): only the title/abstract includes. Full text
from ``sources/fulltext.py`` (PMC XML, open PDF, else abstract only), chunked
and indexed once per run (``rag/``, saved to ``runs/<id>/index`` for
extraction). For each paper and each criterion the hybrid index returns that
paper's best passages, the LLM re-ranker orders them, and the top
``ft_passages`` go to the screeners with their chunk ids. Papers without a full
text are screened on their abstract and flagged ``abstract_only``; PRISMA
counts them as "reports not retrieved".

Rounds: decisions are appended with ``round`` and read through ``latest()``,
restricted to the papers of the current retrieval (a paper dropped by a new
search must not keep an old decision). Re-screening a paper is free when its
text did not change (LLM disk cache).

Concurrency: records are screened in ``workers`` threads. The embedder (one
SQLite connection) is only used from the stage's own thread: retrieval for
every (paper, criterion) runs first, then the LLM calls fan out.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from ..llm import LLM, Usage
from ..rag import HybridIndex, chunk_paper, rerank
from ..sources.fulltext import fetch_fulltexts
from ..state import ReviewState, latest, stage_round

ABSTRACT_CHARS = 4000  # an abstract is ~250-400 words; longer "abstracts" are cut


# --- Schemas -----------------------------------------------------------------------

class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude"]
    criterion: str = Field(description="ID of the criterion the decision rests on (e.g. 'E2', "
                           "'I1'); 'all' when every criterion is met.")
    reason: str = Field(description="One or two sentences.")
    quote: str = Field("", description="Exact words copied from the text that support the "
                       "decision; empty only if the decision rests on missing information.")
    chunk_id: str = Field("", description="Full-text screening: id of the passage quoted, "
                          "e.g. 'doi:10.1/x::12'. Empty for title/abstract screening.")


class Adjudication(BaseModel):
    decision: Literal["include", "exclude", "unresolved"]
    criterion: str = Field(description="ID of the deciding criterion, or 'all'.")
    reason: str = Field(description="Why, in one or two sentences, referring to the "
                        "screeners' arguments.")
    quote: str = Field("", description="Exact words copied from the text, if any.")
    chunk_id: str = Field("", description="Full-text screening: id of the passage quoted.")


# --- Prompts -----------------------------------------------------------------------

def criteria(protocol: dict) -> list[tuple[str, str]]:
    """Numbered criteria: I1.. inclusion, D design (if restricted), E1.. exclusion."""
    out = [(f"I{i}", c) for i, c in enumerate(protocol.get("inclusion_criteria", []), 1)]
    designs = protocol.get("study_designs") or []
    if designs:
        # Seen live: "an umbrella review of ... randomized trials", a network meta-analysis
        # and a trial protocol were included as "randomized controlled trial".
        out.append(("D", "Study design is one of: " + "; ".join(designs)
                    + " (a primary study reporting its own results; systematic reviews, "
                      "meta-analyses and study protocols without results are not)"))
    out += [(f"E{i}", c) for i, c in enumerate(protocol.get("exclusion_criteria", []), 1)]
    return out


def protocol_block(protocol: dict) -> str:
    pico = protocol.get("pico", {})
    lines = [f"Review question: {protocol.get('question', '')}",
             f"Population: {pico.get('population', '')}",
             f"Intervention / exposure: {pico.get('intervention', '')}",
             f"Comparator: {pico.get('comparator', '')}",
             f"Outcomes: {'; '.join(pico.get('outcomes', []))}",
             "Criteria (I = inclusion, must all be met; D = design; E = exclusion, any one "
             "excludes):"]
    lines += [f"  {cid}: {text}" for cid, text in criteria(protocol)]
    return "\n".join(lines)


STYLE = {
    "strict": (
        "You are screener A (strict). Include ONLY if the text shows that every inclusion "
        "criterion (and the design, if listed) is met and no exclusion criterion applies. "
        "If the text does not show that a criterion is met, exclude and name that criterion."),
    "lenient": (
        "You are screener B (lenient). Exclude ONLY if the text states a fact about the "
        "study that contradicts a criterion (e.g. 'participants aged 18-40', 'a cross-"
        "sectional survey') or shows that an exclusion criterion applies. Missing or unclear "
        "information is never a reason to exclude: include in that case."),
}

# Seen live (Makizako 2019, a gold study, excluded by both screeners): the sentence
# "the effects of exercise ... on depressive symptoms remain unclear" was read as the
# study not assessing depressive symptoms.
COMMON_RULE = (
    "Judge what the study itself did: its design, participants, intervention and outcomes. "
    "Background sentences about what is unknown ('remains unclear', 'little is known') state "
    "the study's motivation; they are never evidence that it fails a criterion. Eligibility "
    "never depends on the results: a study that found no effect is as eligible as one that "
    "found an effect. Criteria on publication year or language were applied by the "
    "database search: ignore them.")

STAGE_TEXT = {
    "ta": "This is title/abstract screening: you see only the title and abstract.",
    "ft": ("This is full-text screening: you see the abstract and, for each criterion, the "
           "passages of the full text that best match it. Judge the study as reported in "
           "these passages; cite the passage you rely on by its chunk id."),
    # Full-text stage, but no full text could be obtained (PRISMA: report not retrieved).
    # Seen live (Huang 2015, a gold study): excluded at full text for an age the abstract
    # does not state, although nobody could read the full text.
    "ft_abstract": ("This is full-text screening, but the full text could not be obtained: "
                    "you see only the title and abstract. Do not exclude for information the "
                    "abstract does not report; exclude only for a fact that contradicts a "
                    "criterion."),
}


def screening_prompt(protocol: dict, evidence: str, *, style: str, stage: str) -> str:
    return (
        "You screen records for a PRISMA 2020 systematic review.\n"
        f"{STYLE[style]}\n{COMMON_RULE}\n{STAGE_TEXT[stage]}\n"
        "Quote the exact words of the text that support your decision (copy, do not "
        "paraphrase). Name the single criterion your decision rests on ('all' if you "
        "include).\n\n"
        f"{protocol_block(protocol)}\n\n{evidence}"
    )


def adjudicator_prompt(protocol: dict, evidence: str, a: dict, b: dict, *, stage: str) -> str:
    rule = ("Only the title and abstract are available. Information the abstract does not "
            "report is NEVER a reason to exclude or to answer 'unresolved': include, the full "
            "text will decide. Exclude only for a stated fact that contradicts a criterion."
            if stage in ("ta", "ft_abstract") else
            "At full-text stage, decide from the passages. Answer 'unresolved' only if the "
            "passages contradict each other or the decision needs information that is "
            "missing from them.")
    return (
        "You are the third reviewer of a PRISMA 2020 systematic review. Two screeners "
        "disagree on this record; decide.\n"
        f"{rule} If you truly cannot decide, answer 'unresolved' and a human will.\n"
        f"{COMMON_RULE}\n"
        "Quote the exact words of the text you rely on.\n\n"
        f"{protocol_block(protocol)}\n\n"
        f"Screener A (strict): {a['decision']} on {a['criterion']}. {a['reason']} "
        f"Quote: \"{a['quote']}\"\n"
        f"Screener B (lenient): {b['decision']} on {b['criterion']}. {b['reason']} "
        f"Quote: \"{b['quote']}\"\n\n{evidence}"
    )


def ta_evidence(paper: dict) -> str:
    abstract = (paper.get("abstract") or "").strip()[:ABSTRACT_CHARS]
    return (f"Title: {paper.get('title') or '(no title)'}\n"
            f"Abstract: {abstract or '(no abstract available)'}")


def ft_evidence(paper: dict, by_criterion: dict[str, list[dict]]) -> str:
    """Abstract + each criterion's passages; a passage is printed once, then cited by id."""
    lines = [ta_evidence(paper), "", "Full-text passages:"]
    shown: set[str] = set()
    for cid, hits in by_criterion.items():
        lines.append(f"\nFor criterion {cid}:")
        for h in hits:
            if h["chunk_id"] in shown:
                lines.append(f"[{h['chunk_id']}] (shown above)")
                continue
            shown.add(h["chunk_id"])
            lines.append(f"[{h['chunk_id']}] {h['text']}")
    return "\n".join(lines)


# --- Checks and statistics ------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_found(quote: str, text: str) -> bool:
    """True if the quote (punctuation, case and spacing ignored) is in the text.
    An empty quote is accepted (decision on missing information); '...' splits a quote
    into parts that must each be found."""
    parts = [p for p in (_norm(q) for q in re.split(r"\.\.\.|…", quote or "")) if p]
    hay = _norm(text)
    return all(p in hay for p in parts)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement beyond chance between two raters: (p_o - p_e) / (1 - p_e).

    p_o = observed agreement, p_e = agreement expected if both answered at random
    with their own include/exclude rates. 1 = perfect, 0 = chance level. None for no
    data; 1.0 when both raters gave one and the same label to everything (p_e = 1).
    """
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[label] * cb[label] for label in set(a) | set(b)) / (n * n)
    if pe == 1:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def _sum_usage(usages: list[Usage]) -> dict:
    cost = [u.cost_usd for u in usages]
    return {"llm_calls": len(usages),
            "llm_calls_cached": sum(u.cached for u in usages),
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "cost_usd": None if any(c is None for c in cost) else round(sum(cost), 6)}


# --- One record --------------------------------------------------------------------------

def screen_record(llm: LLM, protocol: dict, paper: dict, evidence: str, *, stage: str,
                  rnd: int, mode: str | None = None) -> tuple[dict, list[Usage]]:
    """A + B, then the adjudicator if they disagree. Never raises: an LLM error leaves the
    record ``unresolved`` (human or auto-include), it is never silently dropped.
    ``mode`` picks the prompt variant (default: the stage; 'ft_abstract' for a full-text
    paper without full text)."""
    mode = mode or stage
    pid = paper["paper_id"]
    item = {"paper_id": pid, "round": rnd, "stage": stage, "title": paper.get("title"),
            "a": None, "b": None, "adjudicator": None, "flags": []}
    if stage == "ta" and not (paper.get("abstract") or "").strip():
        item["flags"].append("no_abstract")
    usages: list[Usage] = []
    text = evidence  # quotes must come from what the screener was shown
    try:
        for key, style, agent in (("a", "strict", "screener_a"), ("b", "lenient", "screener_b")):
            d, u = llm.call_with_usage(ScreeningDecision,
                                       screening_prompt(protocol, evidence, style=style,
                                                        stage=mode), agent=agent)
            usages.append(u)
            item[key] = d.model_dump()
            if not quote_found(d.quote, text):
                item["flags"].append(f"quote_not_found_{key}")
        agree = item["a"]["decision"] == item["b"]["decision"]
        if agree and item["a"]["decision"] == "exclude" and not (
                _evidence(item["a"], text) or _evidence(item["b"], text)):
            # No exclusion without evidence: two exclusions that quote nothing found in the
            # text are not an agreement, the adjudicator decides.
            item["flags"].append("exclusion_without_quote")
            agree = False
        if agree:
            final, by = item["a"]["decision"], "agreement"
        else:
            adj, u = llm.call_with_usage(Adjudication,
                                         adjudicator_prompt(protocol, evidence, item["a"],
                                                            item["b"], stage=mode),
                                         agent="adjudicator")
            usages.append(u)
            item["adjudicator"] = adj.model_dump()
            if not quote_found(adj.quote, text):
                item["flags"].append("quote_not_found_adjudicator")
            final, by = ((adj.decision, "adjudicator") if adj.decision != "unresolved"
                         else ("unresolved", "pending"))
            if final == "exclude" and not _evidence(item["adjudicator"], text):
                item["flags"].append("adjudicator_exclusion_without_quote")
                final, by = "unresolved", "pending"
    except Exception as exc:  # LLM / validation / network error after retries
        item["flags"].append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
        final, by = "unresolved", "pending"
    item["final"], item["resolved_by"] = final, by
    item["criterion"] = _deciding_criterion(item)
    return item, usages


def _evidence(decision: dict, text: str) -> bool:
    """A non-empty quote that really is in the text."""
    return bool((decision.get("quote") or "").strip()) and quote_found(decision["quote"], text)


def _deciding_criterion(item: dict) -> str | None:
    """Criterion of the final decision (for 'excluded, with reasons')."""
    if item["final"] not in ("include", "exclude"):
        return None
    for key in ("adjudicator", "a", "b"):
        d = item.get(key)
        if d and d["decision"] == item["final"]:
            return d["criterion"]
    return None


def _parallel(fn: Callable, items: list, workers: int) -> list:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))  # keeps input order: deterministic output


# --- Counts ----------------------------------------------------------------------------

def current_decisions(decisions: list[dict], papers: list[dict], rnd: int | None = None
                      ) -> list[dict]:
    """Latest decision per paper, restricted to the papers given (current retrieval), and
    to round ``rnd`` if given."""
    ids = {p["paper_id"] for p in papers}
    return [d for d in latest(decisions)
            if d["paper_id"] in ids and (rnd is None or d["round"] == rnd)]


def _agreement(decisions: list[dict]) -> dict:
    pairs = [(d["a"]["decision"], d["b"]["decision"]) for d in decisions if d["a"] and d["b"]]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"pairs": len(pairs),
            "agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4) if pairs else None,
            "kappa": cohen_kappa(a, b)}


def ta_counts(decisions: list[dict]) -> dict:
    final = Counter(d["final"] for d in decisions)
    return {
        "records_screened": len(decisions),
        "records_excluded": final["exclude"],
        "included": final["include"],
        "unresolved": final["unresolved"],
        "excluded_by_criterion": dict(Counter(d["criterion"] for d in decisions
                                              if d["final"] == "exclude")),
        "resolved_by": dict(Counter(d["resolved_by"] for d in decisions)),
        "no_abstract": sum("no_abstract" in d["flags"] for d in decisions),
        "quotes_not_found": sum(any(f.startswith("quote_not_found") for f in d["flags"])
                                for d in decisions),
        "llm_errors": sum(any(f.startswith("llm_error") for f in d["flags"])
                          for d in decisions),
        **_agreement(decisions),
    }


def ft_counts(decisions: list[dict]) -> dict:
    """PRISMA full-text boxes. Papers with no full text are 'reports not retrieved'; by
    project decision they are still screened on the abstract and flagged."""
    base = ta_counts(decisions)
    abstract_only = [d for d in decisions if d.get("fulltext_status") == "abstract_only"]
    return {
        "reports_sought": len(decisions),
        "reports_not_retrieved": len(abstract_only),
        "not_retrieved_reasons": dict(Counter(d.get("fulltext_reason") for d in abstract_only)),
        "reports_assessed": len(decisions) - len(abstract_only),
        "screened_on_abstract_only": len(abstract_only),
        "reports_excluded": base["records_excluded"],
        "excluded_by_criterion": base["excluded_by_criterion"],
        "studies_included": base["included"],
        "included_on_abstract_only": sum(d["final"] == "include" for d in abstract_only),
        "unresolved": base["unresolved"],
        "resolved_by": base["resolved_by"],
        "quotes_not_found": base["quotes_not_found"],
        "llm_errors": base["llm_errors"],
        "pairs": base["pairs"], "agreement": base["agreement"], "kappa": base["kappa"],
    }


PRISMA_KEY = {"ta": "title_abstract", "ft": "full_text"}
STAGE_NAME = {"ta": "ta_screening", "ft": "ft_screening"}


def _counts(stage: str, decisions: list[dict], rnd: int) -> dict:
    return {**(ta_counts(decisions) if stage == "ta" else ft_counts(decisions)), "round": rnd}


def _included(decisions: list[dict]) -> list[str]:
    return [d["paper_id"] for d in decisions if d["final"] == "include"]


# --- Title / abstract stage -------------------------------------------------------------

def make_ta_screening(llm: LLM, *, workers: int = 8, limit: int | None = None):
    """Title/abstract screening of ``unique_records``. ``limit`` screens only the first
    N records (development runs; never in a real review)."""

    def ta_screening(state: ReviewState) -> dict:
        protocol = state["protocol"]
        rnd = stage_round("ta_screening", state.get("completed_stages", []))
        papers = state.get("unique_records", [])[:limit] if limit else state.get(
            "unique_records", [])

        def one(paper):
            return screen_record(llm, protocol, paper, ta_evidence(paper), stage="ta", rnd=rnd)

        results = _parallel(one, papers, workers)
        decisions = [item for item, _ in results]
        counts = _counts("ta", decisions, rnd)
        return {
            "ta_decisions": decisions,
            "prisma_counts": {**state.get("prisma_counts", {}), "title_abstract": counts},
            "completed_stages": ["ta_screening"],
            "audit_log": [{"node": "ta_screening", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "title/abstract screening",
                           "limit": limit, **counts,
                           **_sum_usage([u for _, us in results for u in us])}],
        }

    return ta_screening


# --- Full-text stage ---------------------------------------------------------------------

def ta_included(state: ReviewState) -> list[dict]:
    """Papers of the current retrieval whose latest title/abstract decision is not exclude
    (an unresolved one counts as include)."""
    papers = state.get("unique_records", [])
    keep = {d["paper_id"] for d in current_decisions(state.get("ta_decisions", []), papers)
            if d["final"] != "exclude"}
    return [p for p in papers if p["paper_id"] in keep]


def make_ft_screening(llm: LLM, http, embedder, *, index_dir: str | Path, workers: int = 8,
                      passages: int = 5, candidates: int = 10, chunk_words: int = 250,
                      chunk_overlap: int = 40, use_rerank: bool = True,
                      contact_email: str | None = None,
                      env: Mapping[str, str] | None = None, fetch: Callable = fetch_fulltexts):
    """Full-text screening of the title/abstract includes (see module docstring).

    ``candidates``: passages per (paper, criterion) from the hybrid index, re-ranked by
    the LLM (``use_rerank=False`` keeps the fused order: about half the full-text cost);
    ``passages``: how many of those the screeners read. ``fetch`` is injectable (tests)."""
    index_dir = Path(index_dir)

    def ft_screening(state: ReviewState) -> dict:
        protocol = state["protocol"]
        rnd = stage_round("ft_screening", state.get("completed_stages", []))
        papers = ta_included(state)
        fulltexts = fetch(papers, http, contact_email=contact_email, env=env) if papers else {}

        # One index per run, every sought paper in it (extraction reuses it).
        chunks = [c for p in papers for c in chunk_paper(
            p, fulltexts[p["paper_id"]], size=chunk_words, overlap=chunk_overlap)]
        index = HybridIndex.build(chunks, embedder) if chunks else None
        if index is not None:
            index.save(index_dir)

        # Retrieval in this thread (the embedder's SQLite connection is not shared).
        crit = criteria(protocol)
        hits: dict[str, dict[str, list[dict]]] = {}
        for p in papers:
            ft = fulltexts[p["paper_id"]]
            if ft.status != "abstract_only" and index is not None:
                hits[p["paper_id"]] = {
                    cid: index.search(f"{cid}: {text}", k=candidates,
                                      paper_ids=[p["paper_id"]]) for cid, text in crit}

        def one(paper):
            pid = paper["paper_id"]
            ft = fulltexts[pid]
            usages: list[Usage] = []
            by_criterion: dict[str, list[dict]] = {}
            if pid in hits:
                for cid, text in crit:
                    ranked = hits[pid][cid]
                    if use_rerank:
                        ranked, u = rerank(llm, f"{cid}: {text}", ranked, top=candidates)
                        if u is not None:
                            usages.append(u)
                    by_criterion[cid] = ranked[:passages]
                evidence, mode = ft_evidence(paper, by_criterion), "ft"
            else:
                evidence, mode = ta_evidence(paper), "ft_abstract"
            item, us = screen_record(llm, protocol, paper, evidence, stage="ft", rnd=rnd,
                                     mode=mode)
            item["fulltext_status"] = ft.status
            item["fulltext_reason"] = ft.reason
            item["fulltext_url"] = ft.source_url
            if ft.status == "abstract_only":
                item["flags"].append("abstract_only")
            item["passages"] = {cid: [h["chunk_id"] for h in hs]
                                for cid, hs in by_criterion.items()}
            return item, usages + us

        results = _parallel(one, papers, workers)
        decisions = [item for item, _ in results]
        counts = _counts("ft", decisions, rnd)
        status = Counter(ft.status for ft in fulltexts.values())
        return {
            "ft_decisions": decisions,
            "included": _included(decisions),
            "prisma_counts": {**state.get("prisma_counts", {}), "full_text": counts},
            "completed_stages": ["ft_screening"],
            "audit_log": [{"node": "ft_screening", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "full-text screening",
                           "fulltext_status": dict(status),
                           "index": {"dir": str(index_dir), "chunks": len(chunks)},
                           **counts,
                           **_sum_usage([u for _, us in results for u in us])}],
        }

    return ft_screening


# --- Conflicts: human or auto-include ------------------------------------------------------

def unresolved(state: ReviewState, stage: str) -> tuple[list[dict], int]:
    """Unresolved decisions of the stage's latest round (current papers only)."""
    field = f"{stage}_decisions"
    rnd = state.get("completed_stages", []).count(STAGE_NAME[stage])
    papers = state.get("unique_records", [])
    pending = [d for d in current_decisions(state.get(field, []), papers, rnd)
               if d["final"] == "unresolved"]
    return pending, rnd


def apply_resolutions(answer, pending: list[dict]) -> dict[str, str]:
    """Human answer -> {paper_id: include|exclude}. ``True`` = include all (the default).
    Raises ValueError for an unusable answer (the node asks again)."""
    ids = [d["paper_id"] for d in pending]
    if answer is True:
        return {pid: "include" for pid in ids}
    if not isinstance(answer, dict):
        raise ValueError(f"unrecognised answer: {answer!r}")
    unknown = set(answer) - set(ids)
    missing = [pid for pid in ids if pid not in answer]
    bad = {k: v for k, v in answer.items() if v not in ("include", "exclude")}
    if unknown or missing or bad:
        raise ValueError(f"answer every paper with 'include' or 'exclude' "
                         f"(unknown: {sorted(unknown)}, missing: {missing}, invalid: {bad})")
    return dict(answer)


def make_conflict_review(stage: Literal["ta", "ft"], *, human: bool):
    """Node after a screening stage (fixed edge, not a stage the guardrail sees).

    ``human=True`` pauses with ``interrupt`` when records are unresolved; otherwise
    (``--auto``) they are included. Either way a new decision item (same round) is
    appended, so ``latest()`` returns the resolution and the history keeps both.
    """
    field = f"{stage}_decisions"
    node_name = f"{stage}_conflicts"

    def conflicts(state: ReviewState) -> dict:
        pending, rnd = unresolved(state, stage)
        if not pending:
            return {}
        if human:
            error, asked = None, 0
            while True:
                asked += 1
                answer = interrupt({
                    "kind": "screening_conflicts", "stage": STAGE_NAME[stage], "error": error,
                    "papers": [{"paper_id": d["paper_id"], "title": d["title"],
                                "screener_a": d["a"], "screener_b": d["b"],
                                "adjudicator": d["adjudicator"], "flags": d["flags"]}
                               for d in pending],
                    "help": "resume with {paper_id: 'include' | 'exclude', ...} for every "
                            "paper, or True to include them all"})
                try:
                    resolved = apply_resolutions(answer, pending)
                    break
                except ValueError as exc:
                    error = str(exc)
            by = "human"
        else:
            resolved, by, asked = {d["paper_id"]: "include" for d in pending}, "auto-include", 0

        new = []
        for d in pending:
            final = resolved[d["paper_id"]]
            item = {**d, "final": final, "resolved_by": by}
            item["criterion"] = _deciding_criterion(item) or ("human" if by == "human"
                                                               else "auto")
            new.append(item)
        papers = state.get("unique_records", [])
        decisions = current_decisions(state.get(field, []) + new, papers, rnd)
        update = {
            field: new,
            "prisma_counts": {**state.get("prisma_counts", {}),
                              PRISMA_KEY[stage]: _counts(stage, decisions, rnd)},
            "audit_log": [{"node": node_name, "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "screening conflicts resolved",
                           "resolved_by": by, "attempts": asked,
                           "decisions": {pid: f for pid, f in resolved.items()}}],
        }
        if stage == "ft":
            update["included"] = _included(decisions)
        return update

    conflicts.__name__ = node_name
    return conflicts


def describe_screening(state: ReviewState) -> list[str]:
    """Lines for the Coordinator summary."""
    out = []
    pc = state.get("prisma_counts", {})
    if (ta := pc.get("title_abstract")):
        out.append(f"Title/abstract screening (round {ta['round']}): {ta['records_screened']} "
                   f"screened, {ta['included']} included, {ta['records_excluded']} excluded, "
                   f"kappa {ta['kappa']}")
    if (ft := pc.get("full_text")):
        out.append(f"Full-text screening (round {ft['round']}): {ft['reports_sought']} sought, "
                   f"{ft['reports_not_retrieved']} without full text, "
                   f"{ft['studies_included']} included")
    return out
