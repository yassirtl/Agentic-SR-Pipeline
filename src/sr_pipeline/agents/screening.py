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

**One verdict per criterion** (Part 6b). A screener does not pick a label; it
answers every criterion with ``met: yes | no | unclear`` and a quote. Code then
decides (so the label can never contradict the verdicts):

- a criterion *fails* when an inclusion or design criterion (D, I1..) is ``no``,
  or an exclusion criterion (E1..) is ``yes`` (the exclusion applies);
- a failure stands only with a quote found in the text (``quote_found``);
  otherwise it is downgraded to ``unclear`` and flagged (``unverified_*``);
- A (strict) excludes on any failure, or on any D / I criterion that is
  ``unclear`` (the text does not show it is met); B (lenient) excludes only on
  a failure. Without full text (``ft_abstract``) A follows B's rule;
- the **deciding criterion** is the first failure in the fixed order D, I1..,
  E1.. (for A on missing information: the first unclear D / I). So PRISMA's
  "excluded, with reasons" counts come from code, not from a free label.

**No exclusion without evidence** (code, not prompt): two exclusions with no
verified quote go to the adjudicator, and an adjudicator failure without a
verified quote makes the record ``unresolved``.

**Primary studies only** (D, always present and checked first): reviews,
meta-analyses (incl. network meta-analyses), protocols without results and
editorials fail D. Design guard: if a screener failed D with a verified quote,
the adjudicator can include the record only by quoting evidence that D is met;
otherwise the record is excluded on D (``resolved_by: design_guard``), never
left ``unresolved`` (which ``--auto`` would include). Seen live: the adjudicator
let a network meta-analysis and a trial protocol through.

**Stage rules** (Part 6c, from the 6b run where A excluded all 50 abstracts):
- criteria tagged ``full_text`` (``protocol['full_text_criteria']``, e.g. "groups
  comparable at baseline", "mean and SD reported") are shown at title/abstract
  but never decide there: their verdict is forced to ``unclear`` (``deferred``)
  and does not trigger A's strict rule. They decide at full text;
- multi-arm trials are judged on one eligible pair of arms, named in
  ``comparison`` before the verdicts; other arms never exclude the study
  (Cochrane Handbook: use the relevant arms);
- two exclusions on **different** deciding criteria are not an agreement
  (flag ``different_reasons``): the adjudicator decides. A general check against
  one screener misreading a criterion (6b: A excluded Huang on E1 from "without
  impaired cognition", B on E2);
- a criterion may carry a positive test sentence
  (``protocol['criterion_tests']``), judged instead of the displayed wording.

**Last checks** (Part 6d, from the 6c run; screening rules frozen after this part):
- **D title rule**: a title that states the record is a review, meta-analysis,
  protocol, editorial or commentary (``title_rule``: only where the title names the
  record type, never "screening in systematic reviews" or "two exercise protocols")
  is excluded on D with the title as the quote, before any LLM call
  (``resolved_by: d_title_rule``). Seen live: a "study protocol of a randomized
  controlled trial" got D 'yes' from all three LLM calls;
- **eligibility-list quotes**: an E 'yes' whose verified quote sits in the study's own
  eligibility sentence ("exclusion criteria", "were excluded if", "eligible if") is
  downgraded to ``unclear`` (flag ``eligibility_quote_*``): a study that excluded
  people with X meets the review's "no X". Seen live: Makizako (gold) lost at full text.
  Also catches inclusion lists ("inclusion criteria: ... with dementia"), where the
  exclusion may in fact apply: a record wrongly kept is caught later, one wrongly
  excluded is lost (recall over precision).

**Engineering-topic checks** (Part 10b, bug fixes after the freeze, from the T2 run: 75 of
100 included, "yes" verdicts whose quotes did not support them). Code only, no prompt
change:
- **evidence terms**: a criterion may carry words a supporting quote must contain
  (``protocol['criterion_evidence']``, from topics.yaml or the Planner). A 'yes' whose quote
  contains none of them (whole words, plural allowed) is ``unclear`` (flag
  ``no_evidence_term_*``). Seen: I3 "evaluated on a constrained device" answered yes quoting
  "we applied the MIT-BIH dataset"; I4 "efficiency metric" yes quoting "accuracy,
  precision, recall". Only the words are checked, not whether the quote is in the text;
- **absence at full text**: with the full text available (mode ``ft``, never
  ``ft_abstract``), an inclusion criterion still ``unclear`` after the checks is not met
  (``met: no``, ``absence: True``, flag ``ft_absence_*``): the full paper does not show it.
  Peer-review criteria are exempt (a paper's text rarely states its venue; the arXiv rule
  covers preprints). "Full paper" means the passages retrieved for that criterion (the
  evidence terms are added to the retrieval query), a limit to report;
- **arXiv rule**: a DOI starting with 10.48550 (arXiv) fails the peer-review inclusion
  criterion, when the protocol has one, before any LLM call (``resolved_by: arxiv_rule``;
  an automation tool in PRISMA, like the title rule).

Agreement between A and B is measured with Cohen's kappa (records decided by the title
rule have no A / B pair and are not counted).

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

from ..llm import LLM, Usage, is_fatal
from ..rag import HybridIndex, chunk_paper, rerank
from ..sources.fulltext import fetch_fulltexts
from ..state import ReviewState, latest, stage_round

ABSTRACT_CHARS = 4000  # an abstract is ~250-400 words; longer "abstracts" are cut


# --- Schemas -----------------------------------------------------------------------

class CriterionVerdict(BaseModel):
    # Field order matters: the model writes the evidence before the verdict. Seen live
    # (verdict first): "D: no" quoting "A Randomized Controlled Trial".
    criterion: str = Field(description="Criterion ID as listed: 'D', 'I1', 'E2', ...")
    quote: str = Field("", description="Exact words copied from the text that show whether "
                       "the criterion is met; empty if the text says nothing about it.")
    met: Literal["yes", "no", "unclear"] = Field(
        description="yes = the quote shows the criterion holds for this study; no = the "
                    "quote states a fact showing it does not; unclear = the text does not "
                    "say. For an exclusion criterion (E), yes = the exclusion applies.")
    chunk_id: str = Field("", description="Full-text screening: id of the passage quoted, "
                          "e.g. 'doi:10.1/x::12'. Empty for title/abstract screening.")


COMPARISON = Field("", description="The pair of arms the verdicts judge, chosen first, "
                   "e.g. 'aerobic exercise arm vs usual-care control'. For a multi-arm "
                   "trial, the one pair that fits the criteria best. Empty if the study "
                   "has no arms (e.g. a review).")


class ScreeningDecision(BaseModel):
    # Field order matters: the arm pair is chosen before the criteria are judged on it.
    comparison: str = COMPARISON
    verdicts: list[CriterionVerdict] = Field(
        description="One verdict per criterion, every criterion, in the order listed.")
    reason: str = Field(description="One or two sentences on the record.")


class Adjudication(BaseModel):
    comparison: str = COMPARISON
    verdicts: list[CriterionVerdict] = Field(
        description="Your own verdict per criterion, every criterion, in the order listed.")
    unresolved: bool = Field(False, description="True only if you truly cannot decide; a "
                             "human will.")
    reason: str = Field(description="Why, in one or two sentences, referring to the "
                        "screeners' arguments.")


# --- Prompts -----------------------------------------------------------------------

NOT_PRIMARY = ("(reviews, meta-analyses incl. network meta-analyses, study protocols "
               "without results, editorials and letters do not meet D)")


def design_criterion(protocol: dict) -> str:
    designs = protocol.get("study_designs") or []
    # Seen live: "an umbrella review of ... randomized trials", a network meta-analysis
    # and a trial protocol were included as "randomized controlled trial".
    if designs:
        return ("A primary study reporting its own results, with a design that is one of: "
                + "; ".join(designs) + " " + NOT_PRIMARY)
    return "A primary study reporting its own results, any design " + NOT_PRIMARY


def criteria(protocol: dict) -> list[tuple[str, str]]:
    """Criteria in deciding order: D (design / primary study, always), I1.. inclusion,
    E1.. exclusion. The first failed criterion in this order is the reason for exclusion."""
    out = [("D", design_criterion(protocol))]
    out += [(f"I{i}", c) for i, c in enumerate(protocol.get("inclusion_criteria", []), 1)]
    out += [(f"E{i}", c) for i, c in enumerate(protocol.get("exclusion_criteria", []), 1)]
    return out


# Part 9b: short exclusion reasons for the PRISMA diagram and tables (display only).
DESIGN_SHORT = {"randomized controlled trial": "RCT", "randomised controlled trial": "RCT"}
OTHER_REASONS = {"human": "excluded by a human", "auto": "excluded automatically"}


def criterion_labels(protocol: dict) -> dict[str, str]:
    """Criterion id -> short label (at most 5 words). D from the designs; I/E from the
    protocol's ``criterion_labels`` (topics.yaml or the Planner), else the criterion's first
    5 words. Also the non-criterion reasons ('human')."""
    designs = protocol.get("study_designs") or []
    if len(designs) == 1:
        d = designs[0].strip()
        labels = {"D": f"not a primary {DESIGN_SHORT.get(d.lower(), d)}"}
    else:
        labels = {"D": "not an eligible primary study" if designs else "not a primary study"}
    given = {x["criterion"]: x["label"] for x in protocol.get("criterion_labels") or []}
    for cid, text in criteria(protocol)[1:]:
        words = text.split()
        labels[cid] = given.get(text) or " ".join(words[:5]) + ("..." if len(words) > 5
                                                                 else "")
    return {**labels, **OTHER_REASONS}


def deferred_at_ta(protocol: dict) -> set[str]:
    """IDs of the criteria an abstract can never show (checked only at full text)."""
    full_text = set(protocol.get("full_text_criteria") or [])
    return {cid for cid, text in criteria(protocol) if text in full_text}


def criterion_query(protocol: dict, cid: str, text: str) -> str:
    """Text used to retrieve (and re-rank) full-text passages for a criterion. Part 10b:
    with the criterion's evidence terms, so the passages that could show it are found
    before its absence is concluded."""
    test = (protocol.get("criterion_tests") or {}).get(text)
    terms = _evidence_by_text(protocol).get(text)
    return (f"{cid}: {text}" + (f" ({test})" if test else "")
            + (f" [{', '.join(terms)}]" if terms else ""))


def _evidence_by_text(protocol: dict) -> dict[str, list[str]]:
    return {e["criterion"]: list(e["terms"]) for e in protocol.get("criterion_evidence") or []
            if e.get("terms")}


def evidence_terms(protocol: dict) -> dict[str, list[str]]:
    """Part 10b: criterion id -> words a quote must contain for a 'yes' to count."""
    by_text = _evidence_by_text(protocol)
    return {cid: by_text[text] for cid, text in criteria(protocol) if text in by_text}


PEER_REVIEW = re.compile(r"peer.?review", re.I)


def peer_review_criterion(protocol: dict) -> str | None:
    """Id of the first inclusion criterion about peer review, if any."""
    return next((cid for cid, text in criteria(protocol)
                 if cid.startswith("I") and PEER_REVIEW.search(text)), None)


ARXIV_DOI = "10.48550/"


def arxiv_rule(protocol: dict, paper: dict) -> str | None:
    """Part 10b: the peer-review criterion id when the record is an arXiv preprint (DOI
    10.48550/...) and the protocol requires peer review; else None."""
    doi = (paper.get("doi") or "").strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
    if not doi.startswith(ARXIV_DOI):
        return None
    return peer_review_criterion(protocol)


def protocol_block(protocol: dict) -> str:
    pico = protocol.get("pico", {})
    lines = [f"Review question: {protocol.get('question', '')}",
             f"Population: {pico.get('population', '')}",
             f"Intervention / exposure: {pico.get('intervention', '')}",
             f"Comparator: {pico.get('comparator', '')}",
             f"Outcomes: {'; '.join(pico.get('outcomes', []))}",
             "Criteria (D = design and I = inclusion: all must be met; E = exclusion: any "
             "one that applies excludes):"]
    later, tests = deferred_at_ta(protocol), protocol.get("criterion_tests") or {}
    for cid, text in criteria(protocol):
        lines.append(f"  {cid}{' [full text only]' if cid in later else ''}: {text}")
        if text in tests:
            lines.append(f"      Test for {cid}: {tests[text]}")
    return "\n".join(lines)


STYLE = {
    "strict": (
        "You are screener A (strict). Answer 'yes' only when the text explicitly shows the "
        "criterion holds; if it is only implied or not stated, answer 'unclear'."),
    "lenient": (
        "You are screener B (lenient). Answer 'yes' when the text shows the criterion holds "
        "or clearly implies it; 'unclear' when the text is silent or vague. Missing "
        "information is never a 'no'."),
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
    "database search: never let them decide (inclusion: 'yes', exclusion: 'no').\n"
    # Seen live (6b): Makizako (exercise / horticulture / control) and Huang (exercise /
    # CBT / control), both in the gold review, excluded for their extra arm.
    "Multi-arm trials: the study is eligible if at least one pair of arms fits the criteria "
    "(e.g. an exercise arm vs a control arm without exercise). Choose that pair first and "
    "name it in 'comparison', then judge every criterion on that pair; the trial's other "
    "arms (another intervention, a combined programme) never exclude the study.\n"
    # Seen live (6c): Makizako excluded at full text on E1 by quoting its own exclusion
    # list ("The exclusion criteria were as follows: ... dementia diagnosis").
    "A study's own eligibility list says whom it kept out: a study that excluded people "
    "with X meets the review's 'no X' (for an exclusion criterion on X answer 'no'). Never "
    "quote a study's exclusion criteria as evidence that an exclusion applies to it.")

VERDICT_RULE = (
    "Give one verdict per criterion, for every criterion, in the order listed. First copy "
    "the exact words of the text about that criterion (do not paraphrase), then answer met: "
    "'yes' if they show the study meets it, 'no' only if they state a fact that contradicts "
    "it (e.g. 'participants aged 18-40', 'a meta-analysis of 20 trials'), 'unclear' if the "
    "text says nothing about it. Words that show a criterion IS met (e.g. 'a randomized "
    "controlled trial' for D) mean 'yes', never 'no'. For E criteria, 'yes' means the "
    "exclusion applies to the study. A 'no' (or an E 'yes') without a quote found in the "
    "text is not accepted. When a criterion has a 'Test' sentence, judge that sentence "
    "(for E: 'yes' = the sentence is true of the study).\n"
    "D first: is this record itself a primary study (its own participants and data), or a "
    "review, meta-analysis or protocol? A record that reviews, pools or re-analyses other "
    "studies is 'no' for D even when those studies were randomized trials (seen: 'a "
    "meta-analysis of randomized controlled trials' answered 'yes'); quote the words that "
    "say what the record is.")

STAGE_TEXT = {
    "ta": ("This is title/abstract screening: you see only the title and abstract. "
           "Criteria marked [full text only] cannot be judged from an abstract: answer "
           "'unclear' for them."),
    "ft": ("This is full-text screening: you see the abstract and, for each criterion, the "
           "passages of the full text that best match it. Judge the study as reported in "
           "these passages; give the chunk id of the passage each quote comes from."),
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
        f"{STYLE[style]}\n{COMMON_RULE}\n{STAGE_TEXT[stage]}\n{VERDICT_RULE}\n\n"
        f"{protocol_block(protocol)}\n\n{evidence}"
    )


def _opinion(name: str, d: dict) -> str:
    lines = [f"{name}: {d['decision']}"
             + (f" on {d['criterion']}" if d["decision"] == "exclude" else "")
             + f". {d['reason']}"
             + (f" Comparison judged: {d['comparison']}" if d.get("comparison") else "")]
    for v in d["verdicts"]:
        quote = f' "{v["quote"]}"' if v.get("quote") else ""
        lines.append(f"  {v['criterion']}: {v['met']}{quote}")
    return "\n".join(lines)


def adjudicator_prompt(protocol: dict, evidence: str, a: dict, b: dict, *, stage: str) -> str:
    rule = ("Only the title and abstract are available. Information the abstract does not "
            "report is NEVER a 'no' and never a reason to answer unresolved: the full text "
            "will decide. A 'no' needs a stated fact that contradicts the criterion."
            if stage in ("ta", "ft_abstract") else
            "At full-text stage, decide from the passages. Set unresolved only if the "
            "passages contradict each other or the decision needs information that is "
            "missing from them.")
    return (
        "You are the third reviewer of a PRISMA 2020 systematic review. Two screeners "
        "disagree on this record; give your own verdicts.\n"
        f"{rule} If you truly cannot decide, set unresolved and a human will.\n"
        "If a screener says the record is not a primary study (D: no), you may answer D "
        "'yes' only by quoting words that show the study collected and reports its own "
        "data.\n"
        f"{COMMON_RULE}\n{VERDICT_RULE}\n\n"
        f"{protocol_block(protocol)}\n\n"
        f"{_opinion('Screener A (strict)', a)}\n{_opinion('Screener B (lenient)', b)}\n\n"
        f"{evidence}"
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

def fails(cid: str, met: str) -> bool:
    """D / I criteria fail on 'no'; an exclusion criterion E 'fails' when it applies ('yes')."""
    return met == ("yes" if cid.startswith("E") else "no")


# A D 'no' whose quote names a listed design and nothing that makes the record a
# non-primary report contradicts itself. Seen live (gpt-4o-mini): "D: no" quoting
# "A pilot cluster-randomized controlled trial", which turned a missing-information
# exclusion by A into an agreed exclusion.
NON_PRIMARY_WORDS = re.compile(
    r"review|meta.?analy|protocol|secondary|post.?hoc|re.?analy|pooled|umbrella|editorial|"
    r"commentary|letter|will be", re.I)


# --- Part 6d: D title rule ----------------------------------------------------------
# Seen live (6c): "...: study protocol of a randomized controlled trial" got D 'yes' from
# A, B and the adjudicator. A title that *states* the record is a review, protocol or
# editorial decides D before any LLM call. The words count only where the title names the
# record type (start, after ':' / '.' / '?' / a dash, closing clause): T2 is about LLMs
# for systematic reviews, so "screening performance in systematic reviews" (a gold study)
# must not match; and exercise trials use "protocol" for the programme ("two physical
# exercise protocols ...: a randomized controlled trial").
_TYPE_START = r"(?:^|[:.?!–—]\s*|\s?-\s?)"  # also "adults- a review", "frailty-a review"
# Up to three words that qualify a review type ("a Bayesian network meta-analysis", "Guest
# editorial"). A fixed list, not any word: "Transforming systematic literature reviews:"
# (a T2 gold study) must not match.
_MODIFIERS = (r"(?:(?:an?|the)\s+)?(?:(?:updated|rapid|brief|short|guest|invited|living|"
              r"cumulative|comprehensive|bayesian|network|pairwise|dose[\s-]response|"
              r"individual[\s-](?:patient|participant)[\s-]data|ipd|critical|qualitative|"
              r"quantitative|mixed[\s-]methods?|systematic|narrative|literature|"
              r"meta[\s-]analytic(?:al)?)\s+){0,3}")
_REVIEW_TYPE = (r"(?:systematic\s+(?:literature\s+)?reviews?|meta[\s-]?analys[ie]s|"
                r"meta[\s-]?analytic(?:al)?\s+reviews?|scoping\s+reviews?|umbrella\s+reviews?|editorials?|commentar(?:y|ies))")
_TYPE_END = (r"(?=\s*$|\s*[:.,;?!&(\[–—]|\s-\s|\s+(?:and|of|on|for|in|with|to|"
             r"from|including|protocol|update)\b)")
TITLE_TYPE = re.compile(
    _TYPE_START + _MODIFIERS + r"(?P<type>" + _REVIEW_TYPE + r")" + _TYPE_END
    # "study protocol", "trial protocol", "review protocol" anywhere
    + r"|(?P<proto1>\b(?:study|trial|review|research|rct)\s+protocol)\b"
    # "Protocol for a randomized trial", "protocol of a pilot RCT"
    + r"|(?P<proto2>\bprotocol\s+(?:for|of)\s+(?:an?\s+|the\s+)?(?:[\w,-]+\s+){0,6}?"
      r"(?:trials?|stud(?:y|ies)|rcts?|reviews?|evaluation))\b"
    # ": protocol ...", title starting with "Protocol", or ending with "... protocol"
    + r"|(?P<proto3>" + _TYPE_START + r"(?:an?\s+|the\s+)?protocol\b)"
    + r"|(?P<proto4>\b(?:trial|rct|study|review)\s+protocol\s*\.?\s*$)",
    re.I)


def title_rule(title: str | None) -> str | None:
    """The words of a title that state the record is not a primary study (review,
    meta-analysis, protocol, editorial, commentary), else None."""
    m = TITLE_TYPE.search((title or "").strip())
    if not m:
        return None
    return next(g for g in m.groups() if g).strip(" :.?!-–—")


# --- Part 6d: eligibility-list quotes -------------------------------------------------
# Seen live (6c, Makizako lost at full text): E1 "yes" (disease other than depression)
# quoting "The exclusion criteria were as follows: ... (2) dementia diagnosis". A study's
# own eligibility list says whom it kept out (or let in): it is not evidence that the
# review's exclusion applies to its participants. Phrases are matched on normalised text.
ELIGIBILITY_WORDS = re.compile(
    r"\b(?:exclusion|inclusion|eligibility) criteri|\b(?:were|was|are|is|be|been) excluded\b|"
    r"\bexcluded if\b|\bexclude (?:those|participants|patients|people|individuals|subjects)\b|"
    r"\bineligible\b|\bnot eligible\b|\beligible if\b|\bwere eligible\b|\bare eligible\b")
# A sentence ends at . ! ? after a non-digit (so "1. Dementia" stays in its list),
# followed by a capital; or at a blank line.
_SENTENCE_END = re.compile(r"(?<=[^\d\s][.!?])\s+(?=[A-Z])|\n\s*\n")


def eligibility_sentence(quote: str, text: str) -> str | None:
    """The quote, or a sentence of ``text`` containing part of it, when it states an
    eligibility rule ("exclusion criteria", "were excluded if", "eligible if"...)."""
    if ELIGIBILITY_WORDS.search(_norm(quote)):
        return quote
    parts = [p for p in (_norm(q) for q in re.split(r"\.\.\.|…", quote or "")) if p]
    for sentence in _SENTENCE_END.split(text):
        norm = _norm(sentence)
        if any(p in norm for p in parts) and ELIGIBILITY_WORDS.search(norm):
            return sentence.strip()
    return None


def term_pattern(terms: list[str]) -> re.Pattern | None:
    """Whole words on normalised text, plural allowed: 'FPGA' matches 'FPGAs', '128 kB'
    and '128kB'; 'edge' does not match 'knowledge', 'SoC' not 'social', 'power' not
    'powerful'."""
    alts = sorted({n for n in (_norm(t) for t in terms) if n}, key=len, reverse=True)
    if not alts:
        return None
    return re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(a) for a in alts)
                      + r")(?:e?s)?(?![a-z])")


def has_evidence_term(quote: str, terms: list[str]) -> bool:
    pattern = term_pattern(terms)
    return pattern is None or bool(pattern.search(_norm(quote or "")))


def design_words(protocol: dict) -> re.Pattern | None:
    """Words that name one of the listed designs (None: no design restriction)."""
    words = []
    for d in protocol.get("study_designs") or []:
        first = _norm(d).split(" ")[0] if _norm(d) else ""
        words.append(r"randomi[sz]|\brcts?\b" if first.startswith("randomi")
                     else re.escape(first))
    words = [w for w in words if w]
    return re.compile("|".join(words), re.I) if words else None


def check_verdicts(raw: list[dict], crit_ids: list[str], text: str, who: str,
                   design_re: re.Pattern | None = None, deferred: set[str] = frozenset(),
                   *, evidence: Mapping[str, list[str]] | None = None,
                   absence: set[str] = frozenset()
                   ) -> tuple[list[dict], list[str]]:
    """Verdicts in criterion order, checked by code. A criterion the LLM skipped is
    'unclear'; a failure whose quote is empty or not in the text is downgraded to
    'unclear' (flag ``unverified_<who>_<criterion>``), and so is a D 'no' whose quote
    names a listed design without any non-primary word (flag
    ``contradicting_quote_<who>_D``), and so is an E 'yes' whose quote sits in an
    eligibility sentence (flag ``eligibility_quote_<who>_<criterion>``). Each verdict gets ``verified``: a non-empty quote
    found in the text. Criteria in ``deferred`` (full-text-only, at title/abstract) are
    forced to 'unclear' with ``deferred: True``; the model's answer stays in
    ``met_claimed``.

    Part 10b: a 'yes' on a criterion of ``evidence`` whose quote has none of its terms is
    'unclear' (flag ``no_evidence_term_<who>_<criterion>``); then each criterion of
    ``absence`` (inclusion criteria at full text, full text available) still 'unclear' is
    'no' with ``absence: True`` (flag ``ft_absence_<who>_<criterion>``)."""
    evidence = evidence or {}
    by_id: dict[str, dict] = {}
    for v in raw:
        cid = (v.get("criterion") or "").strip().upper()
        if cid in crit_ids and cid not in by_id:
            by_id[cid] = v
    out, flags = [], []
    for cid in crit_ids:
        v = by_id.get(cid)
        if v is None:
            flags.append(f"verdict_missing_{who}_{cid}")
            out.append({"criterion": cid, "met": "unclear", "quote": "", "chunk_id": "",
                        "verified": False})
            continue
        quote = v.get("quote") or ""
        verified = bool(quote.strip()) and quote_found(quote, text)
        if quote.strip() and not verified and f"quote_not_found_{who}" not in flags:
            flags.append(f"quote_not_found_{who}")
        item = {"criterion": cid, "met": v["met"], "quote": quote,
                "chunk_id": v.get("chunk_id") or "", "verified": verified}
        if cid in deferred:
            if v["met"] != "unclear":
                item["met_claimed"] = v["met"]
            item["met"], item["deferred"] = "unclear", True
        elif fails(cid, v["met"]) and not verified:
            flags.append(f"unverified_{who}_{cid}")
            item["met_claimed"], item["met"] = v["met"], "unclear"
        elif (cid.startswith("E") and v["met"] == "yes"
              and (sentence := eligibility_sentence(quote, text))):
            flags.append(f"eligibility_quote_{who}_{cid}")
            item["met_claimed"], item["met"] = v["met"], "unclear"
            item["eligibility_sentence"] = sentence[:500]
        elif (cid == "D" and v["met"] == "no" and design_re is not None
              and design_re.search(quote) and not NON_PRIMARY_WORDS.search(quote)):
            flags.append(f"contradicting_quote_{who}_D")
            item["met_claimed"], item["met"] = v["met"], "unclear"
        elif (v["met"] == "yes" and cid in evidence
              and not has_evidence_term(quote, evidence[cid])):
            flags.append(f"no_evidence_term_{who}_{cid}")
            item["met_claimed"], item["met"] = v["met"], "unclear"
        out.append(item)
    for item in out:
        if item["criterion"] in absence and item["met"] == "unclear":
            item.setdefault("met_claimed", "unclear")
            item["met"], item["absence"] = "no", True
            flags.append(f"ft_absence_{who}_{item['criterion']}")
    return out, flags


def decide(verdicts: list[dict], rule: str) -> dict:
    """Decision from checked verdicts (in deciding order D, I1.., E1..):
    exclude on the first failed criterion; with the strict rule, also on the first D / I
    criterion left unclear (missing information, no quote)."""
    for v in verdicts:
        if fails(v["criterion"], v["met"]):
            return {"decision": "exclude", "criterion": v["criterion"], "quote": v["quote"],
                    "chunk_id": v["chunk_id"], "missing_info": False,
                    "absence": bool(v.get("absence"))}
    if rule == "strict":
        for v in verdicts:
            if (not v["criterion"].startswith("E") and v["met"] == "unclear"
                    and not v.get("deferred")):
                return {"decision": "exclude", "criterion": v["criterion"], "quote": "",
                        "chunk_id": "", "missing_info": True}
    return {"decision": "include", "criterion": "all", "quote": "", "chunk_id": "",
            "missing_info": False}


def _opinion_item(answer, crit_ids: list[str], text: str, who: str, rule: str,
                  design_re: re.Pattern | None = None, deferred: set[str] = frozenset(),
                  evidence: Mapping[str, list[str]] | None = None,
                  absence: set[str] = frozenset()) -> tuple[dict, list[str]]:
    verdicts, flags = check_verdicts([v.model_dump() for v in answer.verdicts], crit_ids,
                                     text, who, design_re, deferred, evidence=evidence,
                                     absence=absence)
    return {"comparison": answer.comparison, "verdicts": verdicts, "reason": answer.reason,
            **decide(verdicts, rule)}, flags


def _verdict(d: dict | None, cid: str) -> dict | None:
    return next((v for v in (d or {}).get("verdicts", []) if v["criterion"] == cid), None)


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
    if (words := title_rule(paper.get("title"))):
        # The title states a non-primary record: D fails with the title as the quote, no
        # LLM call (A / B stay None, like an LLM error; kappa counts LLM-screened pairs).
        item["flags"].append("d_title_rule")
        item.update(final="exclude", resolved_by="d_title_rule", criterion="D",
                    quote=paper.get("title"), title_words=words)
        return item, []
    if (peer := arxiv_rule(protocol, paper)):
        # Part 10b: an arXiv preprint fails the peer-review criterion (no LLM call).
        item["flags"].append("arxiv_rule")
        item.update(final="exclude", resolved_by="arxiv_rule", criterion=peer,
                    quote=paper.get("doi"))
        return item, []
    usages: list[Usage] = []
    text = evidence  # quotes must come from what the screener was shown
    crit_ids = [cid for cid, _ in criteria(protocol)]
    design_re = design_words(protocol)
    deferred = deferred_at_ta(protocol) if mode == "ta" else set()
    ev_terms = evidence_terms(protocol)
    # Part 10b: with the full text, an inclusion criterion it does not show is not met.
    absence = ({cid for cid, t in criteria(protocol)
                if cid.startswith("I") and not PEER_REVIEW.search(t)}
               if mode == "ft" else set())
    # Without a full text nobody can see what the abstract omits: A does not exclude on it.
    rule_a = "lenient" if mode == "ft_abstract" else "strict"
    try:
        for key, style, agent, rule in (("a", "strict", "screener_a", rule_a),
                                        ("b", "lenient", "screener_b", "lenient")):
            d, u = llm.call_with_usage(ScreeningDecision,
                                       screening_prompt(protocol, evidence, style=style,
                                                        stage=mode), agent=agent)
            usages.append(u)
            item[key], flags = _opinion_item(d, crit_ids, text, key, rule, design_re,
                                             deferred, ev_terms, absence)
            item["flags"] += flags
        agree = item["a"]["decision"] == item["b"]["decision"]
        if agree and item["a"]["decision"] == "exclude" and not (
                _evidence(item["a"], text) or _evidence(item["b"], text)):
            # No exclusion without evidence: two exclusions that quote nothing found in the
            # text are not an agreement, the adjudicator decides.
            item["flags"].append("exclusion_without_quote")
            agree = False
        elif agree and item["a"]["decision"] == "exclude" and \
                item["a"]["criterion"] != item["b"]["criterion"]:
            # Same decision, different reasons: at least one screener misread a criterion
            # (6b: Huang, A on E1 from "without impaired cognition", B on E2).
            item["flags"].append("different_reasons")
            agree = False
        if agree:
            final, by = item["a"]["decision"], "agreement"
            criterion = item["a"]["criterion"]
        else:
            adj, u = llm.call_with_usage(Adjudication,
                                         adjudicator_prompt(protocol, evidence, item["a"],
                                                            item["b"], stage=mode),
                                         agent="adjudicator")
            usages.append(u)
            item["adjudicator"], flags = _opinion_item(adj, crit_ids, text, "adjudicator",
                                                       "lenient", design_re, deferred,
                                                       ev_terms, absence)
            item["adjudicator"]["unresolved"] = adj.unresolved
            item["flags"] += flags
            d = item["adjudicator"]
            final, by, criterion = d["decision"], "adjudicator", d["criterion"]
            if final == "include" and (adj.unresolved or any(
                    f.startswith("unverified_adjudicator_") for f in flags)):
                # It wanted to exclude without evidence, or could not decide: a human will.
                if not adj.unresolved:
                    item["flags"].append("adjudicator_exclusion_without_quote")
                final, by, criterion = "unresolved", "pending", None
            if final != "exclude" and (guard := _design_guard(item)):
                item["flags"].append("design_guard")
                final, by, criterion = "exclude", "design_guard", "D"
                item["guard_quote"] = guard
    except Exception as exc:  # LLM / validation / network error after retries
        if is_fatal(exc):  # no credits / bad key: stop the run, do not mark 100 unresolved
            raise
        item["flags"].append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
        final, by, criterion = "unresolved", "pending", None
    item["final"], item["resolved_by"], item["criterion"] = final, by, criterion
    return item, usages


def _design_guard(item: dict) -> str | None:
    """Quote of a screener's verified D failure, unless the adjudicator quoted evidence
    that D is met. Seen live: the adjudicator included a network meta-analysis and a
    trial protocol that a screener had excluded on D with a quote."""
    failed = [v for key in ("a", "b") if (v := _verdict(item[key], "D"))
              and fails("D", v["met"])]
    if not failed:
        return None
    adj = _verdict(item["adjudicator"], "D")
    if adj and adj["met"] == "yes" and adj["verified"]:
        return None
    return failed[0]["quote"]


def _evidence(decision: dict, text: str) -> bool:
    """A non-empty quote that really is in the text, or (Part 10b) an exclusion on a
    criterion the full text does not show (absence is the evidence)."""
    if decision.get("absence"):
        return True
    return bool((decision.get("quote") or "").strip()) and quote_found(decision["quote"], text)


def _deciding_criterion(item: dict) -> str | None:
    """Criterion of a final decision made after screening (human resolution)."""
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


AUTOMATION = ("d_title_rule", "arxiv_rule")  # code rules decided before any LLM call


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
        "title_rule": sum("d_title_rule" in d["flags"] for d in decisions),
        "arxiv_rule": sum("arxiv_rule" in d["flags"] for d in decisions),
        # PRISMA "marked as ineligible by automation tools", per deciding criterion.
        "automation_by_criterion": dict(Counter(d["criterion"] for d in decisions
                                                if d["final"] == "exclude"
                                                and d.get("resolved_by") in AUTOMATION)),
        "evidence_term_downgrades": sum(any(f.startswith("no_evidence_term") for f in
                                            d["flags"]) for d in decisions),
        "ft_absence": sum(any(f.startswith("ft_absence") for f in d["flags"])
                          for d in decisions),
        "eligibility_quotes": sum(any(f.startswith("eligibility_quote") for f in d["flags"])
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
        "title_rule": base["title_rule"], "eligibility_quotes": base["eligibility_quotes"],
        "arxiv_rule": base["arxiv_rule"],
        "evidence_term_downgrades": base["evidence_term_downgrades"],
        "ft_absence": base["ft_absence"],
        "pairs": base["pairs"], "agreement": base["agreement"], "kappa": base["kappa"],
    }


PRISMA_KEY = {"ta": "title_abstract", "ft": "full_text"}
STAGE_NAME = {"ta": "ta_screening", "ft": "ft_screening"}


def _counts(stage: str, decisions: list[dict], rnd: int) -> dict:
    return {**(ta_counts(decisions) if stage == "ta" else ft_counts(decisions)), "round": rnd}


def _included(decisions: list[dict]) -> list[str]:
    return [d["paper_id"] for d in decisions if d["final"] == "include"]


# --- Title / abstract stage -------------------------------------------------------------

LIMIT_KEYS = ("screen_limit", "records_available", "not_screened")


def screening_sample(papers: list[dict], limit: int | None) -> list[dict]:
    """Part 9b: the ``limit`` best-ranked papers (``rank`` = best position in any database's
    relevance order, so every database contributes its top hits; ties and papers without a
    rank keep the retrieval order). ``None`` / 0 = every paper. Retrieval order is kept."""
    if not limit or limit >= len(papers):
        return list(papers)
    order = sorted(range(len(papers)),
                   key=lambda i: (papers[i].get("rank") or float("inf"), i))
    keep = set(order[:limit])
    return [p for i, p in enumerate(papers) if i in keep]


def limit_counts(n_available: int, n_screened: int, limit: int | None) -> dict:
    """PRISMA keys of the limit: the papers above it are 'not screened (limit N)'."""
    return {"screen_limit": limit or None, "records_available": n_available,
            "not_screened": n_available - n_screened}


def make_ta_screening(llm: LLM, *, workers: int = 8, limit: int | None = None):
    """Title/abstract screening of ``unique_records``. ``limit`` (Part 9b, config
    ``screen_limit``, ``sr-review --limit``) screens only the ``limit`` best-ranked records;
    the others are counted as not screened, so the PRISMA diagram stays honest."""

    def ta_screening(state: ReviewState) -> dict:
        protocol = state["protocol"]
        rnd = stage_round("ta_screening", state.get("completed_stages", []))
        available = state.get("unique_records", [])
        papers = screening_sample(available, limit)

        def one(paper):
            return screen_record(llm, protocol, paper, ta_evidence(paper), stage="ta", rnd=rnd)

        results = _parallel(one, papers, workers)
        decisions = [item for item, _ in results]
        counts = {**_counts("ta", decisions, rnd),
                  **limit_counts(len(available), len(papers), limit)}
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
                    cid: index.search(criterion_query(protocol, cid, text), k=candidates,
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
                        ranked, u = rerank(llm, criterion_query(protocol, cid, text), ranked,
                                          top=candidates)
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
        prev = state.get("prisma_counts", {}).get(PRISMA_KEY[stage]) or {}
        kept = {k: prev[k] for k in LIMIT_KEYS if k in prev}   # the screening limit's counts
        update = {
            field: new,
            "prisma_counts": {**state.get("prisma_counts", {}),
                              PRISMA_KEY[stage]: {**_counts(stage, decisions, rnd), **kept}},
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
                   f"kappa {ta['kappa']}"
                   + (f"; {ta['not_screened']} records above the screening limit of "
                      f"{ta['screen_limit']} not screened" if ta.get("not_screened") else ""))
    if (ft := pc.get("full_text")):
        out.append(f"Full-text screening (round {ft['round']}): {ft['reports_sought']} sought, "
                   f"{ft['reports_not_retrieved']} without full text, "
                   f"{ft['studies_included']} included")
    return out
