"""Writer (Part 11f): Introduction, Discussion, Conclusions and the abstract's Background,
Results and Conclusions, drafted by one LLM call from a facts sheet built by code.

Guards (code, after the call; a section that fails is replaced by text built by code):
- every number written must be in the facts sheet (same rule as the synthesis narrative:
  printed there, or a rounding of a printed number); study counts 0..n and 95 allowed;
- every study label (S12) must be an included study;
- no outside sources: no "et al.", reference brackets, URLs or DOIs (the writer sees no
  literature, so any citation would be invented).
The writer gets only the facts sheet and the question (not the planner's rationale note,
which is about the protocol, not the topic). Without an
LLM (tests, ``--report-only`` without ``--ai-text``) every section is the code version.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from ..agents.extraction import numbers
from ..agents.synthesis import number_supported

AGENT = "writer"
SECTIONS = ("abstract_background", "abstract_results", "abstract_conclusion",
            "introduction", "discussion", "conclusion", "research_agenda")
_LABEL = re.compile(r"\bS\d+\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_OUTSIDE = re.compile(r"\bet al\b|\[\d+(?:[,–-]\s*\d+)*\]|https?://|www\.|\bdoi\b", re.I)


class Prose(BaseModel):
    abstract_background: str = Field(description="One or two sentences: why the question "
                                                 "matters (from the rationale) and the aim.")
    abstract_results: str = Field(description="Two or three sentences: what was found, "
                                              "with the main counts from the facts.")
    abstract_conclusion: str = Field(description="One or two sentences, cautious.")
    introduction: list[str] = Field(description="Two paragraphs: context and gap (from the "
                                                "rationale, no outside facts), then the "
                                                "objective of this review.")
    discussion: list[str] = Field(description="Three paragraphs: main findings; limits of "
                                              "the evidence and of this review; implications "
                                              "for research.")
    conclusion: str = Field(description="One paragraph, two or three sentences.")
    research_agenda: list[str] = Field(default_factory=list, description=(
        "One short paragraph per evidence gap marked 'propose' in the facts, in the "
        "facts' order, starting with its id (e.g. 'G1.'): the gap in one phrase, then a "
        "concrete study that would close it (what to build or measure, on what data and "
        "device, reported how), citing the closest included studies by label. A "
        "proposal, not a result: no expected numbers."))


RULES = """Rules (checked by code; a section that breaks one is replaced):
- Use ONLY the facts below. Do not add any fact, number, study, author, dataset or result
  that is not in them. General statements without numbers are allowed in the Introduction
  only when they follow from the question itself.
- Say nothing about other research or the state of the literature (e.g. "few reviews
  exist", "recent studies show") unless the facts say it. Do not mention PICO or any
  framework by name, and do not describe how the pipeline works (the Methods do).
- Every number you write must appear in the facts (you may round it). Do not compute new
  numbers (no sums, differences or percentages of your own).
- No references to outside literature: no author names, "et al.", years in brackets,
  numbered references, URLs or DOIs. Study labels S1, S2... may be cited only as given.
- Do not rank models or name a "best" model: the studies used different datasets,
  train/test splits and class sets. Say so where it matters.
- Be cautious and specific: if the facts show that most studies were judged on the
  abstract only, or that risk of bias is mostly unclear, say what this means.
- Frame the review around its evidence gaps: name the main gap in abstract_results and
  the proposed direction in abstract_conclusion. A gap marked "not confirmed" may be a
  gap of this sample only: do not present it as a gap in the field.
- Plain academic English, no bullet points, no headings inside the paragraphs.
"""


def prompt(protocol: dict, facts: str) -> str:
    return ("You write the framing sections of a PRISMA 2020 systematic review draft. The "
            "methods and results sections are already written by code; your text must "
            "agree with them.\n"
            f"Review question: {protocol.get('question', '')}\n\n"
            f"{RULES}\nFACTS (the only source you may use):\n{facts}\n")


def check(text: str, facts: str, labels: list[str], n: int) -> list[str]:
    """Problems in one section (empty = it passes)."""
    flags = []
    allowed = numbers(_LABEL.sub(" ", facts)) + [float(i) for i in range(n + 1)] + [95.0]
    stripped = _LABEL.sub(" ", text)
    tokens = _NUM.findall(re.sub(r"(?<=\d),(?=\d{3}\b)", "", stripped))
    bad = [t for t in tokens if not number_supported(t, allowed)]
    bad += [f"{x:g}" for x in numbers(re.sub(r"\d", " ", stripped))
            if not number_supported(f"{x:g}", allowed)]
    if bad:
        flags.append("number not in the facts: " + ", ".join(dict.fromkeys(bad)))
    unknown = [c for c in _LABEL.findall(text) if c not in set(labels)]
    if unknown:
        flags.append("unknown study: " + ", ".join(dict.fromkeys(unknown)))
    ref = _OUTSIDE.search(text)
    if ref:
        flags.append("outside reference: " + ref.group(0))
    if not text.strip():
        flags.append("empty")
    return flags


def code_text(protocol: dict, facts_d: dict) -> dict[str, str | list[str]]:
    """The code version of every section (used when there is no LLM or a check fails). The
    protocol's rationale is a planner note, not article text: it is only a hint for the
    writer, never printed."""
    q = protocol.get("question", "")
    n, inc = facts_d.get("screened"), facts_d.get("included")
    return {
        "abstract_background": f"This review asks: {q}",
        "abstract_results": f"{facts_d.get('identified')} records were identified and "
                            f"{n} were screened; {inc} studies were included. "
                            + (facts_d.get("headline") or ""),
        "abstract_conclusion": "The included studies describe the question but report their "
                               "results in ways that cannot be combined; the evidence needs "
                               "a human check before use.",
        "introduction": [f"This review asks: {q} It follows the PRISMA 2020 statement; the "
                         "methods below describe how each step was done and checked."],
        "discussion": ["The main findings are summarised in the Summary of findings above.",
                       "The limits of the evidence and of this review are listed under "
                       "Limitations below."],
        "conclusion": "This draft maps the studies that answer the question; a reviewer "
                      "should check the included studies and the extracted values before "
                      "drawing conclusions.",
        "research_agenda": [f"{g['id']}. {g['statement']}: a study designed to close this "
                            "gap, reported with the items listed in the Methods, is needed"
                            + (f" (closest included studies: {', '.join(g['closest'][:4])})."
                               if g.get("closest") else ".")
                            for g in facts_d.get("gaps") or [] if g.get("propose")],
    }


def write(llm, protocol: dict, facts: str, facts_d: dict, labels: list[str]) -> dict:
    """{"sections": {name: text or [paragraphs]}, "source": {name: "ai" | "code"},
    "flags": {name: [...]}, "usage": dict | None, "error": str | None}."""
    base = code_text(protocol, facts_d)
    out = {"sections": dict(base), "source": {k: "code" for k in SECTIONS}, "flags": {},
           "usage": None, "error": None}
    if llm is None:
        return out
    try:
        ans, usage = llm.call_with_usage(Prose, prompt(protocol, facts), agent=AGENT)
        out["usage"] = usage.as_dict()
    except Exception as exc:                       # no answer: the code text stays
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    n = int(facts_d.get("included") or 0)
    for name in SECTIONS:
        value = getattr(ans, name)
        paras = value if isinstance(value, list) else [value]
        paras = [p.strip() for p in paras if p and p.strip()]
        if name == "research_agenda" and not (facts_d.get("gaps") or []):
            continue                                   # nothing to propose: code text ([])
        flags = [f for p in paras for f in check(p, facts, labels, n)] if paras else ["empty"]
        if flags:
            out["flags"][name] = flags
            continue
        out["sections"][name] = paras if isinstance(value, list) else paras[0]
        out["source"][name] = "ai"
    return out
