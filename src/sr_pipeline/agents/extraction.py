"""Extraction + risk of bias: one evidence-table row and one risk-of-bias assessment per
included study.

**Fields come from the protocol** (``extraction_fields``), built by code: a fixed core
(design, country, setting, sample sizes, age, sex, intervention and its dose, comparator,
follow-up), then one block per PICO outcome (``O1_measure``, ``O1_timepoints`` + a
structured result), plus any extra fields the protocol lists
(``protocol['extraction_fields'] = {key: question}``, editable at approval). The LLM
answers a list keyed by field id; a field it skips is ``not reported``.

**Never invented** (code, not prompt). Every field is ``quote -> chunk_id -> value``
(evidence written before the claim). A value is accepted (``status: reported``) only if

- its quote is found in the passages the model was shown (``quote_found``: case,
  punctuation and spacing ignored, ``...`` splits a quote into parts), and
- every number in the value appears in the quote (raw text: ``72.3`` must be printed as
  ``72.3``). Totals, differences, percentages or converted units the paper does not print
  are refused.

Otherwise the field is ``not reported`` and the model's value stays in ``value_claimed``
with a flag (``no_quote_*``, ``quote_not_found_*``, ``number_not_in_quote_*``). Numeric
results (per-arm n / mean / SD, effect estimates) are checked the same way, number by
number: a number absent from its quote becomes ``None`` (listed in ``numbers_dropped``).
Number words count as numbers ("Thirty two participants", "twice a week"); an arm's n
may also be printed in the verified sample-size quotes (n_enrolled / n_analysed); a
result left without any printed number is ``not reported``.

The chunk id of a quote is set by code (the passage that contains it), so every value
in the table cites a real passage.

**Passages** (RAG, ``runs/<id>/index`` from full-text screening): per paper, one query
per field group (methods, participants, intervention, one per outcome; the outcome
query adds the protocol criteria that name it, e.g. "GDS, GDS-15, ..."), hybrid search
restricted to that paper, LLM re-rank, top passages. The paper's title + abstract chunk
is always shown. Papers without full text (``abstract_only``) or missing from the index
are extracted from their abstract and flagged: most fields will be ``not reported``,
which is the correct answer.

**Multi-arm trials**: the arm pair judged at full-text screening (``comparison``) is
given as a hint, so extraction reports the same comparison screening accepted.

**Risk of bias**, tool chosen by code from the extracted design category:

- randomized trial -> **RoB 2** (Sterne 2019), for the review's first outcome: 5 domains,
  low / some concerns / high; overall: any high -> high, all low -> low, else some
  concerns (RoB 2's "several some concerns may be high" is a judgement, not applied);
- non-randomized intervention study -> **ROBINS-I** (Sterne 2016): 7 domains,
  low / moderate / serious / critical / no information; overall = most severe domain,
  "no information" when nothing is serious or critical but a domain lacks information;
- diagnostic accuracy study -> **QUADAS-2** (Whiting 2011): 4 domains, low / high /
  unclear; no overall score (the tool advises against one), only ``any_high``.
  Added beyond the plan: two of the three evaluation topics (T1, T2) are
  diagnostic-accuracy reviews;
- anything else -> a short generic **checklist** (5 items, low / high / unclear).

If the design category has no verified quote, the protocol decides when it allows one
design only (every include passed D at screening), else the checklist; the path is
stored in ``tool_chosen_by``. A domain judgment needs a verified quote too: without one
it becomes the tool's "no evidence" level (RoB 2: some concerns, ROBINS-I: no
information, QUADAS-2 / checklist: unclear), flag ``unverified_<domain>``. RoB 2 D1 'low'
also needs a quote naming how allocation was concealed or generated (RoB 2 algorithm;
"randomly assigned" alone -> some concerns, flag ``d1_no_method``).

**Diagnostic 2x2 tables** (Part 8, step 0): studies assessed with QUADAS-2 get one more
call (``DiagnosticExtraction``) over a 2x2 query plus the paper's table chunks: per index
test, ``index_test``, ``reference_standard``, ``threshold``, ``tp``/``fp``/``fn``/``tn`` and
the printed sensitivity / specificity, each number verified in the quote. A missing cell is
never derived; cells that disagree with the printed sensitivity / specificity are flagged
(``check_two_by_two``). Other studies have ``diagnostic: None``.

**Tables**: each outcome query also receives the paper's best table chunks (JATS tables
are chunks of their own), added after re-ranking.

Rounds: items carry ``paper_id`` and ``round``; read them through ``latest()``. An LLM
error never stops the stage: the item is written with every field ``not reported`` (or
every domain at the no-evidence level) and an ``llm_error`` flag.

Concurrency: like full-text screening, all index searches run in the stage's thread
(the embedder's SQLite connection is not shared), then the LLM calls fan out.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import LLM, Usage
from ..rag import HybridIndex, chunk_paper, rerank
from ..rag.index import tokenize
from ..state import ReviewState, latest, stage_round
from .screening import _norm, _parallel, _sum_usage, quote_found

NOT_REPORTED = "not reported"


# --- Fields ------------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    key: str
    question: str
    group: str  # retrieval query group: methods, participants, intervention, O1, ..., extra


def extraction_fields(protocol: dict) -> list[FieldSpec]:
    """Fields to extract, in table order, derived from the protocol."""
    pico = protocol.get("pico", {})
    pop = pico.get("population") or "the population"
    inter = pico.get("intervention") or "the intervention"
    comp = pico.get("comparator") or "the control"
    out = [
        FieldSpec("design", "Study design as stated (e.g. parallel-group randomized "
                  "controlled trial, cluster RCT, cohort study, diagnostic accuracy study)",
                  "methods"),
        FieldSpec("country", "Country or countries where the study was done", "methods"),
        FieldSpec("setting", "Setting (e.g. community, nursing home, hospital, online)",
                  "methods"),
        FieldSpec("n_enrolled", "Number of participants enrolled or randomized (total "
                  "and per arm, as printed)", "participants"),
        FieldSpec("n_analysed", "Number of participants analysed (total and per arm, as "
                  "printed)", "participants"),
        FieldSpec("age", "Participants' age (mean and SD, median, or range, as printed)",
                  "participants"),
        FieldSpec("sex", "Sex or gender of participants (e.g. number or % women)",
                  "participants"),
        FieldSpec("population", f"Who the participants were (review population: {pop}): "
                  "health status, eligibility", "participants"),
        FieldSpec("intervention", f"Intervention ({inter}): type and content",
                  "intervention"),
        FieldSpec("intervention_dose", "Intervention dose: duration, frequency, session "
                  "length, intensity", "intervention"),
        FieldSpec("comparator", f"Comparator / control group (review comparator: {comp})",
                  "intervention"),
        FieldSpec("follow_up", "Timing of outcome assessments / length of follow-up",
                  "intervention"),
    ]
    for i, outcome in enumerate(pico.get("outcomes") or [], 1):
        out += [FieldSpec(f"O{i}_measure", f"Instrument or scale used to measure {outcome}",
                          f"O{i}"),
                FieldSpec(f"O{i}_timepoints", f"Timepoints at which {outcome} was measured",
                          f"O{i}")]
    for key, question in (protocol.get("extraction_fields") or {}).items():
        out.append(FieldSpec(str(key), str(question), "extra"))
    return out


def outcomes(protocol: dict) -> list[tuple[str, str]]:
    return [(f"O{i}", o) for i, o in
            enumerate(protocol.get("pico", {}).get("outcomes") or [], 1)]


def _criteria_naming(protocol: dict, outcome: str) -> list[str]:
    """Protocol criteria sharing a (stemmed) word with the outcome: they often list the
    instruments ("Depression measured with GDS, GDS-15, ..."), which BM25 finds in tables."""
    stems = set(tokenize(outcome))
    crit = (protocol.get("inclusion_criteria") or []) + (protocol.get("exclusion_criteria")
                                                         or [])
    return [c for c in crit if stems & set(tokenize(c))]


def group_queries(protocol: dict) -> dict[str, str]:
    """One retrieval query per field group."""
    pico = protocol.get("pico", {})
    fields = extraction_fields(protocol)
    queries = {
        "methods": "Study design, randomization, country and setting where the study was "
                   "conducted",
        "participants": f"Participants ({pico.get('population', '')}): number enrolled, "
                        "randomized and analysed per group, age mean SD, sex, baseline "
                        "characteristics",
        "intervention": f"Intervention ({pico.get('intervention', '')}) content, duration, "
                        f"frequency, intensity; control group ({pico.get('comparator', '')}); "
                        "follow-up assessments",
    }
    for oid, outcome in outcomes(protocol):
        extra = " ".join(_criteria_naming(protocol, outcome))
        queries[oid] = (f"Results for {outcome}: measured with which instrument, at which "
                        "timepoints; mean, standard deviation and number of participants "
                        "per group after the intervention; between-group difference, "
                        f"confidence interval, p value. {extra}").strip()
    extra = [f.question for f in fields if f.group == "extra"]
    if extra:
        queries["extra"] = "; ".join(extra)
    return queries


# --- Schemas -------------------------------------------------------------------------------
# Field order matters: evidence (quote, chunk) before the claim (value, numbers).

class FieldValue(BaseModel):
    key: str = Field(description="Field id as listed, e.g. 'n_enrolled', 'O1_measure'.")
    quote: str = Field("", description="Exact words copied from a passage that give the "
                       "value; empty if the passages do not report it.")
    chunk_id: str = Field("", description="Id of the passage quoted, e.g. 'doi:10.1/x::12'.")
    value: str = Field("", description="The value, using only what the quote says; numbers "
                       "copied as printed. Empty if not reported.")


class ArmResult(BaseModel):
    arm: str = Field(description="Arm name as in the paper, e.g. 'exercise', 'control'.")
    timepoint: str = Field("", description="e.g. 'post-intervention', '12 weeks'.")
    n: float | None = Field(None, description="Participants analysed in this arm; null if "
                            "not printed.")
    mean: float | None = Field(None, description="Mean score; null if not printed.")
    sd: float | None = Field(None, description="Standard deviation; null if not printed "
                             "(do not convert an SE or CI).")


class Estimate(BaseModel):
    measure: str = Field(description="As printed, e.g. 'mean difference', 'SMD', "
                         "'sensitivity', 'specificity', 'odds ratio'.")
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: str = Field("", description="As printed, e.g. '0.03', '<0.001'.")


class OutcomeResult(BaseModel):
    outcome: str = Field(description="Outcome id as listed: 'O1', 'O2', ...")
    quote: str = Field("", description="Exact words (a sentence or a table row) holding "
                       "every number below; use '...' between separate parts.")
    chunk_id: str = ""
    arms: list[ArmResult] = Field(default_factory=list)
    estimates: list[Estimate] = Field(default_factory=list)


DesignCategory = Literal["randomized trial", "non-randomized intervention study",
                         "diagnostic accuracy study", "other"]


class TwoByTwo(BaseModel):
    """Part 8, step 0: one diagnostic 2x2 table (one index test at one threshold)."""
    index_test: str = Field(description="Index test as named in the paper, e.g. 'Apple Watch "
                            "ECG app', 'GPT-4 zero-shot'.")
    reference_standard: str = Field("", description="Reference standard as named, e.g. "
                                    "'12-lead ECG read by cardiologists', 'human reviewers'.")
    threshold: str = Field("", description="Threshold / cut-off / setting, as printed; empty "
                           "if none.")
    quote: str = Field("", description="Exact words (a table row or sentences) printing every "
                       "number below; '...' between separate parts.")
    chunk_id: str = ""
    tp: int | None = Field(None, description="True positives, only if printed.")
    fp: int | None = Field(None, description="False positives, only if printed.")
    fn: int | None = Field(None, description="False negatives, only if printed.")
    tn: int | None = Field(None, description="True negatives, only if printed.")
    sensitivity: float | None = Field(None, description="Sensitivity as printed (e.g. 94.2 "
                                      "for 94.2%, or 0.942); null if not printed.")
    specificity: float | None = Field(None, description="Specificity as printed; null if "
                                      "not printed.")


class DiagnosticExtraction(BaseModel):
    tables: list[TwoByTwo] = Field(default_factory=list, description="One entry per index "
                                   "test (and threshold) with accuracy results; empty if the "
                                   "passages report none.")


class Extraction(BaseModel):
    fields: list[FieldValue] = Field(description="One entry per field, every field, in the "
                                     "order listed.")
    design_category: DesignCategory = Field(
        description="randomized trial = participants allocated to arms at random; "
                    "non-randomized intervention study = intervention compared without "
                    "random allocation (incl. cohort studies of an intervention); "
                    "diagnostic accuracy study = a test compared against a reference "
                    "standard; other = anything else.")
    results: list[OutcomeResult] = Field(default_factory=list, description="Per outcome "
                                         "listed: the results as printed; omit an outcome "
                                         "with no results in the passages.")


# --- Risk-of-bias tools ------------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    key: str
    name: str
    levels: tuple[str, ...]       # from best to worst (no-evidence level included)
    no_evidence: str              # level of a domain judged without a verified quote
    domains: tuple[tuple[str, str, str], ...]  # (id, name, what to judge / query)


TOOLS = {
    "rob2": Tool("rob2", "RoB 2 (Cochrane risk-of-bias tool for randomized trials)",
                 ("low", "some concerns", "high"), "some concerns", (
        ("D1", "Bias arising from the randomization process",
         "Was the allocation sequence random and concealed until assignment? Were groups "
         "balanced at baseline?"),
        ("D2", "Bias due to deviations from intended interventions",
         "Were participants and staff aware of the assigned intervention? Deviations, "
         "adherence; intention-to-treat analysis?"),
        ("D3", "Bias due to missing outcome data",
         "Were outcome data available for nearly all participants? Dropouts, attrition, "
         "how missing data were handled."),
        ("D4", "Bias in measurement of the outcome",
         "Was the outcome measure appropriate and the same in all groups? Were outcome "
         "assessors blinded to the intervention?"),
        ("D5", "Bias in selection of the reported result",
         "Was the trial registered or a protocol / analysis plan published before the "
         "results? Were all planned outcomes and analyses reported?"))),
    "robins_i": Tool("robins_i", "ROBINS-I (risk of bias in non-randomized studies of "
                     "interventions)",
                     ("low", "moderate", "serious", "critical", "no information"),
                     "no information", (
        ("D1", "Bias due to confounding", "Were confounders measured and controlled for "
         "(adjustment, matching)?"),
        ("D2", "Bias in selection of participants", "Was selection into the study related "
         "to intervention and outcome? Start of follow-up and intervention aligned?"),
        ("D3", "Bias in classification of interventions", "Were intervention groups "
         "clearly defined and recorded at the start?"),
        ("D4", "Bias due to deviations from intended interventions", "Deviations from the "
         "intended intervention, co-interventions, adherence."),
        ("D5", "Bias due to missing data", "Were outcome data available for nearly all "
         "participants? Exclusions, dropouts."),
        ("D6", "Bias in measurement of outcomes", "Outcome measure comparable across "
         "groups; assessors aware of the intervention?"),
        ("D7", "Bias in selection of the reported result", "Selective reporting among "
         "outcomes, analyses or subgroups; pre-registered plan?"))),
    "quadas2": Tool("quadas2", "QUADAS-2 (quality assessment of diagnostic accuracy "
                    "studies)", ("low", "unclear", "high"), "unclear", (
        ("D1", "Patient selection", "Consecutive or random sample? Case-control design "
         "avoided? Inappropriate exclusions avoided?"),
        ("D2", "Index test", "Index test interpreted without knowledge of the reference "
         "standard? Threshold pre-specified?"),
        ("D3", "Reference standard", "Is the reference standard likely to classify the "
         "condition correctly? Interpreted blind to the index test?"),
        ("D4", "Flow and timing", "Appropriate interval between tests? Same reference "
         "standard for all? All participants included in the analysis?"))),
    "checklist": Tool("checklist", "Generic quality checklist", ("low", "unclear", "high"),
                      "unclear", (
        ("C1", "Clear aims and appropriate sample", "Are the aims stated and the sample "
         "recruited appropriately for them?"),
        ("C2", "Valid outcome measurement", "Was the outcome measured with a valid, "
         "reliable instrument, the same way for everyone?"),
        ("C3", "Confounding addressed", "Were confounders identified and dealt with in "
         "design or analysis?"),
        ("C4", "Complete follow-up", "Was follow-up complete, or were losses described and "
         "handled?"),
        ("C5", "Appropriate analysis", "Were statistical methods appropriate and results "
         "reported fully?"))),
}

TOOL_FOR = {"randomized trial": "rob2", "non-randomized intervention study": "robins_i",
            "diagnostic accuracy study": "quadas2", "other": "checklist"}

# RoB 2, domain 1: 'low' needs concealed allocation (signalling question 1.2); a study that
# only says "randomly assigned" is 'some concerns' (RoB 2 algorithm: concealment no
# information -> some concerns). Seen live (Part 7 smoke): 8 of 11 D1 'low' judgments
# quoted only "randomly assigned". Code check: the quote must name an allocation method.
ROB2_D1_RULE = ("D1 is 'low' only if the passages say how allocation was concealed or the "
                "sequence generated (e.g. central or computer-generated allocation, sealed "
                "opaque envelopes, random number table, blocks); the words 'randomized' or "
                "'randomly assigned' alone are 'some concerns'.")
ALLOCATION_METHOD = re.compile(
    r"conceal|sealed|opaque|envelope|central|computer|random number|random[- ]?numbers|"
    r"block|stratif|minimi[sz]|sequence|software|lottery|coin|draw|independent|third party|"
    r"pharmacy|allocation was|allocated by|web[- ]based|telephone", re.I)

ALL_LEVELS = Literal["low", "some concerns", "moderate", "serious", "critical",
                     "no information", "unclear", "high"]


class DomainJudgment(BaseModel):
    domain: str = Field(description="Domain id as listed: 'D1', 'D2', ... or 'C1', ...")
    quote: str = Field("", description="Exact words copied from a passage that support the "
                       "judgment; empty if the passages say nothing about this domain.")
    chunk_id: str = ""
    judgment: ALL_LEVELS = Field(description="One of the levels listed for this tool.")
    reason: str = Field(description="One sentence: why this judgment, from the quote.")


class RiskOfBias(BaseModel):
    domains: list[DomainJudgment] = Field(description="One judgment per domain, every "
                                          "domain, in the order listed.")


def choose_tool(design_category: str | None, verified: bool, protocol: dict
                ) -> tuple[str, str]:
    """(tool key, how it was chosen)."""
    if design_category and verified:
        return TOOL_FOR.get(design_category, "checklist"), "extracted design"
    designs = protocol.get("study_designs") or []
    if len(designs) == 1:  # every include passed D (that one design) at screening
        d = designs[0].lower()
        if "randomi" in d:
            return "rob2", "protocol design (design not verified)"
        if "diagnostic" in d:
            return "quadas2", "protocol design (design not verified)"
        if "non-randomi" in d or "cohort" in d or "quasi" in d:
            return "robins_i", "protocol design (design not verified)"
    return "checklist", "default (design not verified)"


def overall(tool: Tool, judgments: list[str]) -> str | None:
    """Overall judgment by code (None for QUADAS-2: no summary score)."""
    if tool.key == "quadas2" or not judgments:
        return None
    if tool.key == "robins_i":
        order = ["low", "moderate", "serious", "critical"]
        known = [j for j in judgments if j in order]
        worst = max(known, key=order.index) if known else "no information"
        if worst in ("low", "moderate") and "no information" in judgments:
            return "no information"
        return worst
    if "high" in judgments:
        return "high"
    if all(j == "low" for j in judgments):
        return "low"
    return tool.no_evidence  # RoB 2: some concerns; checklist: unclear


# --- Checks ---------------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Number words are read as numbers: journals spell out a number that starts a sentence
# ("Thirty two participants completed ..."), and doses are often in words ("four days a
# week", "twice a week"). Seen live (Part 7 smoke): 11 values refused for this alone.
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())} | {"once": 1, "twice": 2,
                                                              "thrice": 3}
_TENS = {w: 10 * i for i, w in enumerate(
    "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def spelled_numbers(text: str) -> list[float]:
    """Numbers written in words, up to 999 ("thirty two", "fifty-two", "one hundred and
    twenty", "twice")."""
    out, cur, active = [], 0, False
    for w in re.findall(r"[a-z]+", (text or "").lower()) + ["."]:
        if w in _UNITS or w in _TENS:
            cur += _UNITS.get(w, _TENS.get(w, 0))
            active = True
        elif w == "hundred" and active:
            cur *= 100
        elif w == "and" and active and cur >= 100:
            continue
        else:
            if active:
                out.append(float(cur))
            cur, active = 0, False
    return out


def numbers(text: str) -> list[float]:
    """Numbers in a text, in digits or words ("1,234" -> 1234; signs ignored)."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    return [float(n) for n in _NUMBER.findall(text)] + spelled_numbers(text)


def number_in(x: float, *quotes: str) -> bool:
    return any(abs(abs(x) - n) < 1e-9 for q in quotes for n in numbers(q))


def _empty(value: str) -> bool:
    return _norm_value(value) in ("", "not reported", "nr", "n a", "na", "none",
                                  "not stated", "unknown", "not available")


def _norm_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def locate(quote: str, chunks: list[dict], claimed: str = "") -> str:
    """Chunk id that contains the quote: the model's if right, else the first that does
    ('' if the quote spans passages)."""
    by_id = {c["chunk_id"]: c for c in chunks}
    if claimed in by_id and quote_found(quote, by_id[claimed]["text"]):
        return claimed
    return next((c["chunk_id"] for c in chunks if quote_found(quote, c["text"])), "")


def table_quote_found(quote: str, chunks: list[dict]) -> str:
    """Chunk id of the one passage holding a table-row quote that skips columns, else ''.

    Seen live (Part 8): from Makizako's Table 3 the model quoted the exercise and control
    columns of the GDS-15 row and left out the horticulture columns without writing '...',
    so the exact check refused a correct reading. A quote with '|' is split into cells; the
    cells must appear whole and in order inside ONE passage (never stitched across
    passages). Numbers are still checked against the quote itself."""
    if "|" not in quote:
        return ""
    parts = [f" {p} " for p in (_norm(x) for x in re.split(r"\||\.\.\.|…", quote)) if p]
    for c in chunks:
        hay, pos = f" {_norm(c['text'])} ", 0
        for part in parts:
            i = hay.find(part, pos)
            if i < 0:
                break
            pos = i + len(part) - 1
        else:
            return c["chunk_id"]
    return ""


def _find_quote(quote: str, chunks: list[dict], text: str, claimed: str
                ) -> tuple[bool, str, bool]:
    """(found, chunk id, found only as a table row with skipped columns)."""
    if quote and quote_found(quote, text):
        return True, locate(quote, chunks, claimed), False
    cid = table_quote_found(quote or "", chunks)
    return bool(cid), cid, bool(cid)


def check_fields(raw: list[dict], specs: list[FieldSpec], chunks: list[dict]
                 ) -> tuple[dict[str, dict], list[str]]:
    """Field values checked by code (see module docstring). Returns {key: field}, flags."""
    text = "\n".join(c["text"] for c in chunks)
    by_key: dict[str, dict] = {}
    for v in raw:
        k = (v.get("key") or "").strip()
        if k and k not in by_key:
            by_key[k] = v
    out, flags = {}, []
    for spec in specs:
        v = by_key.get(spec.key) or {}
        value, quote = (v.get("value") or "").strip(), (v.get("quote") or "").strip()
        item: dict = {"status": NOT_REPORTED, "value": NOT_REPORTED, "quote": "",
                      "chunk_id": ""}
        if _empty(value):
            out[spec.key] = item
            continue
        if not quote:
            flags.append(f"no_quote_{spec.key}")
        elif not quote_found(quote, text):
            flags.append(f"quote_not_found_{spec.key}")
        elif missing := [n for n in numbers(value) if not number_in(n, quote)]:
            flags.append(f"number_not_in_quote_{spec.key}")
            item["numbers_not_in_quote"] = missing
        else:
            out[spec.key] = {"status": "reported", "value": value, "quote": quote,
                             "chunk_id": locate(quote, chunks, v.get("chunk_id") or "")}
            continue
        item["value_claimed"], item["quote_claimed"] = value, quote
        out[spec.key] = item
    return out, flags


def check_results(raw: list[dict], outcome_ids: list[str], chunks: list[dict],
                  n_quotes: tuple[str, ...] = ()) -> tuple[list[dict], list[str]]:
    """Structured results: quote must be found; each number must be printed in it. An
    arm's ``n`` may also be printed in ``n_quotes`` (the verified quotes of the sample-size
    fields): per-arm sizes usually sit in the participant-flow sentence, not next to the
    means. A result left with no number at all is ``not reported``."""
    text = "\n".join(c["text"] for c in chunks)
    out, flags, seen = [], [], set()
    for r in raw:
        oid = (r.get("outcome") or "").strip().upper()
        if oid not in outcome_ids or oid in seen:
            continue
        seen.add(oid)
        quote = (r.get("quote") or "").strip()
        found, cid, table_row = _find_quote(quote, chunks, text, r.get("chunk_id") or "")
        if not found:
            flags.append(f"result_quote_not_found_{oid}")
            out.append({"outcome": oid, "status": NOT_REPORTED, "claimed": r})
            continue
        if table_row:
            flags.append(f"table_row_quote_{oid}")
        dropped = []

        def keep(x, where, *also):
            if x is None or number_in(x, quote, *also):
                return x
            dropped.append(f"{where}={x:g}")
            return None

        arms = [{**a, "n": keep(a.get("n"), f"{a.get('arm')}.n", *n_quotes),
                 "mean": keep(a.get("mean"), f"{a.get('arm')}.mean"),
                 "sd": keep(a.get("sd"), f"{a.get('arm')}.sd")} for a in r.get("arms") or []]
        ests = []
        for e in r.get("estimates") or []:
            p = e.get("p_value") or ""
            if p and not all(number_in(n, quote) for n in numbers(p)):
                dropped.append(f"{e.get('measure')}.p={p}")
                p = ""
            ests.append({**e, "value": keep(e.get("value"), f"{e.get('measure')}.value"),
                         "ci_low": keep(e.get("ci_low"), f"{e.get('measure')}.ci_low"),
                         "ci_high": keep(e.get("ci_high"), f"{e.get('measure')}.ci_high"),
                         "p_value": p})
        if dropped:
            flags.append(f"number_not_in_quote_{oid}")
        kept = [x for a in arms for x in (a["mean"], a["sd"])] + [
            x for e in ests for x in (e["value"], e["ci_low"], e["ci_high"])]
        if all(x is None for x in kept) and not any(e["p_value"] for e in ests):
            # No outcome number (group sizes alone are not a result). Seen live: abstracts
            # saying only "a significant decrease", with n per arm from the flow sentence.
            flags.append(f"result_no_numbers_{oid}")
            out.append({"outcome": oid, "status": NOT_REPORTED, "claimed": r,
                        "numbers_dropped": dropped})
            continue
        out.append({"outcome": oid, "status": "reported", "quote": quote, "chunk_id": cid,
                    "arms": arms, "estimates": ests, "numbers_dropped": dropped})
    return out, flags


def _tolerance(x: float) -> float:
    """Half a unit of the last printed decimal: 94.2 -> 0.05, 94 -> 0.5, 0.942 -> 0.0005."""
    text = f"{x:.10g}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** -decimals + 1e-9


def check_two_by_two(raw: list[dict], chunks: list[dict]) -> tuple[list[dict], list[str]]:
    """Diagnostic 2x2 tables checked by code (Part 8, step 0).

    - quote found in the passages, else the table is ``not reported``;
    - every cell and the printed sensitivity / specificity must be printed in the quote,
      else that number becomes ``None`` (``numbers_dropped``); a missing cell is never
      derived from n, prevalence or sensitivity (same rule as totals in Part 7);
    - ``complete`` = the four cells verified;
    - consistency: when the cells and a printed sensitivity (specificity) both exist,
      TP/(TP+FN) (TN/(TN+FP)) must equal the printed value within half a unit of its last
      printed digit (percent if > 1). A mismatch is flagged (``consistent: False``), never
      fixed: the pooled analysis leaves such a table out.
    """
    text = "\n".join(c["text"] for c in chunks)
    out, flags = [], []
    for i, t in enumerate(raw, 1):
        quote = (t.get("quote") or "").strip()
        base = {k: t.get(k) or "" for k in ("index_test", "reference_standard", "threshold")}
        found, cid, table_row = _find_quote(quote, chunks, text, t.get("chunk_id") or "")
        if not found:
            flags.append(f"2x2_quote_not_found_{i}")
            out.append({**base, "status": NOT_REPORTED, "claimed": t})
            continue
        dropped, cells = [], {}
        for k in ("tp", "fp", "fn", "tn", "sensitivity", "specificity"):
            x = t.get(k)
            if x is not None and not number_in(float(x), quote):
                dropped.append(f"{k}={x:g}")
                x = None
            cells[k] = x
        if table_row:
            flags.append(f"table_row_quote_2x2_{i}")
        if dropped:
            flags.append(f"number_not_in_quote_2x2_{i}")
        if all(cells[k] is None for k in cells):
            flags.append(f"2x2_no_numbers_{i}")
            out.append({**base, "status": NOT_REPORTED, "claimed": t,
                        "numbers_dropped": dropped})
            continue
        complete = all(cells[k] is not None for k in ("tp", "fp", "fn", "tn"))
        item = {**base, "status": "reported", "quote": quote, "chunk_id": cid, **cells,
                "complete": complete, "consistent": None, "numbers_dropped": dropped}
        if not complete:
            flags.append(f"2x2_incomplete_{i}")
        else:
            checks = []
            for name, num, den in (("sensitivity", cells["tp"], cells["tp"] + cells["fn"]),
                                   ("specificity", cells["tn"], cells["tn"] + cells["fp"])):
                printed = cells[name]
                if printed is None or den == 0:
                    continue
                computed = num / den * (100 if printed > 1 else 1)
                ok = abs(computed - printed) <= _tolerance(printed)
                checks.append(ok)
                if not ok:
                    flags.append(f"2x2_inconsistent_{name}_{i}")
                    item.setdefault("mismatch", {})[name] = {"printed": printed,
                                                             "from_cells": round(computed, 4)}
            item["consistent"] = all(checks) if checks else None
        out.append(item)
    return out, flags


def check_domains(raw: list[dict], tool: Tool, chunks: list[dict]
                  ) -> tuple[list[dict], list[str]]:
    """Domain judgments in tool order. Missing domain, invalid level, or a judgment
    without a verified quote -> the tool's no-evidence level (flagged)."""
    text = "\n".join(c["text"] for c in chunks)
    by_id: dict[str, dict] = {}
    for d in raw:
        did = (d.get("domain") or "").strip().upper()
        if did and did not in by_id:
            by_id[did] = d
    out, flags = [], []
    for did, name, _ in tool.domains:
        d = by_id.get(did)
        item = {"domain": did, "name": name, "judgment": tool.no_evidence, "quote": "",
                "chunk_id": "", "verified": False, "reason": ""}
        if d is None:
            flags.append(f"domain_missing_{did}")
            out.append(item)
            continue
        quote, level = (d.get("quote") or "").strip(), d.get("judgment")
        item["reason"] = d.get("reason") or ""
        verified = bool(quote) and quote_found(quote, text)
        if verified:
            item.update(quote=quote, verified=True,
                        chunk_id=locate(quote, chunks, d.get("chunk_id") or ""))
        if level not in tool.levels:
            flags.append(f"invalid_level_{did}")
            item["judgment_claimed"] = level
        elif level != tool.no_evidence and not verified:
            flags.append(f"unverified_{did}")
            item["judgment_claimed"] = level
            if quote:
                item["quote_claimed"] = quote
        elif (tool.key == "rob2" and did == "D1" and level == "low"
              and not ALLOCATION_METHOD.search(quote)):
            flags.append("d1_no_method")
            item["judgment_claimed"] = level
        else:
            item["judgment"] = level
        out.append(item)
    return out, flags


# --- Prompts ---------------------------------------------------------------------------------

def evidence_block(chunks: list[dict]) -> str:
    return "Passages:\n" + "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)


def extraction_prompt(protocol: dict, paper: dict, chunks: list[dict],
                      comparison: str = "") -> str:
    pico = protocol.get("pico", {})
    fields = "\n".join(f"  {f.key}: {f.question}" for f in extraction_fields(protocol))
    outs = "\n".join(f"  {oid}: {o}" for oid, o in outcomes(protocol)) or "  (none)"
    hint = (f"Multi-arm trials: report the arms of the comparison accepted at screening: "
            f"{comparison}. Other arms are not extracted.\n" if comparison else "")
    return (
        "You extract data from one study for the evidence table of a PRISMA 2020 "
        "systematic review.\n"
        f"Review question: {protocol.get('question', '')}\n"
        f"Population: {pico.get('population', '')}; intervention: "
        f"{pico.get('intervention', '')}; comparator: {pico.get('comparator', '')}.\n\n"
        "Rules:\n"
        "- For every field, in order: first copy the exact words of a passage (quote), "
        "give that passage's id, then write the value using only what the quote says.\n"
        "- If the passages do not report a field, leave quote and value empty. 'Not "
        "reported' is a correct answer; never guess, never use general knowledge or other "
        "studies.\n"
        "- Copy numbers exactly as printed. Never compute totals, differences, means, "
        "percentages, or convert units (e.g. an SE into an SD): a value whose numbers are "
        "not in its quote is rejected.\n"
        "- Results: for each outcome listed, copy the post-intervention (or last "
        "follow-up) results per arm (n, mean, SD) and any between-group estimate with its "
        "confidence interval and p value, from one quote that prints every number (a "
        "table row or a sentence; '...' between separate parts). Leave a number null if it "
        "is not printed.\n"
        f"{hint}"
        "- design_category: judge from the design quote.\n\n"
        f"Fields:\n{fields}\n\nOutcomes:\n{outs}\n\n"
        f"Study: {paper.get('title') or '(no title)'}\n\n{evidence_block(chunks)}"
    )


DIAGNOSTIC_QUERY = ("Diagnostic accuracy of the index test against the reference standard: "
                    "true positives, false positives, false negatives, true negatives (2x2 "
                    "table, confusion matrix), sensitivity, specificity, threshold or cut-off")


def diagnostic_prompt(protocol: dict, paper: dict, chunks: list[dict]) -> str:
    return (
        "You extract diagnostic accuracy results (2x2 tables) from one study for a PRISMA "
        "2020 systematic review.\n"
        f"Review question: {protocol.get('question', '')}\n\n"
        "Rules:\n"
        "- One entry per index test (and per threshold or setting) whose accuracy against "
        "the reference standard is reported.\n"
        "- First copy the exact words (a table row or sentences) printing the numbers, give "
        "the passage id, then the numbers.\n"
        "- tp, fp, fn, tn: only if each count is printed. Never compute a cell from the "
        "number of participants, prevalence, sensitivity or specificity: leave it null.\n"
        "- sensitivity / specificity: as printed (94.2 for '94.2%'), null if not printed.\n"
        "- If the passages report no accuracy result, return an empty list.\n\n"
        f"Study: {paper.get('title') or '(no title)'}\n\n{evidence_block(chunks)}"
    )


def rob_prompt(protocol: dict, paper: dict, tool: Tool, chunks: list[dict],
               outcome: str | None) -> str:
    domains = "\n".join(f"  {did}: {name}. {what}" for did, name, what in tool.domains)
    target = (f"Assess the result for the outcome: {outcome}.\n" if outcome and
              tool.key in ("rob2", "robins_i") else "")
    return (
        f"You assess risk of bias with {tool.name} for a PRISMA 2020 systematic review.\n"
        f"Review question: {protocol.get('question', '')}\n{target}"
        f"Levels: {', '.join(tool.levels)}.\n"
        + (f"{ROB2_D1_RULE}\n" if tool.key == "rob2" else "") +
        "For every domain, in order: first copy the exact words of a passage that bear on "
        "it (quote) and its id, then give the judgment and a one-sentence reason based on "
        "that quote. Judge only what the passages report. If they say nothing about a "
        f"domain, leave the quote empty and answer '{tool.no_evidence}': any other "
        "judgment without a quote found in the passages is rejected.\n\n"
        f"Domains:\n{domains}\n\n"
        f"Study: {paper.get('title') or '(no title)'}\n\n{evidence_block(chunks)}"
    )


# --- Stage --------------------------------------------------------------------------------------

def _paper_chunks(pid: str, paper: dict, index: HybridIndex | None) -> list[dict]:
    """The paper's title + abstract chunk(s): from the index, else built from the record."""
    if index is not None and pid in index.rows_of:
        rows = index.rows_of[pid]
        head = [index.chunks[r] for r in rows if index.chunks[r]["section"]
                == "Title and abstract"]
        if head:
            return head
    return chunk_paper(paper, None)


def table_hits(index: HybridIndex | None, pid: str, query: str, top: int) -> list[dict]:
    """The paper's table chunks (section 'Table ...', JATS only), best first for the query,
    at most ``top``. Results often sit only in a table (Part 7: Makizako's GDS-15 means are
    in its Table 3, which the outcome query never reached)."""
    if index is None or top <= 0 or pid not in index.rows_of:
        return []
    n = len(index.rows_of[pid])
    hits = index.search(query, k=n, paper_ids=[pid], pool=n)
    return [h for h in hits if h["section"].startswith("Table")][:top]


def _merge(base: list[dict], groups: dict[str, list[dict]]) -> list[dict]:
    """Abstract chunks first, then each group's passages, each chunk once."""
    seen, out = set(), []
    for c in base + [h for hs in groups.values() for h in hs]:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            out.append({k: c[k] for k in ("chunk_id", "paper_id", "section", "text")})
    return out


def _screening_info(state: ReviewState) -> dict[str, dict]:
    """Latest full-text decision per paper: full-text status and the arm pair judged."""
    info = {}
    for d in latest(state.get("ft_decisions", [])):
        judged = d.get("adjudicator") or d.get("a") or {}
        info[d["paper_id"]] = {"fulltext_status": d.get("fulltext_status"),
                               "comparison": (judged or {}).get("comparison") or ""}
    return info


def included_papers(state: ReviewState) -> list[dict]:
    ids = list(dict.fromkeys(state.get("included") or []))
    by_id = {p["paper_id"]: p for p in state.get("unique_records", [])}
    return [by_id[i] for i in ids if i in by_id]


def make_extraction(llm: LLM, embedder, *, index_dir: str | Path, workers: int = 8,
                    passages: int = 5, candidates: int = 10, use_rerank: bool = True,
                    table_passages: int = 3, diagnostic_tables: int = 8):
    """Extraction + risk-of-bias stage over ``included`` (see module docstring).
    ``candidates`` / ``passages`` / ``use_rerank`` share the full-text screening settings
    (``ft_candidates``, ``ft_passages``, ``ft_rerank``). Each outcome query also gets the
    paper's ``table_passages`` best table chunks; the 2x2 query of a diagnostic study gets
    up to ``diagnostic_tables`` (Part 8, step 0). Tables are added after re-ranking, so
    they never push a re-ranked passage out."""
    index_dir = Path(index_dir)

    def _retrieve(index, pid: str, queries: dict[str, str], has_index: bool
                  ) -> dict[str, list[dict]]:
        if not has_index:
            return {}
        return {g: index.search(q, k=candidates, paper_ids=[pid]) for g, q in queries.items()}

    def _rank(queries: dict[str, str], hits: dict[str, list[dict]]
              ) -> tuple[dict[str, list[dict]], list[Usage]]:
        out, usages = {}, []
        for g, hs in hits.items():
            if use_rerank and hs:
                hs, u = rerank(llm, queries[g], hs, top=candidates)
                if u is not None:
                    usages.append(u)
            out[g] = hs[:passages]
        return out, usages

    def extraction(state: ReviewState) -> dict:
        protocol = state["protocol"]
        rnd = stage_round("extraction", state.get("completed_stages", []))
        papers = included_papers(state)
        info = _screening_info(state)
        index = None
        if (index_dir / "meta.json").exists() and papers:
            index = HybridIndex.load(index_dir, embedder)
        specs = extraction_fields(protocol)
        oids = [oid for oid, _ in outcomes(protocol)]
        queries = group_queries(protocol)

        # Phase 1, this thread: retrieval for the extraction queries.
        in_index = {p["paper_id"]: index is not None and p["paper_id"] in index.rows_of
                    for p in papers}
        hits = {p["paper_id"]: _retrieve(index, p["paper_id"], queries,
                                         in_index[p["paper_id"]]) for p in papers}
        tables = {p["paper_id"]: {f"{oid}_tables": table_hits(index, p["paper_id"],
                                                             queries[oid], table_passages)
                                  for oid in oids} for p in papers}

        def extract_one(paper):
            pid = paper["paper_id"]
            status = info.get(pid, {}).get("fulltext_status")
            comparison = info.get(pid, {}).get("comparison", "")
            item = {"paper_id": pid, "round": rnd, "title": paper.get("title"),
                    "year": paper.get("year"), "fulltext_status": status,
                    "comparison_hint": comparison, "flags": [],
                    "diagnostic": None}  # list of 2x2 tables for diagnostic studies only
            if status == "abstract_only" or not in_index[pid]:
                item["flags"].append("abstract_only")
            if not in_index[pid]:
                item["flags"].append("not_in_index")
            ranked, usages = _rank(queries, hits[pid])
            ranked |= {g: hs for g, hs in tables[pid].items() if hs}
            chunks = _merge(_paper_chunks(pid, paper, index), ranked)
            item["passages"] = {g: [h["chunk_id"] for h in hs] for g, hs in ranked.items()}
            try:
                ans, u = llm.call_with_usage(
                    Extraction, extraction_prompt(protocol, paper, chunks, comparison),
                    agent="extraction")
                usages.append(u)
                fields, f1 = check_fields([f.model_dump() for f in ans.fields], specs, chunks)
                n_quotes = tuple(fields[k]["quote"] for k in ("n_enrolled", "n_analysed")
                                 if fields[k]["status"] == "reported")
                results, f2 = check_results([r.model_dump() for r in ans.results], oids,
                                            chunks, n_quotes)
                item["flags"] += f1 + f2
                design_ok = fields["design"]["status"] == "reported"
                item.update(fields=fields, results=results,
                            design_category=ans.design_category,
                            design_category_verified=design_ok)
            except Exception as exc:  # LLM / validation / network error after retries
                item["flags"].append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
                fields, _ = check_fields([], specs, chunks)
                item.update(fields=fields, results=[], design_category=None,
                            design_category_verified=False)
            n_rep = sum(f["status"] == "reported" for f in item["fields"].values())
            item["reported"], item["requested"] = n_rep, len(specs)
            return item, usages, chunks

        ext = _parallel(extract_one, papers, workers)

        # Phase 2: tool per study, retrieval per domain (this thread), then RoB calls.
        primary = (protocol.get("pico", {}).get("outcomes") or [None])[0]
        plan = []
        for (item, _, _), paper in zip(ext, papers):
            tool_key, chosen_by = choose_tool(item["design_category"],
                                              item["design_category_verified"], protocol)
            tool = TOOLS[tool_key]
            dq = {did: f"{name}: {what}" for did, name, what in tool.domains}
            plan.append((paper, item, tool, chosen_by, dq,
                         _retrieve(index, paper["paper_id"], dq, in_index[paper["paper_id"]])))

        def rob_one(args):
            paper, ext_item, tool, chosen_by, dq, dhits = args
            pid = paper["paper_id"]
            ranked, usages = _rank(dq, dhits)
            chunks = _merge(_paper_chunks(pid, paper, index), ranked)
            item = {"paper_id": pid, "round": rnd, "title": paper.get("title"),
                    "tool": tool.key, "tool_name": tool.name, "tool_chosen_by": chosen_by,
                    "outcome": primary if tool.key in ("rob2", "robins_i") else None,
                    "flags": [f for f in ext_item["flags"] if f in ("abstract_only",
                                                                    "not_in_index")],
                    "passages": {d: [h["chunk_id"] for h in hs] for d, hs in ranked.items()}}
            try:
                ans, u = llm.call_with_usage(RiskOfBias, rob_prompt(protocol, paper, tool,
                                                                    chunks, primary),
                                             agent="risk_of_bias")
                usages.append(u)
                domains, flags = check_domains([d.model_dump() for d in ans.domains], tool,
                                               chunks)
                item["flags"] += flags
            except Exception as exc:
                item["flags"].append(f"llm_error: {type(exc).__name__}: {exc}"[:300])
                domains, _ = check_domains([], tool, chunks)
            item["domains"] = domains
            item["overall"] = overall(tool, [d["judgment"] for d in domains])
            if tool.key == "quadas2":
                item["any_high"] = any(d["judgment"] == "high" for d in domains)
            return item, usages

        rob = _parallel(rob_one, plan, workers)

        # Phase 3 (Part 8, step 0): 2x2 tables for diagnostic accuracy studies only (the
        # studies assessed with QUADAS-2: verified diagnostic design, or the protocol's
        # single design). Retrieval here, LLM calls in threads.
        dq = {"diagnostic": DIAGNOSTIC_QUERY}
        diag_plan = [(paper, item, _retrieve(index, paper["paper_id"], dq,
                                             in_index[paper["paper_id"]]),
                      table_hits(index, paper["paper_id"], DIAGNOSTIC_QUERY,
                                 diagnostic_tables))
                     for paper, item, tool, *_ in plan if tool.key == "quadas2"]

        def diag_one(args):
            paper, ext_item, dhits, dtables = args
            pid = paper["paper_id"]
            ranked, usages = _rank(dq, dhits)
            if dtables:
                ranked["diagnostic_tables"] = dtables
            chunks = _merge(_paper_chunks(pid, paper, index), ranked)
            ext_item["passages"] |= {g: [h["chunk_id"] for h in hs]
                                     for g, hs in ranked.items()}
            try:
                ans, u = llm.call_with_usage(DiagnosticExtraction,
                                             diagnostic_prompt(protocol, paper, chunks),
                                             agent="extraction")
                usages.append(u)
                tables_, flags = check_two_by_two([t.model_dump() for t in ans.tables],
                                                  chunks)
                ext_item["flags"] += flags
            except Exception as exc:
                ext_item["flags"].append(f"llm_error_2x2: {type(exc).__name__}: {exc}"[:300])
                tables_ = []
            ext_item["diagnostic"] = tables_
            return usages

        diag = _parallel(diag_one, diag_plan, workers)

        extractions = [i for i, _, _ in ext]
        robs = [i for i, _ in rob]
        summary = extraction_summary(extractions, robs)
        return {
            "extractions": extractions,
            "risk_of_bias": robs,
            "completed_stages": ["extraction"],
            "audit_log": [{"node": "extraction", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "extraction and risk of bias",
                           "index": str(index_dir) if index is not None else None,
                           **summary,
                           **_sum_usage([u for _, us, _ in ext for u in us]
                                        + [u for _, us in rob for u in us]
                                        + [u for us in diag for u in us])}],
        }

    return extraction


# --- Summary -------------------------------------------------------------------------------------

FLAG_KINDS = ("no_quote", "quote_not_found", "number_not_in_quote", "result_quote_not_found",
              "result_no_numbers", "d1_no_method", "2x2_quote_not_found", "2x2_no_numbers",
              "2x2_incomplete", "2x2_inconsistent", "llm_error_2x2",
              "unverified", "domain_missing", "invalid_level", "llm_error", "abstract_only",
              "not_in_index")


def _kind(flag: str) -> str:
    return next((k for k in FLAG_KINDS if flag.startswith(k)), flag)


def extraction_summary(extractions: list[dict], robs: list[dict]) -> dict:
    """Counts for the audit log; flags counted by kind (extraction and RoB items)."""
    flags = Counter(_kind(f) for i in extractions for f in i["flags"])
    flags.update(f"rob_{_kind(f)}" for r in robs for f in r["flags"]
                 if _kind(f) not in ("abstract_only", "not_in_index"))  # counted once above
    by_status: dict[str, list[int]] = {}
    for e in extractions:
        key = "abstract_only" if "abstract_only" in e["flags"] else "full_text"
        r = by_status.setdefault(key, [0, 0])
        r[0] += e["reported"]
        r[1] += e["requested"]
    return {
        "studies": len(extractions),
        "fields_reported": sum(e["reported"] for e in extractions),
        "fields_requested": sum(e["requested"] for e in extractions),
        "reported_by_text": {k: f"{a}/{b}" for k, (a, b) in by_status.items()},
        "results_reported": sum(r["status"] == "reported" for e in extractions
                                for r in e["results"]),
        "two_by_two": {"studies_assessed": sum(e.get("diagnostic") is not None
                                               for e in extractions),
                       "reported": sum(t["status"] == "reported" for e in extractions
                                       for t in e.get("diagnostic") or []),
                       "complete_consistent": sum(
                           bool(t.get("complete")) and t.get("consistent") is not False
                           for e in extractions for t in e.get("diagnostic") or [])},
        "tools": dict(Counter(r["tool"] for r in robs)),
        "tool_chosen_by": dict(Counter(r["tool_chosen_by"] for r in robs)),
        "overall": dict(Counter(f"{r['tool']}:{r['overall']}" for r in robs)),
        "flags": dict(flags),
    }


def describe_extraction(state: ReviewState) -> list[str]:
    """Line for the Coordinator summary."""
    ext = latest(state.get("extractions", []))
    if not ext:
        return []
    rnd = max(e["round"] for e in ext)
    ext = [e for e in ext if e["round"] == rnd]
    rob = [r for r in latest(state.get("risk_of_bias", [])) if r["round"] == rnd]
    s = extraction_summary(ext, rob)
    return [f"Extraction (round {rnd}): {s['studies']} studies, fields reported "
            f"{s['fields_reported']}/{s['fields_requested']}, results for "
            f"{s['results_reported']} outcome(s); risk of bias {s['overall']}"]
