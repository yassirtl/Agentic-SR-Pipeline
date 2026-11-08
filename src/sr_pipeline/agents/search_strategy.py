"""Search strategy: protocol -> one Boolean query per database (PRISMA item 7).

Division of labour:
- the LLM does the part that needs domain knowledge: split the question into
  concept blocks and list, for each, synonyms and spelling variants (plus MeSH
  headings and publication types for PubMed);
- code turns those concepts into each database's syntax (``Source.build_query``):
  synonyms OR-ed inside a block, blocks AND-ed together.

Which blocks. The textbook advice "search population AND intervention only,
leave outcomes to screening" assumes every hit is screened. Here at most
``max_records`` records per database are downloaded (most relevant first): when
P AND I matches 300,000 records, the relevant ones sit far beyond the cap and
are never seen, so recall is lost, not gained. Hence:
- an outcome block when the outcome is specific and defines the topic
  (e.g. depression), or when P AND I is far too broad;
- a design block only if the protocol restricts designs. It is written by code
  (validated filters, ``sources/design_filters.py``), never by the LLM: a block
  the LLM writes with role "design" is dropped and replaced.

Count and refine. Before anything is downloaded, each database is asked for its
hit count only (one small, cached request). ``max_records`` is the download
ceiling: a database is *too broad* above it (some matching records would never
be screened) and *too narrow* below ``min_hits``. If any database is out of
range, the counts go back to the LLM, which writes a narrower or broader plan;
at most ``max_refinements`` times. If a database is still above the ceiling, its
top ``max_records`` records are downloaded and the rest is reported, never
hidden: ``capped`` and ``not_retrieved`` per database (PRISMA flow diagram). Every attempt
(blocks, queries, counts, reason) is kept in ``search_strategy['attempts']`` for
the report (PRISMA-S: how the search was developed). The attempt used in the
end is the one with the fewest databases out of range (ties: the latest), so a
refinement that overshoots does not replace a better earlier plan. An attempt
identical to an earlier one is marked ``same_as`` and the LLM is told so.

Setting block (Part 10b, PICOS). In the T2 run the device part of the question
("on resource-constrained devices") sat inside the intervention, the blocks were
ECG AND machine learning AND accuracy, and the top 100 records by relevance were
generic heartbeat classifiers. The protocol now has ``pico.setting``, and the
LLM adds a setting block when it is specific. An evaluation topic can fix a
block's keywords (``protocol['search_terms']``, e.g. ``{"setting": [...]}``):
code builds that block, replaces the LLM's block of the same role in every
attempt (like the design block), and an empty list removes the role's block.

Narrowing that keeps recall (Part 4d). Told "too broad", gpt-4o-mini used to
*add* synonyms. A refinement now returns ``RefinedPlan``: the plan plus
``removed_terms`` / ``changed_terms`` with a reason each, and the prompt allows
only three narrowing moves (drop generic terms, turn a short ambiguous word into
a phrase, add a missing outcome block); dropping a specific synonym is not
allowed. Code records what really changed (``term_diff``) and stops the loop
early when the counts move by less than ``MIN_PROGRESS`` (5%) versus the
previous attempt in every database (``no_progress``): another call would not help.

Recall fixes (Part 10c, after the T1/T3 v1 runs: the search found 9/18 and 8/30 gold
studies while screening kept 13/14 of the gold it saw). Three causes, three fixes:
- *Nothing broadened.* 80 and 278 records were "in range" (``min_hits`` 20), so the loop
  stopped. Now, when all databases together return fewer than ``broaden_below`` hits
  (default 3 x the screening limit; the sum of counts is an upper bound of the unique
  records), the LLM is asked to broaden with three moves only: drop the outcome block,
  split a phrase into its key words, add common synonyms or product names. Code checks
  that the total rose by ``MIN_PROGRESS``; a broadening that did not is flagged
  ``broaden_failed``, stops the loop and never beats an earlier attempt.
- *Phrases instead of words.* T3's blocks were "classify ECG", "ECG classification",
  "ML models" where the review searched ECG, electrocardiogram, heartbeat. The prompt
  asks for single words or short established phrases; code flags keywords of more than
  3 words (``long_keywords``) and keywords sharing a word with another block
  (``mixed_keywords``), shown to the LLM at refinement.
- *Blocks that narrow without defining the topic.* T1 had a setting block repeating the
  intervention (smartwatch / wearable / device) and an outcome block of accuracy terms.
  Code merges an LLM setting block into the intervention block when half its keywords
  are already covered there (``setting_merged``), and drops an LLM outcome block made of
  measurement terms (``METRIC_TERMS``: they say how results are reported, not what the
  topic is; ``outcome_dropped``). Blocks fixed by the topic are never touched.
"""

from collections.abc import Mapping

from pydantic import BaseModel, Field

from ..llm import LLM, Usage
from ..sources import SOURCES, Concept, HttpClient, count_all
from ..sources.design_filters import design_block
from ..state import ReviewState, stage_round

OK, TOO_BROAD, TOO_NARROW, ERROR = "ok", "too broad", "too narrow", "error"
MIN_PROGRESS = 0.05  # a refinement must move some database's count by at least 5%
MAX_KEYWORD_WORDS = 3
SETTING_OVERLAP = 0.5   # share of setting keywords covered by the intervention -> merge
METRIC_SHARE = 2 / 3    # share of outcome keywords that are measurement terms -> drop

# How a result is measured, not what a study is about (Part 10c). Normalised forms
# (lower case, hyphens as spaces, plural "s" removed; see ``_norm``).
METRIC_TERMS = frozenset({
    "accuracy", "diagnostic accuracy", "classification accuracy", "detection accuracy",
    "sensitivity", "specificity", "precision", "recall", "f1", "f1 score", "f score",
    "auc", "auroc", "roc", "roc curve", "area under the curve",
    "predictive value", "positive predictive value", "negative predictive value", "ppv",
    "npv", "true positive", "false positive", "true negative", "false negative",
    "detection rate", "error rate", "misclassification", "diagnostic performance",
    "performance", "efficiency", "efficacy", "effectiveness", "validity", "reliability",
    "agreement", "kappa", "latency", "inference time", "execution time", "throughput",
    "energy", "energy consumption", "power consumption", "memory", "memory footprint",
    "model size",
})
# Generic task words (Part 10c, seen in the T3 smoke run of the fixes): OR-ed into a
# block, one of them voids the block, e.g. (ECG OR classification) matches every
# classification paper. Single words only; phrases ("arrhythmia detection") are kept.
TASK_TERMS = frozenset({
    "classify", "classification", "detect", "detection", "recognition", "identification",
    "prediction", "analysis", "algorithm", "model", "method", "approach", "system",
    "framework", "evaluation", "estimation", "monitoring",
})
_STOP = frozenset({"a", "an", "and", "the", "of", "in", "on", "for", "with", "to", "by",
                   "or", "at", "from", "based", "using"})


class SearchPlan(BaseModel):
    """What the LLM writes."""

    concepts: list[Concept] = Field(
        min_length=1, max_length=5,
        description="Concept blocks AND-ed together: population and intervention, plus a "
                    "setting and an outcome block when the rules ask for them. No design "
                    "block.")
    rationale: str = Field(description="Two or three sentences: which blocks and why.")


class RemovedTerm(BaseModel):
    term: str = Field(description="Keyword dropped from a block.")
    reason: str = Field(description="One line: why it pulled in unrelated records.")


class ChangedTerm(BaseModel):
    old: str = Field(description="Keyword as it was, e.g. 'aged'.")
    new: str = Field(description="What replaced it, e.g. 'aged 65'.")
    reason: str = Field(description="One line: why.")


class RefinedPlan(SearchPlan):
    """What the LLM writes when it revises a plan after seeing the hit counts."""

    removed_terms: list[RemovedTerm] = Field(
        default_factory=list, description="Every keyword dropped since the previous attempt.")
    changed_terms: list[ChangedTerm] = Field(
        default_factory=list, description="Every keyword rewritten since the previous attempt.")


# --- prompts ---------------------------------------------------------------------

def _protocol_lines(protocol: dict) -> str:
    pico = protocol["pico"]
    designs = protocol.get("study_designs", [])
    return (
        f"Question: {protocol['question']}\n"
        f"Population: {pico['population']}\n"
        f"Intervention/exposure: {pico['intervention']}\n"
        f"Comparator: {pico['comparator']}\n"
        + (f"Setting: {pico['setting']}\n" if pico.get("setting") else "")
        + f"Outcomes: {', '.join(pico['outcomes'])}\n"
        f"Study designs: {', '.join(designs) or 'any'}"
        + (" (a validated design filter for these is added by code)" if designs else "")
        + "\n"
        "Inclusion criteria: " + "; ".join(protocol.get("inclusion_criteria", [])) + "\n"
        f"Databases: {', '.join(protocol['databases'])}\n"
    )


def build_prompt(protocol: dict, max_records: int = 3000) -> str:
    return (
        "You are an information specialist designing the literature search of a PRISMA 2020 "
        "systematic review. Synonyms inside a concept block are joined with OR, blocks are "
        "joined with AND.\n"
        f"At most {max_records} records per database are downloaded and screened (most "
        "relevant first); records beyond that are never screened and are reported as not "
        "retrieved. Aim for a query that is broad enough to find every eligible study and "
        f"specific enough to stay on the topic and under {max_records} hits.\n"
        "Rules:\n"
        "- Always a population/problem block and an intervention/exposure block.\n"
        "- Add an outcome block when the primary outcome is specific and defines the topic "
        "(a disease, symptom or measurable condition, e.g. depression), or when population "
        "AND intervention would match far too many records. Skip it for generic outcomes "
        "(e.g. effectiveness, wellbeing) that abstracts often do not name, and for "
        "measurement terms (accuracy, sensitivity, specificity, precision, efficiency, "
        "latency): they say how results are reported, not what the topic is, and code "
        "drops such a block.\n"
        "- Add a setting block (role setting) when the protocol names a specific setting "
        "that eligible studies must have (a device or platform, e.g. microcontroller, "
        "wearable, smartphone, FPGA; or a care setting). Keep those words out of the "
        "intervention block, and do not add a setting block that repeats the intervention "
        "(code merges it into the intervention block).\n"
        "- Never write a study-design block (randomized, trial, cohort...): when the protocol "
        "restricts designs, a validated design filter is added by code.\n"
        "- One concept per block, and each term in exactly one block. The population block "
        "holds only who is studied (age group, setting), never the condition being measured "
        "as the outcome.\n"
        "- For each block give 3 to 10 keywords: the words authors write in titles and "
        "abstracts, 1 to 3 words each: single words or short established phrases (synonyms, "
        "abbreviations, common product or brand names, British/American spellings, singular "
        "forms). Never phrases describing a result ('reduction in ...', 'improvement in "
        "...'), and never a phrase that restates the question or joins two concepts (e.g. "
        "'exercise for depression'): each concept's words go in that concept's own block.\n"
        "- OR inside a block means 'any of these': never OR a generic task word "
        "(classification, detection, analysis, algorithm, model) with a topic's specific "
        "words: (ECG OR classification) matches every classification paper. Such single "
        "words are removed by code; established phrases (e.g. 'arrhythmia detection') are "
        "fine. Plain words only (no wildcards, no quotes, no AND/OR).\n"
        "- For each block give the matching MeSH headings (official spelling), or none if you "
        "are not sure. They are used for PubMed only.\n"
        "- Set each block's role (population, intervention, setting or outcome). Leave "
        "publication_types and title_terms empty.\n\n"
        + _protocol_lines(protocol)
    )


def _describe_attempt(a: dict) -> str:
    same = f" (identical to attempt {a['same_as']}: you did not change it)" \
        if a.get("same_as") else ""
    lines = [f"Attempt {a['attempt']}{same}:"]
    for c in a["concepts"]:
        extra = (f"; MeSH: {', '.join(c['mesh_terms'])}" if c.get("mesh_terms") else "") + \
                (f"; publication types: {', '.join(c['publication_types'])}"
                 if c.get("publication_types") else "")
        fixed = " (fixed, added by code)" if c.get("fixed") or c.get("role") == "design" \
            else ""
        lines.append(f"  [{c.get('role', 'other')}]{fixed} {c['name']}: "
                     f"{', '.join(c['keywords'])}{extra}")
    if a.get("generic_removed"):
        lines.append("  (generic words removed by code: " + ", ".join(a["generic_removed"]) + ")")
    for c in a.get("llm_blocks_replaced") or []:
        lines.append(f"  (your {c.get('role')} block was replaced or removed by code: the protocol fixes "
                     "it; do not change it)")
    if a.get("outcome_dropped"):
        lines.append("  (outcome block of measurement terms dropped by code: "
                     + ", ".join(a["outcome_dropped"]) + ")")
    if a.get("setting_merged"):
        lines.append("  (setting block merged into the intervention block by code, same words: "
                     + ", ".join(a["setting_merged"]["covered"]) + ")")
    for key, what in (("long_keywords", "more than 3 words"),
                      ("mixed_keywords", "sharing a word with another block")):
        if a.get(key):
            lines.append(f"  keywords {what}: {', '.join(a[key])}")
    for db, status in a["status"].items():
        hits = a["counts"].get(db)
        lines.append(f"  {db}: {hits if hits is not None else 'count failed'} hits -> {status}")
    if a.get("short"):
        lines.append(f"  all databases together: {a['total']} hits -> too few "
                     f"(at least {a['broaden_below']} wanted)")
    return "\n".join(lines)


def build_refine_prompt(protocol: dict, attempts: list[dict], *, max_records: int,
                        max_hits: int, min_hits: int, broaden_below: int = 0) -> str:
    broaden = (
        f"If all databases together have fewer than {broaden_below} hits (too few: eligible "
        "studies are probably missed), broaden. These are the ONLY allowed moves:\n"
        "  (a) remove the outcome block (unless it is fixed, added by code);\n"
        "  (b) replace a multi-word phrase with its key single word or a shorter established "
        "phrase (e.g. 'depressive symptom scores' -> 'depressive symptoms'; 'classify "
        "images' -> 'image', 'classification', each in the block of its concept);\n"
        "  (c) add common synonyms, abbreviations or product/brand names that authors write "
        "in titles and abstracts.\n"
        "  Not allowed: adding a block, removing or changing a block fixed by code.\n"
    ) if broaden_below else ""
    return (
        build_prompt(protocol, max_records)
        + "\nThe previous plan(s) were run and the databases returned these hit counts "
        f"(target: between {min_hits} and {max_hits} hits per database):\n"
        + "\n".join(_describe_attempt(a) for a in attempts)
        + "\n\nWrite a revised plan. It must differ from every previous attempt.\n"
        "If too broad, these are the ONLY allowed moves:\n"
        "  (a) remove ambiguous or generic keywords that match unrelated records (e.g. "
        "'training', 'fitness', 'sadness');\n"
        "  (b) turn a short ambiguous keyword into a phrase (e.g. 'aged' -> 'aged 65', "
        "'aged 60');\n"
        "  (c) add an outcome block if there is none.\n"
        "  Not allowed: adding keywords; removing a specific synonym that eligible studies "
        "use (e.g. 'elderly', 'depressive symptoms'). MeSH headings are not keywords: keep "
        "them (e.g. Aged is the correct heading for 65 and over).\n"
        "If too narrow: add synonyms, or drop the least essential block.\n"
        + broaden +
        "List every keyword you removed in removed_terms and every keyword you rewrote in "
        "changed_terms, with a one-line reason each.\n"
        "Blocks marked (fixed, added by code) are set by the protocol: removing or changing "
        "their terms has no effect.\n"
        "The blocks are shared by all databases. A specialised database may legitimately "
        "have few hits on this topic: do not broaden the whole search only for it when the "
        "other databases are in range.\n"
    )


# --- queries, counts, classification ------------------------------------------------

def build_queries(concepts: list[Concept], databases: list[str]) -> dict[str, str]:
    """Database syntax only; no network (``HttpClient`` without cache is never used here)."""
    http = HttpClient(cache_dir=None)
    return {db: SOURCES[db](http).build_query(concepts) for db in databases}


def overlap(concepts: list[dict]) -> list[str]:
    """Keywords that appear in more than one block (a sign of mixed-up concepts)."""
    seen: dict[str, int] = {}
    for i, c in enumerate(concepts):
        for k in {k.lower() for k in c["keywords"]}:
            seen[k] = -1 if k in seen and seen[k] != i else i
    return sorted(k for k, v in seen.items() if v == -1)


def _word(w: str) -> str:
    """Lower case, plural "s" removed (devices -> device, not class -> clas)."""
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(term: str) -> list[str]:
    return [_word(w) for w in term.replace("-", " ").replace("/", " ").split()]


def _norm(term: str) -> str:
    return " ".join(_words(term))


def _covers(a: str, b: str) -> bool:
    """Two keywords name the same thing: one's words are a subset of the other's
    ('device' / 'wrist-worn devices'), or they are the same once spaces are removed
    ('smart watch' / 'smartwatch')."""
    wa, wb = set(_words(a)), set(_words(b))
    return bool(wa and wb) and (wa <= wb or wb <= wa or
                                "".join(_words(a)) == "".join(_words(b)))


def is_metric_block(keywords: list[str]) -> bool:
    """Outcome block made of measurement terms (Part 10c, fix 3)."""
    return bool(keywords) and \
        sum(_norm(k) in METRIC_TERMS for k in keywords) / len(keywords) >= METRIC_SHARE


def merge_setting(concepts: list[Concept]) -> tuple[list[Concept], dict | None]:
    """Part 10c, fix 3: an LLM setting block that repeats the intervention (half of its
    keywords covered by an intervention keyword) is merged into the intervention block:
    the covered keywords are dropped as duplicates, the others are OR-ed with the
    intervention keywords. Returns the blocks and what was done (None: no merge)."""
    setting = next((c for c in concepts if c.role == "setting"), None)
    inter = next((c for c in concepts if c.role == "intervention"), None)
    if setting is None or inter is None:
        return concepts, None
    covered = [k for k in setting.keywords if any(_covers(k, i) for i in inter.keywords)]
    if len(covered) / len(setting.keywords) < SETTING_OVERLAP:
        return concepts, None
    added = [k for k in setting.keywords if k not in covered]
    merged = inter.model_copy(update={
        "keywords": inter.keywords + added,
        "mesh_terms": inter.mesh_terms + [m for m in setting.mesh_terms
                                          if m not in inter.mesh_terms]})
    out = [merged if c is inter else c for c in concepts if c is not setting]
    return out, {"covered": covered, "added": added}


def keyword_flags(concepts: list[dict]) -> dict[str, list[str]]:
    """Part 10c, fix 2 (audit and refine prompt; nothing is changed): keywords longer than
    ``MAX_KEYWORD_WORDS`` words, and keywords that share a content word with another
    block (a phrase joining two concepts, e.g. 'ECG classifiers' in the model block)."""
    blocks = [c for c in concepts if c.get("role") != "design"]
    llm_blocks = [c for c in blocks if not c.get("fixed")]   # the LLM cannot change the rest
    long = [k for c in llm_blocks for k in c["keywords"]
            if len(k.split()) > MAX_KEYWORD_WORDS]
    vocab = [{w for k in c["keywords"] for w in _words(k) if w not in _STOP} for c in blocks]
    mixed = []
    for i, c in enumerate(blocks):
        if c.get("fixed"):
            continue
        others = set().union(*(v for j, v in enumerate(vocab) if j != i))
        mixed += [k for k in c["keywords"]
                  if len(k.split()) > 1 and set(_words(k)) & others]
    return {"long_keywords": long, "mixed_keywords": sorted(set(mixed), key=str.lower)}


def term_diff(old: list[dict], new: list[dict]) -> dict[str, list[str]]:
    """Keywords really removed / added between two attempts (code, not the LLM's claim)."""
    words = lambda cs: {k.lower() for c in cs if c.get("role") != "design" for k in c["keywords"]}
    before, after = words(old), words(new)
    return {"removed": sorted(before - after), "added": sorted(after - before)}


def no_progress(previous: dict, current: dict, threshold: float = MIN_PROGRESS) -> bool:
    """True when no database's count moved by ``threshold`` or more (relative)."""
    shared = [db for db in current["counts"] if db in previous["counts"]]
    if not shared:
        return False
    def moved(db):
        old, new = previous["counts"][db], current["counts"][db]
        return abs(new - old) / old >= threshold if old else new != old
    return not any(moved(db) for db in shared)


def classify(counts: dict[str, int], errors: dict[str, str], *, max_hits: int,
             min_hits: int) -> dict[str, str]:
    status = {}
    for db, n in counts.items():
        status[db] = TOO_BROAD if n > max_hits else TOO_NARROW if n < min_hits else OK
    status.update({db: ERROR for db in errors})
    return status


def out_of_range(attempt: dict) -> int:
    return sum(s in (TOO_BROAD, TOO_NARROW) for s in attempt["status"].values())


def choose_attempt(attempts: list[dict]) -> dict:
    """Fewest databases out of range; ties go to the latest attempt, except an attempt
    flagged ``no_progress`` or ``broaden_failed``: it did not move the counts the way it
    was asked to, so it never beats an equal earlier one (it may still have dropped
    useful terms)."""
    return min(reversed(attempts), key=lambda a: (
        out_of_range(a), bool(a.get("no_progress") or a.get("broaden_failed"))))


def rose(previous: dict, current: dict, threshold: float = MIN_PROGRESS) -> bool:
    """Part 10c: a broadening attempt must raise the total hits by ``threshold``."""
    return current["total"] > previous["total"] * (1 + threshold)


def drop_generic(topic: list[Concept]) -> tuple[list[Concept], list[str]]:
    """Part 10c: remove single-word generic task terms (``TASK_TERMS``) from the LLM's
    blocks; a block left with no keyword is dropped. Returns the blocks and the removed
    keywords."""
    removed = [k for c in topic for k in c.keywords if _norm(k) in TASK_TERMS]
    out = []
    for c in topic:
        kept = [k for k in c.keywords if _norm(k) not in TASK_TERMS]
        if kept:
            out.append(c.model_copy(update={"keywords": kept}) if len(kept) < len(c.keywords)
                       else c)
    return out, removed


def code_blocks(topic: list[Concept]) -> tuple[list[Concept], list[str], dict | None,
                                               list[str]]:
    """Part 10c, on the LLM's own blocks (fixed roles are already removed): drop generic
    task words, drop an outcome block of measurement terms, merge a setting block that
    repeats the intervention. Returns the blocks, the dropped outcome keywords, the merge
    and the removed generic words."""
    topic, generic = drop_generic(topic)
    dropped = [k for c in topic if c.role == "outcome" and is_metric_block(c.keywords)
               for k in c.keywords]
    topic = [c for c in topic if not (c.role == "outcome" and is_metric_block(c.keywords))]
    topic, merged = merge_setting(topic)
    return topic, dropped, merged, generic


def sum_usage(usages: list[Usage]) -> dict:
    """One audit entry for the stage, whatever the number of LLM calls."""
    costs = [u.cost_usd for u in usages]
    return {"model": usages[-1].model, "llm_calls": len(usages),
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "cost_usd": None if None in costs else sum(c or 0.0 for c in costs),
            "cached": all(u.cached for u in usages)}


# --- the stage ---------------------------------------------------------------------

def topic_concepts(plan: SearchPlan, fixed_roles: frozenset[str] = frozenset()
                   ) -> tuple[list[Concept], int]:
    """The LLM's blocks without any design block (code owns designs), without the blocks
    whose role code fixes (``fixed_roles``, Part 10b) and without the code-only fields.
    Returns the blocks and how many design blocks were dropped."""
    kept = [c.model_copy(update={"publication_types": [], "title_terms": []})
            for c in plan.concepts if c.role != "design" and c.role not in fixed_roles]
    return kept, sum(c.role == "design" for c in plan.concepts)


def fixed_blocks(protocol: dict) -> tuple[frozenset[str], list[Concept]]:
    """Part 10b: (roles fixed by ``protocol['search_terms']``, the blocks code builds for
    them). A role with an empty list is fixed to "no block"."""
    terms = protocol.get("search_terms") or {}
    blocks = [Concept(name=f"{role}: fixed by the protocol", role=role, keywords=list(kw))
              for role, kw in terms.items() if kw]
    return frozenset(terms), blocks


def merge_blocks(topic: list[Concept], fixed: list[Concept]) -> list[Concept]:
    """LLM blocks and fixed blocks in PICOS order (population, intervention, setting,
    outcome, other); the design block is appended after by the caller."""
    order = {"population": 0, "intervention": 1, "setting": 2, "outcome": 3, "other": 4}
    return sorted(topic + fixed, key=lambda c: order.get(c.role, 4))


def make_search_strategy(llm: LLM, http: HttpClient | None = None, *, max_records: int = 3000,
                         min_hits: int = 20, max_refinements: int = 2, broaden_below: int = 0,
                         contact_email: str | None = None,
                         env: Mapping[str, str] | None = None):
    """Build the stage. Without ``http`` hit counts are not checked (one LLM call, no
    network): useful in tests of other stages; real runs always pass ``http``.

    ``max_records`` is both the download ceiling and the upper end of the "ok" range.
    ``broaden_below`` (Part 10c; 0 = off): total hits under which the search broadens."""
    max_hits = max_records

    def search_strategy(state: ReviewState) -> dict:
        protocol = state["protocol"]
        databases = protocol["databases"]
        filters = {"years": protocol.get("years", {}),
                   "languages": protocol.get("languages", [])}
        attempts: list[dict] = []
        usages: list[Usage] = []
        prompt = build_prompt(protocol, max_records)
        design, design_mapping = design_block(protocol.get("study_designs", []))
        fixed_roles, fixed = fixed_blocks(protocol)
        setting = (protocol.get("pico") or {}).get("setting") or ""

        for number in range(1, max_refinements + 2):
            schema = SearchPlan if number == 1 else RefinedPlan
            plan, usage = llm.call_with_usage(schema, prompt, agent="search_strategy")
            usages.append(usage)
            topic, dropped = topic_concepts(plan, fixed_roles)
            topic, outcome_dropped, merged, generic = code_blocks(topic)
            final = merge_blocks(topic, fixed) + ([design] if design else [])
            queries = build_queries(final, databases)
            concepts = [c.model_dump() | ({"fixed": True} if c in fixed else {})
                        for c in final]
            attempt = {"attempt": number, "concepts": concepts, "queries": queries,
                       "rationale": plan.rationale, "counts": {}, "errors": {}, "status": {},
                       "same_as": next((a["attempt"] for a in attempts
                                        if a["concepts"] == concepts), None),
                       "overlap": overlap(concepts), "llm_design_blocks_dropped": dropped,
                       "llm_blocks_replaced": [c.model_dump() for c in plan.concepts
                                               if c.role in fixed_roles],
                       "outcome_dropped": outcome_dropped, "setting_merged": merged,
                       "generic_removed": generic,
                       **keyword_flags(concepts),
                       # The protocol names a setting but no block searches it (a setting
                       # merged into the intervention block is searched there).
                       "setting_block_missing": bool(setting) and not merged and not any(
                           c.role == "setting" for c in final)}
            if isinstance(plan, RefinedPlan):
                attempt.update(
                    removed_terms=[t.model_dump() for t in plan.removed_terms],
                    changed_terms=[t.model_dump() for t in plan.changed_terms],
                    term_diff=term_diff(attempts[-1]["concepts"], concepts),
                    no_progress=False)
            attempts.append(attempt)
            if http is None:
                attempt["reason"] = "hit counts not checked (no HTTP client)"
                break
            counts, errors = count_all(queries, filters, http, contact_email=contact_email,
                                       env=env)
            attempt.update(counts=counts, errors=errors,
                           status=classify(counts, errors, max_hits=max_hits,
                                           min_hits=min_hits))
            bad = {db: s for db, s in attempt["status"].items() if s in (TOO_BROAD, TOO_NARROW)}
            # Part 10c: too few hits in total to fill the screening (an upper bound of the
            # unique records: duplicates are removed later, at retrieval).
            total = sum(counts.values())
            short = bool(broaden_below) and not errors and TOO_BROAD not in bad.values() \
                and total < broaden_below
            attempt.update(total=total, short=short, broaden_below=broaden_below)
            if not bad and not short:
                attempt["reason"] = "all databases within range"
                break
            summary = ", ".join([f"{db} {s} ({counts[db]} hits)" for db, s in bad.items()]
                                + ([f"{total} hits in all databases, below {broaden_below}: "
                                    "too few"] if short else []))
            previous = attempts[-2] if number > 1 else None
            if previous is not None and previous.get("short") and not rose(previous, attempt):
                attempt["broaden_failed"] = True
                attempt["reason"] = (f"{summary}; broadening did not raise the total by "
                                     f"{MIN_PROGRESS:.0%} ({previous['total']} -> {total}), "
                                     "refinement stopped")
                break
            if previous is not None and no_progress(previous, attempt):
                attempt["no_progress"] = True
                attempt["reason"] = (f"{summary}; no progress (counts changed by less than "
                                     f"{MIN_PROGRESS:.0%}), refinement stopped")
                if TOO_BROAD in bad.values():
                    attempt["reason"] += f" (download capped at {max_records} per database)"
                break
            if number > max_refinements:
                attempt["reason"] = f"{summary}; refinement limit reached"
                if TOO_BROAD in bad.values():
                    attempt["reason"] += f" (download capped at {max_records} per database)"
                break
            attempt["reason"] = f"{summary}; refinement requested"
            prompt = build_refine_prompt(protocol, attempts, max_records=max_records,
                                         max_hits=max_hits, min_hits=min_hits,
                                         broaden_below=broaden_below if short else 0)

        chosen = choose_attempt(attempts)
        hits = chosen["counts"]
        strategy = {
            "concepts": chosen["concepts"],
            "queries": chosen["queries"],
            # Filters come from the protocol scope, applied by each connector.
            "filters": filters,
            "rationale": chosen["rationale"],
            "counts": chosen["counts"],
            "chosen_attempt": chosen["attempt"],
            "targets": {"min_hits": min_hits, "max_records": max_records,
                        "broaden_below": broaden_below},
            "design_filter": design_mapping,
            # Expected from the counts; the retrieval stage records the actual numbers.
            "capped": {db: n > max_records for db, n in hits.items()},
            "not_retrieved": {db: max(0, n - max_records) for db, n in hits.items()},
            "attempts": attempts,
        }
        completed = state.get("completed_stages", [])
        return {
            "search_strategy": strategy,
            "completed_stages": ["search_strategy"],
            "audit_log": [{"node": "search_strategy", "step": state.get("steps_taken", 0),
                           "round": stage_round("search_strategy", completed),
                           "event": "search strategy built", "queries": chosen["queries"],
                           "attempts": len(attempts), "chosen_attempt": chosen["attempt"],
                           "counts_per_attempt": [a["counts"] for a in attempts],
                           "removed_terms": [t["term"] for a in attempts
                                             for t in a.get("removed_terms", [])],
                           "changed_terms": [f"{t['old']} -> {t['new']}" for a in attempts
                                             for t in a.get("changed_terms", [])],
                           "stopped_no_progress": any(a.get("no_progress") for a in attempts),
                           "broaden_below": broaden_below,
                           "broadening_requested": any(a.get("short") for a in attempts),
                           "broaden_failed": any(a.get("broaden_failed") for a in attempts),
                           "outcome_dropped": chosen["outcome_dropped"],
                           "generic_removed": chosen["generic_removed"],
                           "setting_merged": chosen["setting_merged"],
                           "long_keywords": chosen["long_keywords"],
                           "mixed_keywords": chosen["mixed_keywords"],
                           "design_filter": design_mapping,
                           "fixed_blocks": sorted(fixed_roles),
                           "setting_block_missing": chosen["setting_block_missing"],
                           "not_retrieved": strategy["not_retrieved"],
                           **sum_usage(usages)}],
        }

    return search_strategy
