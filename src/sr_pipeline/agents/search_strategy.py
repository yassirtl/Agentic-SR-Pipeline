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

Narrowing that keeps recall (Part 4d). Told "too broad", gpt-4o-mini used to
*add* synonyms. A refinement now returns ``RefinedPlan``: the plan plus
``removed_terms`` / ``changed_terms`` with a reason each, and the prompt allows
only three narrowing moves (drop generic terms, turn a short ambiguous word into
a phrase, add a missing outcome block); dropping a specific synonym is not
allowed. Code records what really changed (``term_diff``) and stops the loop
early when the counts move by less than ``MIN_PROGRESS`` (5%) versus the
previous attempt in every database (``no_progress``): another call would not help.
"""

from collections.abc import Mapping

from pydantic import BaseModel, Field

from ..llm import LLM, Usage
from ..sources import SOURCES, Concept, HttpClient, count_all
from ..sources.design_filters import design_block
from ..state import ReviewState, stage_round

OK, TOO_BROAD, TOO_NARROW, ERROR = "ok", "too broad", "too narrow", "error"
MIN_PROGRESS = 0.05  # a refinement must move some database's count by at least 5%


class SearchPlan(BaseModel):
    """What the LLM writes."""

    concepts: list[Concept] = Field(
        min_length=1, max_length=4,
        description="Concept blocks AND-ed together: population and intervention, plus an "
                    "outcome block when the rules ask for it. No design block.")
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
        f"Outcomes: {', '.join(pico['outcomes'])}\n"
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
        "(e.g. effectiveness, wellbeing) that abstracts often do not name.\n"
        "- Never write a study-design block (randomized, trial, cohort...): when the protocol "
        "restricts designs, a validated design filter is added by code.\n"
        "- One concept per block, and each term in exactly one block. The population block "
        "holds only who is studied (age group, setting), never the condition being measured "
        "as the outcome.\n"
        "- For each block give 3 to 10 keywords: the words authors write in titles and "
        "abstracts, 1 to 3 words each (synonyms, abbreviations, British/American spellings, "
        "singular forms). Never phrases describing a result ('reduction in ...', "
        "'improvement in ...'). Plain words only (no wildcards, no quotes, no AND/OR).\n"
        "- For each block give the matching MeSH headings (official spelling), or none if you "
        "are not sure. They are used for PubMed only.\n"
        "- Set each block's role (population, intervention or outcome). Leave "
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
        fixed = " (fixed, added by code)" if c.get("role") == "design" else ""
        lines.append(f"  [{c.get('role', 'other')}]{fixed} {c['name']}: "
                     f"{', '.join(c['keywords'])}{extra}")
    for db, status in a["status"].items():
        hits = a["counts"].get(db)
        lines.append(f"  {db}: {hits if hits is not None else 'count failed'} hits -> {status}")
    return "\n".join(lines)


def build_refine_prompt(protocol: dict, attempts: list[dict], *, max_records: int,
                        max_hits: int, min_hits: int) -> str:
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
        "List every keyword you removed in removed_terms and every keyword you rewrote in "
        "changed_terms, with a one-line reason each.\n"
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
    flagged ``no_progress``: it narrowed nothing, so it never beats an equal earlier one
    (it may still have dropped useful terms)."""
    return min(reversed(attempts), key=lambda a: (out_of_range(a), bool(a.get("no_progress"))))


def sum_usage(usages: list[Usage]) -> dict:
    """One audit entry for the stage, whatever the number of LLM calls."""
    costs = [u.cost_usd for u in usages]
    return {"model": usages[-1].model, "llm_calls": len(usages),
            "input_tokens": sum(u.input_tokens for u in usages),
            "output_tokens": sum(u.output_tokens for u in usages),
            "cost_usd": None if None in costs else sum(c or 0.0 for c in costs),
            "cached": all(u.cached for u in usages)}


# --- the stage ---------------------------------------------------------------------

def topic_concepts(plan: SearchPlan) -> tuple[list[Concept], int]:
    """The LLM's blocks without any design block (code owns designs) and without the
    code-only fields. Returns the blocks and how many design blocks were dropped."""
    kept = [c.model_copy(update={"publication_types": [], "title_terms": []})
            for c in plan.concepts if c.role != "design"]
    return kept, len(plan.concepts) - len(kept)


def make_search_strategy(llm: LLM, http: HttpClient | None = None, *, max_records: int = 3000,
                         min_hits: int = 20, max_refinements: int = 2,
                         contact_email: str | None = None,
                         env: Mapping[str, str] | None = None):
    """Build the stage. Without ``http`` hit counts are not checked (one LLM call, no
    network): useful in tests of other stages; real runs always pass ``http``.

    ``max_records`` is both the download ceiling and the upper end of the "ok" range."""
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

        for number in range(1, max_refinements + 2):
            schema = SearchPlan if number == 1 else RefinedPlan
            plan, usage = llm.call_with_usage(schema, prompt, agent="search_strategy")
            usages.append(usage)
            topic, dropped = topic_concepts(plan)
            final = topic + ([design] if design else [])
            queries = build_queries(final, databases)
            concepts = [c.model_dump() for c in final]
            attempt = {"attempt": number, "concepts": concepts, "queries": queries,
                       "rationale": plan.rationale, "counts": {}, "errors": {}, "status": {},
                       "same_as": next((a["attempt"] for a in attempts
                                        if a["concepts"] == concepts), None),
                       "overlap": overlap(concepts), "llm_design_blocks_dropped": dropped}
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
            if not bad:
                attempt["reason"] = "all databases within range"
                break
            summary = ", ".join(f"{db} {s} ({counts[db]} hits)" for db, s in bad.items())
            if number > 1 and no_progress(attempts[-2], attempt):
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
                                         max_hits=max_hits, min_hits=min_hits)

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
            "targets": {"min_hits": min_hits, "max_records": max_records},
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
                           "design_filter": design_mapping,
                           "not_retrieved": strategy["not_retrieved"],
                           **sum_usage(usages)}],
        }

    return search_strategy
