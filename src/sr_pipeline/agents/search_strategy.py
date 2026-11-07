"""Search strategy: protocol -> one Boolean query per database (PRISMA item 7).

Division of labour:
- the LLM does the part that needs domain knowledge: split the question into
  2-3 *concepts* and list, for each, synonyms and spelling variants (plus MeSH
  headings for PubMed);
- code turns those concepts into each database's syntax (``Source.build_query``):
  synonyms OR-ed inside a concept, concepts AND-ed together.

So the queries are always syntactically valid, the Boolean logic is identical
in every database, and the exact strings are stored in the state for the report.

Why only 2-3 concepts: every extra AND removes papers. Outcomes and comparators
are often not mentioned in titles/abstracts, so searching on them loses relevant
studies; screening (not the search) is where eligibility is checked. A search
for a systematic review aims at recall (find everything), not precision.
"""

from pydantic import BaseModel, Field

from ..llm import LLM
from ..sources import SOURCES, Concept, HttpClient
from ..state import ReviewState, stage_round


class SearchPlan(BaseModel):
    """What the LLM writes."""

    concepts: list[Concept] = Field(
        min_length=1, max_length=4,
        description="Concept blocks AND-ed together; usually 2 (population, intervention), "
                    "at most 3.")
    rationale: str = Field(description="Two or three sentences: which concepts and why.")


def build_prompt(protocol: dict) -> str:
    pico = protocol["pico"]
    return (
        "You are an information specialist designing the literature search of a PRISMA 2020 "
        "systematic review. The goal is high recall: find every eligible study; screening "
        "removes irrelevant ones later.\n"
        "Split the question into concept blocks. Synonyms inside a block are joined with OR, "
        "blocks are joined with AND.\n"
        "Rules:\n"
        "- Usually 2 blocks (population/problem and intervention/exposure); add a third only "
        "if the result would otherwise be far too broad. Outcomes and comparators are rarely "
        "searched: they are often missing from abstracts.\n"
        "- For each block give 3 to 10 keywords: synonyms, abbreviations, British/American "
        "spellings, singular forms. Plain words or phrases only (no wildcards, no quotes, no "
        "AND/OR).\n"
        "- For each block give the matching MeSH headings (official spelling), or none if you "
        "are not sure. They are used for PubMed only.\n\n"
        f"Question: {protocol['question']}\n"
        f"Population: {pico['population']}\n"
        f"Intervention/exposure: {pico['intervention']}\n"
        f"Comparator: {pico['comparator']}\n"
        f"Outcomes: {', '.join(pico['outcomes'])}\n"
        f"Study designs: {', '.join(protocol.get('study_designs', [])) or 'any'}\n"
        "Inclusion criteria: " + "; ".join(protocol.get("inclusion_criteria", [])) + "\n"
    )


def build_queries(concepts: list[Concept], databases: list[str]) -> dict[str, str]:
    """Database syntax only; no network (``HttpClient`` without cache is never used here)."""
    http = HttpClient(cache_dir=None)
    return {db: SOURCES[db](http).build_query(concepts) for db in databases}


def make_search_strategy(llm: LLM):
    def search_strategy(state: ReviewState) -> dict:
        protocol = state["protocol"]
        plan, usage = llm.call_with_usage(SearchPlan, build_prompt(protocol),
                                          agent="search_strategy")
        queries = build_queries(plan.concepts, protocol["databases"])
        strategy = {
            "concepts": [c.model_dump() for c in plan.concepts],
            "queries": queries,
            # Filters come from the protocol scope, applied by each connector.
            "filters": {"years": protocol.get("years", {}),
                        "languages": protocol.get("languages", [])},
            "rationale": plan.rationale,
        }
        completed = state.get("completed_stages", [])
        return {
            "search_strategy": strategy,
            "completed_stages": ["search_strategy"],
            "audit_log": [{"node": "search_strategy", "step": state.get("steps_taken", 0),
                           "round": stage_round("search_strategy", completed),
                           "event": "search strategy built", "queries": queries,
                           **usage.as_dict()}],
        }

    return search_strategy
