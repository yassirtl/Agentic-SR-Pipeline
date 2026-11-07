"""Retrieval stage: run the search strategy on every database, deduplicate, count for PRISMA.

No LLM here: the queries were written (and their hit counts checked) by the
search strategy stage. This stage downloads the records, merges duplicates
(``dedup.py``) and fills the *Identification* part of the PRISMA 2020 flow
diagram:

    identified (hits per database)
      - not retrieved (hits above the max_records download ceiling)
      = downloaded
      - duplicates removed (by rule: same id, DOI, PMID, PMC id, title)
      = records screened (unique papers, the input of title/abstract screening)

The ceiling loss is stored under ``prisma_counts["identification"]``. PRISMA's
own "reports not retrieved" (a full text that cannot be obtained) is a different
number, counted at full-text screening (Part 6), and must not be mixed with it.

Full texts are *not* fetched here. PRISMA seeks full texts only for the papers
that pass title/abstract screening; fetching ~2,800 of them now would cost hours
of requests for papers that are mostly excluded on their abstract. The full-text
stage uses ``sources/fulltext.py`` and ``rag/`` on the screened-in papers.

Loop-backs: ``records`` is append-only, so a second run (after the coordinator
sent the review back to ``search_strategy``) keeps the old records for the audit
trail. Every record carries ``round``; deduplication and counts use the current
round only, and ``unique_records`` is replaced.
"""

from collections.abc import Callable, Mapping

from ..dedup import deduplicate
from ..sources import HttpClient, search_all
from ..state import ReviewState, stage_round


def make_retrieval(http: HttpClient | None, *, max_records: int = 3000,
                   contact_email: str | None = None, env: Mapping[str, str] | None = None,
                   search: Callable = search_all):
    """Build the stage. ``search`` is injectable for tests (same signature as ``search_all``).

    The download ceiling is the one the search strategy was checked against
    (``search_strategy['targets']['max_records']``); ``max_records`` is only the
    fallback for a strategy without targets.
    """

    def retrieval(state: ReviewState) -> dict:
        strategy = state["search_strategy"]
        completed = state.get("completed_stages", [])
        rnd = stage_round("retrieval", completed)
        cap = strategy.get("targets", {}).get("max_records", max_records)

        results, errors = search(strategy, http, max_records=cap,
                                 contact_email=contact_email, env=env)
        records = [{**rec.model_dump(mode="json"), "round": rnd}
                   for res in results.values() for rec in res.records]
        dd = deduplicate(records)
        papers = [{**p, "round": rnd} for p in dd["papers"]]

        databases = {db: {"query": res.query, "identified": res.total_hits,
                          "downloaded": len(res.records), "not_retrieved": res.not_retrieved,
                          "capped": res.capped, "searched_at": res.searched_at}
                     for db, res in results.items()}
        identification = {
            "databases": databases,
            "errors": errors,
            "identified": sum(d["identified"] for d in databases.values()),
            "not_retrieved": sum(d["not_retrieved"] for d in databases.values()),
            "downloaded": len(records),
            "duplicates_removed": dd["counts"]["duplicates_removed"],
            "duplicates_by_rule": dd["counts"]["by_rule"],
            "records_screened": len(papers),
            "max_records": cap,
            "round": rnd,
        }
        prisma = {**state.get("prisma_counts", {}), "identification": identification}

        audit = {"node": "retrieval", "step": state.get("steps_taken", 0), "round": rnd,
                 "event": "records retrieved and deduplicated",
                 "http_requests": getattr(http, "network_calls", None),
                 **{k: v for k, v in identification.items() if k not in ("round",)},
                 "title_conflicts_kept_apart": len(dd["conflicts"])}
        return {
            "records": records,
            "unique_records": papers,
            # Merge log for the report and audits (who was merged with whom, and why).
            "retrieval": {"round": rnd, "duplicates": dd["duplicates"],
                          "conflicts": dd["conflicts"]},
            "prisma_counts": prisma,
            "completed_stages": ["retrieval"],
            "audit_log": [audit],
        }

    return retrieval


def describe_search(state: ReviewState) -> str | None:
    """One line for the Coordinator: per-database hits / downloaded / capped / errors."""
    ident = state.get("prisma_counts", {}).get("identification")
    if not ident:
        return None
    parts = [f"{db} {d['identified']} hits, {d['downloaded']} downloaded"
             + (f" (capped, {d['not_retrieved']} not retrieved)" if d["capped"] else "")
             for db, d in ident["databases"].items()]
    parts += [f"{db} FAILED ({err})" for db, err in ident["errors"].items()]
    line = (f"Retrieval (round {ident['round']}): " + "; ".join(parts)
            + f"; {ident['duplicates_removed']} duplicates removed, "
              f"{ident['records_screened']} unique records to screen")
    targets = state.get("search_strategy", {}).get("targets")
    if targets:
        line += (f". Search target per database: {targets['min_hits']}-"
                 f"{targets['max_records']} hits")
    return line
