"""Database connectors. All return ``Record`` objects (see base.py).

``search_all`` runs the search strategy on every database of the protocol; the
retrieval stage (Part 5) uses it. One database failing does not stop the
others: its error is returned so it can be logged and reported.
"""

import os
from collections.abc import Mapping

from .arxiv import ArxivSource
from .base import Concept, Record, SearchResult, Source
from .http import HttpClient, HttpError
from .openalex import OpenAlexSource
from .pubmed import PubMedSource

SOURCES: dict[str, type[Source]] = {
    "openalex": OpenAlexSource,
    "pubmed": PubMedSource,
    "arxiv": ArxivSource,
}
API_KEY_ENV = {"openalex": "OPENALEX_API_KEY", "pubmed": "NCBI_API_KEY"}


def make_source(name: str, http: HttpClient, *, contact_email: str | None = None,
                env: Mapping[str, str] | None = None) -> Source:
    env = os.environ if env is None else env
    cls = SOURCES[name]
    if name == "arxiv":
        return cls(http)
    return cls(http, contact_email=contact_email, api_key=env.get(API_KEY_ENV[name]) or None)


def search_all(strategy: dict, http: HttpClient, *, max_records: int,
               contact_email: str | None = None, env: Mapping[str, str] | None = None
               ) -> tuple[dict[str, SearchResult], dict[str, str]]:
    """Run ``strategy['queries']`` (one query per database) with its filters.

    Returns ``(results, errors)``, both keyed by database name.
    """
    filters = strategy.get("filters", {})
    results, errors = {}, {}
    for name, query in strategy["queries"].items():
        source = make_source(name, http, contact_email=contact_email, env=env)
        try:
            results[name] = source.search(query, years=filters.get("years"),
                                          languages=filters.get("languages"),
                                          max_records=max_records)
        except (HttpError, ValueError, KeyError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return results, errors


__all__ = ["SOURCES", "Concept", "HttpClient", "HttpError", "Record", "SearchResult",
           "Source", "make_source", "search_all"]
