"""Semantic Scholar connector (https://api.semanticscholar.org/api-docs/graph, "paper bulk
search"): about 200 million papers across all fields, with abstracts, DOIs, PubMed / PMC /
arXiv ids and open-access PDF links.

Query syntax of the bulk search: ``+`` AND, ``|`` OR, ``-`` NOT, "quoted phrases", parentheses;
searched in title and abstract. The year range is the ``year`` parameter ("2021-",
"2015-2025"). There is no language filter.

The bulk search returns up to 1000 papers per request and pages with a continuation
``token``; it cannot sort by relevance, so results are sorted by citation count (the most
used papers first), the closest stand-in for a relevance order when the screening limit
keeps the best-ranked records. Without an API key the service is shared and rate-limited:
one request per second at most, and the HTTP client retries on 429.
"""

from .base import (Concept, Record, SearchResult, Source, clean_text, normalize_doi, phrase,
                   unique_terms)
from .http import HttpClient

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
HOST = "api.semanticscholar.org"
FIELDS = "title,abstract,year,venue,authors,externalIds,url,openAccessPdf"


def year_param(years: dict | None) -> str | None:
    start, end = (years or {}).get("start"), (years or {}).get("end")
    end_date = (years or {}).get("end_date")
    end = end or (int(str(end_date)[:4]) if end_date else None)
    if not start and not end:
        return None
    return f"{start or ''}-{end or ''}"


class SemanticScholarSource(Source):
    name = "semanticscholar"

    def __init__(self, http: HttpClient, **_):
        self.http = http
        http.set_min_interval(HOST, 1.1)

    def build_query(self, concepts: list[Concept]) -> str:
        blocks = []
        for c in concepts:
            terms = [phrase(t) for t in unique_terms(c.keywords + c.title_terms)]
            if terms:
                blocks.append("(" + " | ".join(terms) + ")")
        return " + ".join(blocks)

    def _params(self, query, years, fields, token=None) -> dict:
        return {"query": query, "fields": fields, "year": year_param(years),
                "sort": "citationCount:desc", "token": token}

    def count(self, query, *, years=None, languages=None) -> int:
        data = self.http.get_json(API_URL, self._params(query, years, "paperId"))
        return int(data.get("total") or 0)

    def search(self, query, *, years=None, languages=None, max_records=500) -> SearchResult:
        records: list[Record] = []
        total, token = 0, None
        while len(records) < max_records:
            data = self.http.get_json(API_URL, self._params(query, years, FIELDS, token))
            total = int(data.get("total") or 0)
            page = [r for r in (parse_paper(p) for p in data.get("data") or []) if r]
            if not page:
                break
            records.extend(page)
            token = data.get("token")
            if not token:
                break
        records = records[:max_records]
        return SearchResult(source=self.name, query=query, total_hits=total, records=records,
                            capped=total > max_records)


def parse_paper(p: dict) -> Record | None:
    pid, title = p.get("paperId"), clean_text(p.get("title"))
    if not pid or not title:
        return None
    ext = p.get("externalIds") or {}
    pmcid = ext.get("PubMedCentral")
    return Record(
        id=f"semanticscholar:{pid}", source="semanticscholar", source_id=pid, title=title,
        abstract=clean_text(p.get("abstract")),
        authors=[a["name"] for a in p.get("authors") or [] if a.get("name")],
        year=p.get("year"), venue=clean_text(p.get("venue")) or None,
        doi=normalize_doi(ext.get("DOI")), pmid=str(ext["PubMed"]) if ext.get("PubMed") else None,
        pmcid=(pmcid if str(pmcid).upper().startswith("PMC") else f"PMC{pmcid}") if pmcid
        else None,
        url=p.get("url") or f"https://www.semanticscholar.org/paper/{pid}",
        oa_pdf_url=((p.get("openAccessPdf") or {}).get("url") or None),
    )
