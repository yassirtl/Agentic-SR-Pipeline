"""OpenAlex connector (https://docs.openalex.org): all disciplines, ~250M works.

Query: the ``title_and_abstract.search`` filter, which accepts Boolean AND / OR /
NOT, parentheses and "quoted phrases" (the plain ``search=`` parameter also
matches full texts and returns far too many hits for a review). Words are
stemmed; there are no wildcards.

Paging uses page numbers (cursor paging now requires an API key).

Without an API key every request costs $0.001 out of a free $0.10/day budget
shared by everyone on the same IP address; a free key (OPENALEX_API_KEY) gives
$1/day of its own. A 500-record search costs 3 requests.

Abstracts come as an *inverted index* (word -> positions) for copyright reasons
and are rebuilt here. Some publishers remove them: ``abstract`` is then None.
"""

from .base import (LANGUAGE_CODES, Concept, Record, SearchResult, Source, clean_text,
                   join_blocks, normalize_doi, phrase, unique_terms)
from .http import HttpClient

BASE_URL = "https://api.openalex.org/works"
HOST = "api.openalex.org"
FIELDS = ("id,doi,display_name,publication_year,authorships,primary_location,"
          "best_oa_location,ids,abstract_inverted_index")


def rebuild_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return clean_text(" ".join(word for _, word in words))


def year_filter(years: dict | None) -> str | None:
    start, end = (years or {}).get("start"), (years or {}).get("end")
    if start and end:
        return f"publication_year:{start}-{end}"
    if start:
        return f"publication_year:>{start - 1}"
    if end:
        return f"publication_year:<{end + 1}"
    return None


def _strip(url: str | None, prefix: str) -> str | None:
    return url[len(prefix):] if url and url.startswith(prefix) else url


class OpenAlexSource(Source):
    name = "openalex"

    def __init__(self, http: HttpClient, *, contact_email: str | None = None,
                 api_key: str | None = None, per_page: int = 200):
        self.http, self.contact_email, self.api_key = http, contact_email, api_key
        self.per_page = per_page
        http.set_min_interval(HOST, 0.1)  # OpenAlex allows 10 requests/s

    def build_query(self, concepts: list[Concept]) -> str:
        return join_blocks([" OR ".join(phrase(t) for t in unique_terms(c.keywords))
                            for c in concepts])

    def search(self, query, *, years=None, languages=None, max_records=500) -> SearchResult:
        filters = [f"title_and_abstract.search:{query}"]
        if (yf := year_filter(years)):
            filters.append(yf)
        codes = [LANGUAGE_CODES[l.lower()] for l in languages or [] if l.lower() in LANGUAGE_CODES]
        if codes:
            filters.append("language:" + "|".join(codes))

        # Page numbers, not cursors: cursor paging needs an API key, page paging
        # works without one (up to 10,000 results, far above max_records).
        # per-page stays constant so page N always means the same slice.
        per_page = min(self.per_page, max_records)
        records: list[Record] = []
        page, total = 1, 0
        while len(records) < max_records:
            data = self.http.get_json(BASE_URL, {
                "filter": ",".join(filters), "select": FIELDS,
                "per-page": per_page, "page": page,
                "mailto": self.contact_email, "api_key": self.api_key,
            })
            total = data["meta"]["count"]
            records.extend(r for r in map(parse_work, data["results"]) if r is not None)
            if len(data["results"]) < per_page or page * per_page >= total:
                break
            page += 1
        records = records[:max_records]
        return SearchResult(source=self.name, query=query, total_hits=total, records=records,
                            capped=total > len(records))


def parse_work(work: dict) -> Record | None:
    title = clean_text(work.get("display_name"))
    if not title:
        return None
    source_id = _strip(work["id"], "https://openalex.org/")
    primary = work.get("primary_location") or {}
    best_oa = work.get("best_oa_location") or {}
    ids = work.get("ids") or {}
    doi = normalize_doi(work.get("doi"))
    return Record(
        id=f"openalex:{source_id}", source="openalex", source_id=source_id, doi=doi,
        title=title, abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        authors=[a["author"]["display_name"] for a in work.get("authorships") or []
                 if (a.get("author") or {}).get("display_name")],
        year=work.get("publication_year"),
        venue=((primary.get("source") or {}).get("display_name")),
        url=primary.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else None),
        oa_pdf_url=best_oa.get("pdf_url"),
        pmid=_strip(ids.get("pmid"), "https://pubmed.ncbi.nlm.nih.gov/"),
        pmcid=_strip(ids.get("pmcid"), "https://www.ncbi.nlm.nih.gov/pmc/articles/"),
    )
