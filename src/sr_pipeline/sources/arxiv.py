"""arXiv connector (https://info.arxiv.org/help/api/user-manual.html): preprints in
physics, maths, computer science, quantitative biology, statistics, economics.

Query syntax: field prefixes (``ti:`` title, ``abs:`` abstract), AND / OR /
ANDNOT, parentheses, "quoted phrases". The year range is a ``submittedDate``
clause. There is no language filter (almost everything is in English).

Rate limit: arXiv asks for one request every 3 seconds. Every arXiv paper is
open access, so ``oa_pdf_url`` is always set. Preprints are not peer reviewed:
screening must keep that in mind.
"""

import re
import xml.etree.ElementTree as ET
from datetime import date

from .base import (Concept, Record, SearchResult, Source, clean_text, join_blocks,
                   normalize_doi, phrase, unique_terms)
from .http import HttpClient

API_URL = "https://export.arxiv.org/api/query"
HOST = "export.arxiv.org"
NS = {"atom": "http://www.w3.org/2005/Atom",
      "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
      "arxiv": "http://arxiv.org/schemas/atom"}


def date_clause(years: dict | None) -> str | None:
    start, end = (years or {}).get("start"), (years or {}).get("end")
    if not start and not end:
        return None
    lo = f"{start or 1991}01010000"
    hi = f"{end or date.today().year}12312359"
    return f"submittedDate:[{lo} TO {hi}]"


class ArxivSource(Source):
    name = "arxiv"

    def __init__(self, http: HttpClient, *, page_size: int = 200, **_):
        self.http, self.page_size = http, page_size
        http.set_min_interval(HOST, 3.0)

    def build_query(self, concepts: list[Concept]) -> str:
        blocks = []
        for c in concepts:
            terms = [phrase(t) for t in unique_terms(c.keywords)]
            blocks.append(" OR ".join(f"{f}:{t}" for t in terms for f in ("ti", "abs")))
        return join_blocks(blocks)

    def search(self, query, *, years=None, languages=None, max_records=500) -> SearchResult:
        full = query
        if (dc := date_clause(years)):
            full = f"({query}) AND {dc}"
        records: list[Record] = []
        total = 0
        while len(records) < max_records:
            xml = self.http.get_text(API_URL, {
                "search_query": full, "start": len(records),
                "max_results": min(self.page_size, max_records - len(records)),
                "sortBy": "relevance", "sortOrder": "descending",
            })
            total, page = parse_feed(xml)
            if not page:  # end of results (or a transient empty page: stop, never loop)
                break
            records.extend(page)
            if len(records) >= total:
                break
        records = records[:max_records]
        return SearchResult(source=self.name, query=full, total_hits=total, records=records,
                            capped=total > len(records))


def parse_feed(xml: str) -> tuple[int, list[Record]]:
    root = ET.fromstring(xml)
    total = int(root.findtext("opensearch:totalResults", "0", NS))
    out = []
    for e in root.findall("atom:entry", NS):
        abs_url = e.findtext("atom:id", "", NS).strip()
        # http://arxiv.org/abs/2101.00001v2 -> 2101.00001 (version dropped)
        source_id = re.sub(r"v\d+$", "", abs_url.split("/abs/")[-1])
        title = clean_text(e.findtext("atom:title", None, NS))
        if not source_id or not title:
            continue
        pdf = next((l.get("href") for l in e.findall("atom:link", NS)
                    if l.get("title") == "pdf"), None)
        published = e.findtext("atom:published", "", NS)
        out.append(Record(
            id=f"arxiv:{source_id}", source="arxiv", source_id=source_id, title=title,
            abstract=clean_text(e.findtext("atom:summary", None, NS)),
            authors=[n for a in e.findall("atom:author", NS)
                     if (n := clean_text(a.findtext("atom:name", "", NS)))],
            year=int(published[:4]) if published[:4].isdigit() else None,
            venue=clean_text(e.findtext("arxiv:journal_ref", None, NS)) or "arXiv",
            doi=normalize_doi(e.findtext("arxiv:doi", None, NS)),
            url=f"https://arxiv.org/abs/{source_id}",
            oa_pdf_url=pdf or f"https://arxiv.org/pdf/{source_id}",
        ))
    return total, out
