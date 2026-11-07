"""PubMed connector via NCBI E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25501/).

Two steps:
1. ``esearch`` runs the query and returns the hit count and up to ``retmax`` PMIDs.
2. ``efetch`` downloads the records (XML) for those PMIDs, 200 per request.

Query syntax: each synonym is searched in title/abstract (``[tiab]``) and each
MeSH heading with ``[mh]`` (the heading *and* the narrower headings under it).
MeSH = Medical Subject Headings, the controlled vocabulary NCBI indexers assign
to every article: it finds papers that use other words for the same concept.
Indexing takes weeks to months, so recent papers only match on [tiab]: that is
why both are OR-ed together. A misspelled MeSH heading matches nothing but does
no harm, since the [tiab] synonyms still match. The study-design block (built
by code, see ``design_filters.py``) also uses publication types (``[pt]``, e.g.
"randomized controlled trial"[pt], the indexers' label for the design) and
title-only words (``[ti]``).

Rate limit: 3 requests/s without key, 10/s with NCBI_API_KEY.
Open access: a PMC id means the full text is free in PubMed Central.
"""

import xml.etree.ElementTree as ET
from datetime import date

from .base import (Concept, Record, SearchResult, Source, clean_text, join_blocks,
                   normalize_doi, unique_terms)
from .http import HttpClient

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HOST = "eutils.ncbi.nlm.nih.gov"
FETCH_BATCH = 200
TOOL = "sr-pipeline"


def _tag(term: str, tag: str) -> str:
    return f'"{term}"[{tag}]' if " " in term else f"{term}[{tag}]"


class PubMedSource(Source):
    name = "pubmed"

    def __init__(self, http: HttpClient, *, contact_email: str | None = None,
                 api_key: str | None = None):
        self.http, self.contact_email, self.api_key = http, contact_email, api_key
        http.set_min_interval(HOST, 0.11 if api_key else 0.34)

    def build_query(self, concepts: list[Concept]) -> str:
        blocks = []
        for c in concepts:
            terms = [_tag(t, "tiab") for t in unique_terms(c.keywords)]
            terms += [_tag(t, "mh") for t in unique_terms(c.mesh_terms)]
            terms += [_tag(t, "ti") for t in unique_terms(c.title_terms)]
            terms += [_tag(t, "pt") for t in unique_terms(c.publication_types)]
            blocks.append(" OR ".join(terms))
        return join_blocks(blocks)

    def _params(self, **extra) -> dict:
        return {"db": "pubmed", "tool": TOOL, "email": self.contact_email,
                "api_key": self.api_key, **extra}

    @staticmethod
    def _term(query, languages) -> str:
        if not languages:
            return query
        return f"({query}) AND (" + " OR ".join(f"{l.lower()}[la]" for l in languages) + ")"

    @staticmethod
    def _dates(years) -> dict:
        start, end = (years or {}).get("start"), (years or {}).get("end")
        if not (start or end):
            return {}
        # PubMed needs both bounds
        return {"datetype": "pdat", "mindate": str(start or 1800),
                "maxdate": str(end or date.today().year)}

    def count(self, query, *, years=None, languages=None) -> int:
        found = self.http.get_json(f"{EUTILS}/esearch.fcgi", self._params(
            term=self._term(query, languages), retmax=0, retmode="json",
            **self._dates(years)))["esearchresult"]
        return int(found.get("count", 0))

    def search(self, query, *, years=None, languages=None, max_records=500) -> SearchResult:
        term = self._term(query, languages)
        dates = self._dates(years)

        found = self.http.get_json(f"{EUTILS}/esearch.fcgi", self._params(
            term=term, retmax=max_records, retmode="json", sort="relevance",
            **dates))["esearchresult"]
        pmids = found.get("idlist", [])[:max_records]

        records: list[Record] = []
        for i in range(0, len(pmids), FETCH_BATCH):
            xml = self.http.get_text(f"{EUTILS}/efetch.fcgi", self._params(
                id=",".join(pmids[i:i + FETCH_BATCH]), retmode="xml"))
            records.extend(parse_efetch(xml))
        total = int(found.get("count", 0))
        # The full search string (with filters) is what PRISMA asks to report.
        return SearchResult(source=self.name, query=term, total_hits=total, records=records,
                            capped=total > max_records)


def _text(el: ET.Element | None) -> str | None:
    """Text of an element including inline markup (<i>, <sup>...)."""
    return clean_text("".join(el.itertext())) if el is not None else None


def _year(article: ET.Element) -> int | None:
    for path in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year",
                 "Journal/JournalIssue/PubDate/MedlineDate"):
        value = _text(article.find(path))
        if value and value[:4].isdigit():
            return int(value[:4])
    return None


def parse_efetch(xml: str) -> list[Record]:
    out = []
    for node in ET.fromstring(xml).findall("PubmedArticle"):
        citation = node.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        if citation is None or article is None:
            continue
        pmid = citation.findtext("PMID")
        title = _text(article.find("ArticleTitle"))
        if not pmid or not title:
            continue
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            text = _text(ab)
            if text:
                label = ab.get("Label")
                parts.append(f"{label}: {text}" if label else text)
        authors = []
        for a in article.findall("AuthorList/Author"):
            name = " ".join(x for x in (a.findtext("ForeName"), a.findtext("LastName")) if x)
            name = name or a.findtext("CollectiveName")
            if name:
                authors.append(clean_text(name))
        ids = {i.get("IdType"): (i.text or "").strip()
               for i in node.findall("PubmedData/ArticleIdList/ArticleId")}
        pmcid = ids.get("pmc") or None
        out.append(Record(
            id=f"pubmed:{pmid}", source="pubmed", source_id=pmid, pmid=pmid, pmcid=pmcid,
            doi=normalize_doi(ids.get("doi")), title=title,
            abstract=" ".join(parts) or None, authors=authors, year=_year(article),
            venue=_text(article.find("Journal/Title")),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            oa_pdf_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/" if pmcid else None,
        ))
    return out
