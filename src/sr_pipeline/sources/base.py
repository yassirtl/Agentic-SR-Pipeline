"""Common record model shared by every database connector.

Each database answers in its own format (OpenAlex JSON, PubMed XML, arXiv Atom).
Connectors translate those answers into ``Record`` so the rest of the pipeline
(deduplication, screening, extraction) never sees a database-specific format.
"""

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# Language names used in the protocol -> ISO 639-1 codes (OpenAlex) and PubMed tags.
LANGUAGE_CODES = {
    "english": "en", "french": "fr", "german": "de", "spanish": "es", "italian": "it",
    "portuguese": "pt", "chinese": "zh", "japanese": "ja", "arabic": "ar", "russian": "ru",
    "dutch": "nl",
}


class Record(BaseModel):
    """One bibliographic record as returned by one database.

    ``id`` is unique across databases (``"<source>:<source_id>"``). The same paper
    found in two databases gives two records; deduplication (Part 5) merges them.
    """

    id: str
    source: str
    source_id: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    oa_pdf_url: str | None = Field(None, description="Open-access PDF, if the source knows one.")
    pmid: str | None = None
    pmcid: str | None = None


class SearchResult(BaseModel):
    """Answer of one database to one query (PRISMA item 7 + 'records identified')."""

    source: str
    query: str
    total_hits: int = Field(description="Hits reported by the database (may exceed len(records)).")
    records: list[Record]
    searched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))
    capped: bool = Field(False, description="True if max_records stopped the download early.")


def normalize_doi(doi: str | None) -> str | None:
    """'https://doi.org/10.1/ABC' -> '10.1/abc' (DOIs are case-insensitive)."""
    if not doi:
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I)
    return doi.lower() or None


def clean_text(text: str | None) -> str | None:
    """Collapse whitespace (XML sources keep line breaks inside titles/abstracts)."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# --- Search concepts and the connector interface ----------------------------

class Concept(BaseModel):
    """One block of the Boolean query: synonyms joined with OR.

    Blocks are joined with AND. Example: (older adults OR elderly OR aged)
    AND (exercise OR physical activity).
    """

    name: str = Field(description="Short label, e.g. 'population: older adults'.")
    role: Literal["population", "intervention", "outcome", "design", "other"] = Field(
        "other", description="Which part of the question this block searches.")
    keywords: list[str] = Field(
        min_length=1,
        description="Free-text synonyms and spelling variants searched in title/abstract. "
                    "Plain words or phrases, no wildcards, no Boolean operators.")
    mesh_terms: list[str] = Field(
        default_factory=list,
        description="Matching MeSH headings (PubMed only), exact official spelling.")
    publication_types: list[str] = Field(
        default_factory=list,
        description="PubMed publication types (tag [pt]), design block only, e.g. "
                    "'randomized controlled trial'. Leave empty otherwise.")


def clean_term(term: str) -> str:
    """Strip characters that are syntax in some database (quotes, brackets, commas, |, *)."""
    term = re.sub(r'["()\[\],|*:]', " ", term)
    return re.sub(r"\s+", " ", term).strip()


def unique_terms(terms: list[str]) -> list[str]:
    seen, out = set(), []
    for t in map(clean_term, terms):
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def phrase(term: str) -> str:
    """Quote multi-word terms so they are searched as a phrase."""
    return f'"{term}"' if " " in term else term


def join_blocks(blocks: list[str]) -> str:
    return " AND ".join(f"({b})" for b in blocks if b)


class Source:
    """Base class of a connector. Subclasses set ``name`` and implement the three methods.

    ``count`` returns the number of hits only (one small request, no records), with
    the same filters as ``search``: the search strategy uses it to check that a
    query is neither far too broad nor far too narrow before anything is downloaded.
    """

    name: str = ""

    def build_query(self, concepts: list[Concept]) -> str:
        raise NotImplementedError

    def search(self, query: str, *, years: dict | None = None, languages: list[str] | None = None,
               max_records: int = 500) -> SearchResult:
        raise NotImplementedError

    def count(self, query: str, *, years: dict | None = None,
              languages: list[str] | None = None) -> int:
        raise NotImplementedError
