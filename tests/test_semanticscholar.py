"""Part 12: Semantic Scholar connector (bulk search): query syntax, years, record mapping,
count, registry and deduplication with another source."""

import json

import httpx

from sr_pipeline.dedup import deduplicate
from sr_pipeline.sources import SOURCES, HttpClient, make_source
from sr_pipeline.sources.base import Concept, Record
from sr_pipeline.sources.semanticscholar import (API_URL, SemanticScholarSource, parse_paper,
                                                 year_param)


def test_query_syntax_and_years():
    src = SemanticScholarSource(HttpClient())
    q = src.build_query([Concept(name="a", keywords=["ECG", "heart rhythm"]),
                         Concept(name="b", keywords=["FPGA"])])
    assert q == '(ECG | "heart rhythm") + (FPGA)'
    assert year_param({"start": 2021}) == "2021-"
    assert year_param({"start": 2015, "end": 2025}) == "2015-2025"
    assert year_param({"start": 2021, "end_date": "2026-09-21"}) == "2021-2026"
    assert year_param(None) is None


def test_record_mapping():
    r = parse_paper({"paperId": "abc", "title": " A  title ", "year": 2022,
                     "authors": [{"name": "Ann"}], "venue": "",
                     "externalIds": {"DOI": "10.1/X", "PubMed": 123, "PubMedCentral": "999",
                                     "ArXiv": "2101.1"},
                     "openAccessPdf": {"url": "https://x/pdf"}})
    assert (r.id, r.title, r.doi, r.pmid, r.pmcid, r.venue) == (
        "semanticscholar:abc", "A title", "10.1/x", "123", "PMC999", None)
    assert r.url.endswith("/paper/abc") and r.oa_pdf_url == "https://x/pdf"
    assert parse_paper({"paperId": "x", "title": ""}) is None


def test_count_reads_total_and_sends_the_filters():
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, text=json.dumps({"total": 42, "data": []}))
    http = HttpClient(transport=httpx.MockTransport(handler))
    n = SemanticScholarSource(http).count("(ECG)", years={"start": 2021})
    assert n == 42 and seen["year"] == "2021-" and seen["fields"] == "paperId"
    assert str(API_URL).endswith("/paper/search/bulk")


def test_registry_needs_no_key_and_dedups_with_openalex():
    assert "semanticscholar" in SOURCES
    assert isinstance(make_source("semanticscholar", HttpClient(), env={}),
                      SemanticScholarSource)
    a = Record(id="openalex:W1", source="openalex", source_id="W1", title="Same paper",
               doi="10.1/same")
    b = parse_paper({"paperId": "s1", "title": "Same paper",       # through the connector
                     "externalIds": {"DOI": "10.1/SAME", "PubMed": 5}})
    out = deduplicate([a.model_dump(), b.model_dump()])
    assert len(out["papers"]) == 1
