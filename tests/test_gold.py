"""Part 6, step 0: gold lists, coverage lookups, end_date in the protocol and connectors.

Answers are crafted after the real ones seen on 2026-11-07 (Cai's medRxiv preprint whose
2025 journal version PubMed matches by title; a Value in Health abstract found only by
title; a Chinese-journal article in neither database). No network.
"""

from pathlib import Path

import httpx
import pytest

from sr_pipeline.evaluation.gold import coverage, load_gold, summarize
from sr_pipeline.protocol import YearRange
from sr_pipeline.sources import HttpClient
from sr_pipeline.sources.arxiv import date_clause
from sr_pipeline.sources.openalex import year_filter
from sr_pipeline.sources.pubmed import PubMedSource

ROOT = Path(__file__).resolve().parents[1]

CAI = {"row": "1", "first_author": "Cai", "year": "2024", "type": "preprint",
       "title": "Utilizing ChatGPT to select literature for meta-analysis shows workload "
                "reduction while maintaining a similar recall level as manual curation",
       "doi": "10.1101/2023.09.06.23295072", "pmid": ""}
ATTRI = {"row": "2", "first_author": "Attri", "year": "2024", "type": "conference abstract",
         "title": "Transforming systematic literature reviews: unleashing the potential of "
                  "GPT-4, a cutting-edge large language model, to elevate research synthesis",
         "doi": "", "pmid": ""}
AN = {"row": "3", "first_author": "An", "year": "2019", "type": "non-English journal article",
      "title": "Effects of physical exercise on the mental health of older adults",
      "doi": "", "pmid": ""}
GUO = {"row": "4", "first_author": "Guo", "year": "2024", "type": "journal article",
       "title": "Automated Paper Screening for Clinical Reviews Using Large Language Models",
       "doi": "10.2196/48996", "pmid": "38214966"}


def handler(request: httpx.Request) -> httpx.Response:
    url, p = request.url, dict(request.url.params)
    path = url.path
    if url.host == "api.openalex.org":
        if path == "/works/doi:10.1101/2023.09.06.23295072":
            return httpx.Response(200, json={"id": "https://openalex.org/W1",
                                             "doi": "https://doi.org/10.1101/2023.09.06.23295072",
                                             "ids": {}})
        if path == "/works/doi:10.2196/48996":
            return httpx.Response(200, json={"id": "https://openalex.org/W4", "doi": None,
                                             "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/38214966"}})
        if path == "/works" and "Transforming" in p["filter"]:
            return httpx.Response(200, json={"results": [
                {"id": "https://openalex.org/W9", "display_name": "Something else entirely",
                 "publication_year": 2024, "doi": None, "ids": {}},
                {"id": "https://openalex.org/W2", "publication_year": 2024,
                 "display_name": "MSR21 Transforming Systematic Literature Reviews: Unleashing the "
                                 "Potential of GPT-4, a Cutting-Edge Large Language Model, to "
                                 "Elevate Research Synthesis",
                 "doi": "https://doi.org/10.1016/j.jval.2024.03.1490", "ids": {}}]})
        if path == "/works":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404, text="not found")
    if path.endswith("esummary.fcgi"):
        ids = p["id"].split(",")
        known = {
            "38214966": {"title": "Automated Paper Screening for Clinical Reviews Using Large "
                                  "Language Models: Data Analysis Study", "pubdate": "2024 Jan 12",
                         "articleids": [{"idtype": "doi", "value": "10.2196/48996"}]},
            "40295957": {"title": "Utilizing Large language models to select literature for "
                                  "meta-analysis shows workload reduction while maintaining a "
                                  "similar recall level as manual curation.",
                         "pubdate": "2025 Apr 28",
                         "articleids": [{"idtype": "doi", "value": "10.1186/s12874-025-02569-3"}]},
        }
        result = {"uids": ids, **{i: known.get(i, {"error": "cannot get document summary"})
                                  for i in ids}}
        return httpx.Response(200, json={"result": result})
    if path.endswith("esearch.fcgi"):
        term = p["term"]
        ids = ["40295957"] if "Utilizing ChatGPT" in term else []
        return httpx.Response(200, json={"esearchresult": {"idlist": ids}})
    raise AssertionError(f"unexpected request {url}")


def run(rows, tmp_path):
    http = HttpClient(tmp_path / "http", transport=httpx.MockTransport(handler),
                      sleep=lambda s: None)
    return coverage(rows, http), http


def test_coverage_routes_and_rules(tmp_path):
    out, _ = run([CAI, ATTRI, AN, GUO], tmp_path)
    cai, attri, an, guo = out
    # Preprint: OpenAlex by DOI; PubMed's title match is the 2025 journal version (other DOI): rejected.
    assert (cai["in_openalex"], cai["openalex_by"], cai["in_pubmed"]) == ("yes", "doi", "no")
    # Conference abstract without identifiers: found by title, its DOI recorded for Part 10.
    assert (attri["in_openalex"], attri["openalex_by"], attri["openalex_id"]) == ("yes", "title", "W2")
    assert attri["found_doi"] == "10.1016/j.jval.2024.03.1490"
    assert (an["in_openalex"], an["in_pubmed"]) == ("no", "no")
    assert (guo["in_pubmed"], guo["pubmed_by"], guo["found_pmid"]) == ("yes", "pmid", "38214966")
    s = summarize(out)
    assert (s["gold"], s["openalex"], s["pubmed"], s["either"]) == (4, 3, 1, 3)
    assert s["by_type"]["non-English journal article"] == "0/1"


def test_coverage_rerun_is_served_from_cache(tmp_path):
    run([CAI, ATTRI], tmp_path)
    _, http = run([CAI, ATTRI], tmp_path)
    assert http.network_calls == 0


def test_gold_files_match_the_reviews():
    sizes = {"T1": 18, "T3": 30, "D0": 15, "dropped_kim2025": 14}
    for topic, n in sizes.items():
        rows = load_gold(ROOT / "evaluation" / "gold" / f"{topic}.csv")
        assert len(rows) == n
        assert [int(r["row"]) for r in rows] == list(range(1, n + 1))
        assert all(r["title"] and r["first_author"] and r["year"] for r in rows)
    t1 = load_gold(ROOT / "evaluation" / "gold" / "T1.csv")
    assert all(r["doi"] for r in t1)


# --- end_date ------------------------------------------------------------------------

def test_end_date_sets_end_year_and_checks_consistency():
    y = YearRange(end_date="2024-01-07")
    assert y.end == 2024
    with pytest.raises(ValueError, match="disagree"):
        YearRange(end=2023, end_date="2024-01-07")
    with pytest.raises(ValueError, match="after"):
        YearRange(start=2025, end_date="2024-01-07")


def test_end_date_reaches_every_database():
    years = YearRange(start=2015, end_date="2021-02-28").model_dump(mode="json")
    assert year_filter(years) == "publication_year:>2014,to_publication_date:2021-02-28"
    assert PubMedSource._dates(years) == {"datetype": "pdat", "mindate": "2015",
                                          "maxdate": "2021/02/28"}
    assert date_clause(years) == "submittedDate:[201501010000 TO 202102282359]"
    # Without end_date nothing changes.
    assert year_filter({"start": 2015, "end": None}) == "publication_year:>2014"
    assert PubMedSource._dates({"start": None, "end": None}) == {}
