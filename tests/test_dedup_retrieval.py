"""Part 5: deduplication rules, retrieval stage, PRISMA identification counts.

Duplicates are crafted after real cases seen in the development data
(runs/20261107-160209-aec820): the same trial in OpenAlex and PubMed, a Figshare
deposit of a BMC article, F1000 article versions, a "Correction:" notice, a
conference abstract with the article's title but its own DOI.
"""

from sr_pipeline.agents.retrieval import describe_search, make_retrieval
from sr_pipeline.coordinator import build_prompt, summarize_state
from sr_pipeline.dedup import base_doi, deduplicate, is_deposit, normalize_title
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.sources import Record, SearchResult

TITLE = "Effects of aerobic exercise on depressive symptoms in older adults: a randomized trial"


def rec(rid: str, title: str = TITLE, **kw) -> dict:
    source = rid.split(":")[0]
    return {"id": rid, "source": source, "source_id": rid.split(":")[1], "doi": None,
            "title": title, "abstract": None, "authors": [], "year": 2021, "venue": None,
            "url": None, "oa_pdf_url": None, "pmid": None, "pmcid": None, **kw}


def merged(records: list[dict]) -> dict:
    return deduplicate(records)


# --- identifier rules ----------------------------------------------------------

def test_same_doi_in_two_databases_is_one_paper():
    out = merged([rec("openalex:W1", doi="10.1/abc"),
                  rec("pubmed:11", doi="10.1/abc", abstract="Pubmed abstract.", pmid="11")])
    assert out["counts"] == {"records": 2, "duplicates_removed": 1, "unique": 1,
                             "by_rule": {"id": 0, "doi": 1, "pmid": 0, "pmcid": 0, "title": 0}}
    paper = out["papers"][0]
    assert paper["paper_id"] == "doi:10.1/abc"
    assert paper["sources"] == ["pubmed", "openalex"]
    assert paper["record_ids"] == ["openalex:W1", "pubmed:11"]
    assert paper["abstract"] == "Pubmed abstract." and paper["pmid"] == "11"
    assert out["duplicates"] == [{"paper_id": "doi:10.1/abc", "record": "pubmed:11",
                                  "matched": "openalex:W1", "rule": "doi"}]


def test_pmid_links_a_record_without_doi():
    out = merged([rec("openalex:W1", title="Totally different wording here", pmid="42"),
                  rec("pubmed:42", pmid="42", doi="10.1/x")])
    assert out["counts"]["unique"] == 1 and out["counts"]["by_rule"]["pmid"] == 1
    assert out["papers"][0]["paper_id"] == "doi:10.1/x"  # the DOI wins, from either record


def test_same_record_returned_twice_by_a_database():
    out = merged([rec("openalex:W1", doi="10.1/a"), rec("openalex:W1", doi="10.1/a")])
    assert out["counts"]["by_rule"]["id"] == 1 and out["counts"]["unique"] == 1
    assert out["papers"][0]["sources"] == ["openalex"]


def test_identifier_chain_groups_three_records():
    # A=B by DOI, B=C by PMID -> one paper.
    out = merged([rec("openalex:W1", doi="10.1/a"),
                  rec("pubmed:7", doi="10.1/a", pmid="7"),
                  rec("openalex:W2", title="Other title words for this one", pmid="7")])
    assert out["counts"]["unique"] == 1 and out["counts"]["duplicates_removed"] == 2


def test_pmid_match_with_different_dois_is_kept_apart():
    out = merged([rec("openalex:W1", doi="10.1/a", pmid="7"),
                  rec("pubmed:7", doi="10.9/zzz", pmid="7")])
    assert out["counts"]["unique"] == 2
    assert out["conflicts"][0]["reason"] == "different DOIs"


# --- title rule ------------------------------------------------------------------

def test_title_variants_merge():
    a = rec("openalex:W1", title="Effects of aerobic exercise on depressive symptoms in "
                                 "older adults – a randomized trial.", year=2021)
    b = rec("pubmed:9", title="Effects of aerobic exercise on depressive symptoms in "
                              "older adults - a randomised trial", year=2022, pmid="9")
    out = merged([a, b])
    assert out["counts"]["by_rule"]["title"] == 1 and out["counts"]["unique"] == 1
    assert out["papers"][0]["paper_id"] == "pmid:9"


def test_normalize_title():
    assert normalize_title("Exercise &amp; Dépression: a  Trial.") == "exercise depression a trial"


def test_title_match_with_different_dois_is_kept_apart():
    # Conference abstract vs article (seen: 10.1093/geroni/... vs the journal DOI).
    out = merged([rec("openalex:W1", doi="10.1136/bmjopen-1"),
                  rec("openalex:W2", doi="10.1093/geroni/igae098")])
    assert out["counts"]["unique"] == 2
    assert out["conflicts"] == [{"records": ["openalex:W1", "openalex:W2"], "rule": "title 100",
                                 "reason": "different DOIs"}]


def test_title_guards_years_short_titles_and_notices():
    out = merged([rec("openalex:W1", year=2015), rec("openalex:W2", year=2018)])
    assert out["counts"]["unique"] == 2 and out["conflicts"][0]["reason"] == "years differ"

    out = merged([rec("openalex:W1", title="Exercise and depression"),
                  rec("openalex:W2", title="Exercise and depression.")])
    assert out["counts"]["unique"] == 2 and out["conflicts"][0]["reason"] == "title too short"

    # Only long titles score >= 95 despite the "Correction:" prefix (real case: Baduanjin).
    long = ("Effects of Baduanjin exercise on sleep quality, depression and anxiety in "
            "community-dwelling older adults: a systematic review and meta-analysis")
    out = merged([rec("openalex:W1", title=f"Correction: {long}"),
                  rec("openalex:W2", title=long)])
    assert out["counts"]["unique"] == 2 and out["conflicts"][0]["reason"] == "notice vs article"


def test_deposit_doi_merges_with_the_article_and_keeps_the_journal_doi():
    # A Figshare "collection" of a BMC paper's supplementary files, deposited later.
    out = merged([rec("openalex:W1", doi="10.6084/m9.figshare.c.8333692", year=2024,
                      abstract="Deposit text."),
                  rec("openalex:W2", doi="10.1186/s12877-026-07071-z", year=2021)])
    assert out["counts"]["unique"] == 1 and out["counts"]["by_rule"]["title"] == 1
    assert out["papers"][0]["doi"] == "10.1186/s12877-026-07071-z"
    assert out["papers"][0]["paper_id"] == "doi:10.1186/s12877-026-07071-z"


def test_doi_versions_and_deposits():
    assert base_doi("10.12688/f1000research.27123.2") == "10.12688/f1000research.27123"
    assert base_doi("10.2337/figshare.13286318.v1") == "10.2337/figshare.13286318"
    assert base_doi("10.1186/s12877-020-01824-0") == "10.1186/s12877-020-01824-0"
    assert is_deposit("10.5281/zenodo.1") and is_deposit("10.2337/figshare.1")
    assert not is_deposit("10.1186/x") and not is_deposit(None)
    out = merged([rec("openalex:W1", doi="10.12688/f1000research.27123.1"),
                  rec("pubmed:5", doi="10.12688/f1000research.27123.2", pmid="5")])
    assert out["counts"]["by_rule"]["doi"] == 1 and out["counts"]["unique"] == 1


def test_paper_id_without_any_identifier_is_the_smallest_record_id():
    out = merged([rec("pubmed:3", title="A short distinct title about walking groups"),
                  rec("openalex:W9", title="A short distinct title about walking groups.")])
    assert out["papers"][0]["paper_id"] == "openalex:W9"


def test_pmc_pdf_link_is_not_preferred():
    out = merged([rec("pubmed:1", doi="10.1/a", oa_pdf_url="https://pmc.ncbi.nlm.nih.gov/x/pdf/"),
                  rec("openalex:W1", doi="10.1/a", oa_pdf_url="https://link.springer.com/a.pdf")])
    assert out["papers"][0]["oa_pdf_url"] == "https://link.springer.com/a.pdf"


# --- retrieval stage ---------------------------------------------------------------

def record(rid: str, **kw) -> Record:
    return Record(**rec(rid, **kw))


def fake_search(results: dict, errors: dict | None = None, calls: list | None = None):
    def search(strategy, http, *, max_records, contact_email=None, env=None):
        if calls is not None:
            calls.append(max_records)
        return results, dict(errors or {})
    return search


STRATEGY = {"queries": {"openalex": "q1", "pubmed": "q2"}, "filters": {},
            "targets": {"min_hits": 20, "max_records": 2}}


def two_databases() -> dict:
    return {
        "openalex": SearchResult(source="openalex", query="q1", total_hits=5, capped=True,
                                 records=[record("openalex:W1", doi="10.1/a"),
                                          record("openalex:W2", title="Another trial of tai "
                                                                       "chi for mood in elders")]),
        "pubmed": SearchResult(source="pubmed", query="q2", total_hits=1,
                               records=[record("pubmed:1", doi="10.1/a", pmid="1")]),
    }


def test_retrieval_stage_counts_for_prisma():
    calls = []
    node = make_retrieval(None, max_records=999, search=fake_search(two_databases(), calls=calls))
    out = node({"search_strategy": STRATEGY, "completed_stages": ["planner", "search_strategy"]})
    assert calls == [2]  # the ceiling the strategy was checked against, not the fallback
    ident = out["prisma_counts"]["identification"]
    assert ident["identified"] == 6 and ident["not_retrieved"] == 3
    assert ident["downloaded"] == 3 and ident["duplicates_removed"] == 1
    assert ident["records_screened"] == 2 and ident["duplicates_by_rule"]["doi"] == 1
    assert ident["databases"]["openalex"]["capped"] is True
    assert ident["databases"]["openalex"]["not_retrieved"] == 3
    assert [r["round"] for r in out["records"]] == [1, 1, 1]
    assert [p["paper_id"] for p in out["unique_records"]] == ["doi:10.1/a", "openalex:W2"]
    assert out["retrieval"]["duplicates"][0]["rule"] == "doi"
    assert out["completed_stages"] == ["retrieval"]
    assert out["audit_log"][0]["records_screened"] == 2


def test_retrieval_keeps_earlier_prisma_counts_and_reports_errors():
    node = make_retrieval(None, search=fake_search(two_databases(), {"arxiv": "HttpError: 503"}))
    state = {"search_strategy": STRATEGY, "prisma_counts": {"other": 1}}
    out = node(state)
    assert out["prisma_counts"]["other"] == 1
    assert out["prisma_counts"]["identification"]["errors"] == {"arxiv": "HttpError: 503"}
    line = describe_search({**state, **out})
    assert "arxiv FAILED (HttpError: 503)" in line
    assert "openalex 5 hits, 2 downloaded (capped, 3 not retrieved)" in line
    assert "2 unique records to screen" in line and "20-2 hits" in line


def test_second_round_appends_records_and_replaces_unique_records():
    first = make_retrieval(None, search=fake_search(two_databases()))
    second = make_retrieval(None, search=fake_search({"pubmed": SearchResult(
        source="pubmed", query="q3", total_hits=1, records=[record("pubmed:2", doi="10.1/b")])}))
    state = {"search_strategy": STRATEGY, "completed_stages": ["planner", "search_strategy"]}
    out1 = first(state)
    state = {**state, "records": out1["records"],
             "completed_stages": state["completed_stages"] + ["retrieval", "search_strategy"]}
    out2 = second(state)
    assert out2["records"][0]["round"] == 2  # the reducer appends these to round 1's records
    assert [p["paper_id"] for p in out2["unique_records"]] == ["doi:10.1/b"]
    assert out2["prisma_counts"]["identification"]["round"] == 2


def test_coordinator_sees_the_search_results():
    node = make_retrieval(None, search=fake_search(two_databases()))
    state = {"question": "q", "search_strategy": STRATEGY}
    state = {**state, **node(state)}
    assert "Retrieval (round 1)" in summarize_state(state)
    assert "you may run 'search_strategy' again" in build_prompt(state)


def test_retrieval_inside_the_graph():
    def search_strategy_stub(state):
        return {"search_strategy": STRATEGY, "completed_stages": ["search_strategy"]}

    llm = ScriptedLLM({"coordinator": [{"next": s, "rationale": "go"} for s in STAGES]})
    graph = build_graph(llm, stages={"search_strategy": search_strategy_stub,
                                     "retrieval": make_retrieval(
                                         None, search=fake_search(two_databases()))})
    final = graph.invoke({"question": "q"}, run_config("retrieval-graph"))
    assert len(final["records"]) == 3 and len(final["unique_records"]) == 2
    assert final["prisma_counts"]["identification"]["records_screened"] == 2
    assert final["completed_stages"][-1] == "report"
