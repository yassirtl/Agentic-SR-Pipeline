"""Part 4c: search recall fixes.

1. No silent truncation: max_records is the download ceiling and the top of the
   "ok" range; what stays above it is reported (capped, not_retrieved).
2. The Planner does not invent study-design restrictions.
3. Design filters are validated filters written in code, every design included.
"""

from sr_pipeline.agents.planner import build_prompt as planner_prompt
from sr_pipeline.agents.planner import choose_designs, make_planner
from sr_pipeline.agents.search_strategy import build_queries
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import Template
from sr_pipeline.sources import SearchResult
from sr_pipeline.sources.design_filters import design_block, match_design

from test_search_refine import (INT, OUT, POP, PROTOCOL, FakeCounts, by_terms, plan, run_stage,
                               shrinking_pop)

RCT = ["randomized controlled trial"]


def queries_for(designs, databases=("openalex", "pubmed", "arxiv")):
    from sr_pipeline.sources import Concept
    block, _ = design_block(designs)
    return build_queries([Concept(**POP), Concept(**INT), block], list(databases))


# --- Fix 1: no silent truncation ----------------------------------------------------

def test_hits_below_the_ceiling_are_ok_and_fully_downloaded():
    # 2,375 hits was "ok" before but only 500 were downloaded; now it is in range
    # and nothing is lost.
    out, llm, _ = run_stage([plan(POP, INT, OUT)], fake=FakeCounts(lambda q: 2375),
                            max_records=3000)
    s = out["search_strategy"]
    assert len(llm.calls) == 1 and s["attempts"][0]["status"] == {"openalex": "ok",
                                                                  "pubmed": "ok"}
    assert s["targets"] == {"min_hits": 20, "max_records": 3000, "broaden_below": 0}
    assert s["capped"] == {"openalex": False, "pubmed": False}
    assert s["not_retrieved"] == {"openalex": 0, "pubmed": 0}


def test_still_above_the_ceiling_after_refinements_is_capped_and_reported():
    # each refinement narrows a little but never gets under the ceiling
    out, llm, _ = run_stage([plan(shrinking_pop(n), INT) for n in (3, 2, 1)],
                            fake=by_terms(4000), max_records=3000)
    s = out["search_strategy"]
    assert len(llm.calls) == 3                         # first plan + 2 refinements
    assert "refinement limit reached" in s["attempts"][-1]["reason"]
    assert "download capped at 3000 per database" in s["attempts"][-1]["reason"]
    assert s["counts"] == {"openalex": 4000, "pubmed": 5000}
    assert s["capped"] == {"openalex": True, "pubmed": True}
    assert s["not_retrieved"] == {"openalex": 1000, "pubmed": 2000}
    assert out["audit_log"][0]["not_retrieved"] == {"openalex": 1000, "pubmed": 2000}
    # the LLM was told what the ceiling means
    assert "between 20 and 3000" in llm.calls[1][1]
    assert "reported as not retrieved" in llm.calls[0][1]


def test_search_result_reports_not_retrieved():
    r = SearchResult(source="pubmed", query="q", total_hits=2375, records=[], capped=True)
    assert r.not_retrieved == 2375
    assert r.model_dump()["not_retrieved"] == 2375     # written to search_results.json


# --- Fix 2: no invented study designs -------------------------------------------------

QUESTION = "Does exercise reduce depressive symptoms in older adults?"


def planner_draft(designs):
    return {"title": "t", "rationale": "r", "inclusion_criteria": [], "exclusion_criteria": [],
            "pico": {"population": "older adults", "intervention": "exercise",
                     "comparator": "any control", "outcomes": ["depressive symptoms"]},
            "study_designs": designs, "databases": ["pubmed"], "database_reason": "health"}


def run_planner(question, template, designs):
    llm = ScriptedLLM({"planner": [planner_draft(designs)]})
    return make_planner(llm, template)({"question": question, "completed_stages": []}), llm


def test_planner_drops_designs_the_question_does_not_ask_for():
    out, llm = run_planner(QUESTION, Template(),
                           ["randomized controlled trial", "cohort study", "case-control study"])
    assert out["protocol"]["study_designs"] == []
    entry = out["audit_log"][0]
    assert entry["study_designs_source"] == "none (any design)"
    assert entry["study_designs_dropped"] == ["randomized controlled trial", "cohort study",
                                              "case-control study"]
    assert "Study designs: only if the question or the user" in llm.calls[0][1]


def test_planner_keeps_designs_named_by_the_question():
    question = "In randomized trials, does exercise reduce depression in older adults?"
    out, _ = run_planner(question, Template(), RCT)
    assert out["protocol"]["study_designs"] == RCT
    assert out["audit_log"][0]["study_designs_source"] == "question"


def test_template_designs_win_exactly():
    template = Template(study_designs=["cohort study"])
    out, _ = run_planner(QUESTION, template, ["randomized controlled trial", "cohort"])
    assert out["protocol"]["study_designs"] == ["cohort study"]
    assert out["audit_log"][0]["study_designs_source"] == "template"
    assert "Study designs required by the user: cohort study" in \
        planner_prompt(QUESTION, template)


def test_choose_designs_recognises_design_words():
    for q in ("Effect of X in RCTs?", "A cohort of nurses: does X cause Y?",
              "Case-control evidence on X and Y?", "Cross-sectional prevalence of X?"):
        assert choose_designs(q, Template(), ["d"])[1] == "question", q
    assert choose_designs("Is X associated with Y?", Template(), ["d"])[0] == []


# --- Fix 3: validated design filters ----------------------------------------------------

def test_rct_filter_is_the_cochrane_filter():
    q = queries_for(RCT)
    pubmed_design = q["pubmed"].rsplit(" AND ", 1)[1]
    for term in ('"randomized controlled trial"[pt]', '"controlled clinical trial"[pt]',
                 "randomized[tiab]", "placebo[tiab]", "randomly[tiab]", "trial[ti]",
                 "randomised[tiab]"):
        assert term in pubmed_design, term
    # no exact-phrase design keyword, no MeSH heading misused as a publication type
    assert '"randomized controlled trial"[tiab]' not in q["pubmed"]
    assert "Studies\"[pt]" not in q["pubmed"]
    # keyword databases get the words, never PubMed tags
    assert "randomised" in q["openalex"] and "[" not in q["openalex"]
    assert "ti:trial" in q["arxiv"] and "abs:randomised" in q["arxiv"]
    assert "abs:trial" not in q["arxiv"]                # title-only term stays title-only


def test_every_listed_design_appears():
    designs = ["randomized controlled trial", "cohort study", "case-control study"]
    block, mapping = design_block(designs)
    assert [m["filter"] for m in mapping] == designs
    q = queries_for(designs)["pubmed"]
    assert '"Cohort Studies"[mh]' in q and '"Case-Control Studies"[mh]' in q
    assert "randomized[tiab]" in q and "cohort[tiab]" in q and "case-control[tiab]" in q
    assert q.count(" AND ") == 2                          # one design block, designs OR-ed


def test_unknown_design_is_searched_by_name_and_flagged():
    assert match_design("non-randomised controlled trial") is None   # not the RCT filter
    assert match_design("Randomised trial") == "randomized controlled trial"
    block, mapping = design_block(["interrupted time series"])
    assert mapping == [{"design": "interrupted time series", "filter": "unvalidated"}]
    assert block.keywords == ["interrupted time series"]
    assert design_block([]) == (None, [])


def test_code_replaces_a_design_block_written_by_the_llm():
    llm_design = {"name": "design", "role": "design", "keywords": ["cohort study"],
                  "publication_types": ["Cohort Studies"]}
    out, _, fake = run_stage([plan(POP, INT, OUT, llm_design)],
                             protocol={**PROTOCOL, "study_designs": RCT},
                             fake=FakeCounts(lambda q: 800))
    s = out["search_strategy"]
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention", "outcome",
                                                  "design"]
    assert s["attempts"][0]["llm_design_blocks_dropped"] == 1
    assert '"Cohort Studies"[pt]' not in s["queries"]["pubmed"]
    assert "trial[ti]" in s["queries"]["pubmed"]
    assert s["design_filter"] == [{"design": RCT[0], "filter": RCT[0]}]
    # the counted query is the final one, design filter included
    pubmed_count = next(r for r in fake.requests if r.url.host == "eutils.ncbi.nlm.nih.gov")
    assert "trial[ti]" in dict(pubmed_count.url.params)["term"]


def test_no_designs_means_no_design_block_even_if_the_llm_writes_one():
    llm_design = {"name": "design", "role": "design", "keywords": ["randomized"]}
    out, _, _ = run_stage([plan(POP, INT, OUT, llm_design)], fake=FakeCounts(lambda q: 800))
    s = out["search_strategy"]
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention", "outcome"]
    assert "randomized" not in s["queries"]["openalex"] and s["design_filter"] == []
