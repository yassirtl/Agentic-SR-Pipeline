"""Part 4b: search precision fixes (outcome / design blocks, count-and-refine loop).

Hit counts come from a fake transport that invents a count from the query text,
so the refine loop is tested without network or API key.
"""

import httpx
import pytest

from sr_pipeline.agents.search_strategy import (_describe_attempt, build_prompt, build_queries,
                                                choose_attempt, make_search_strategy, overlap)
from sr_pipeline.config import load_config
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.sources import Concept, HttpClient, count_all

POP = {"name": "population", "role": "population", "keywords": ["older adults", "elderly"],
       "mesh_terms": ["Aged"]}
INT = {"name": "intervention", "role": "intervention", "keywords": ["exercise"],
       "mesh_terms": ["Exercise"]}
OUT = {"name": "outcome", "role": "outcome", "keywords": ["depression", "depressive symptoms"],
       "mesh_terms": ["Depression"]}
RARE = {"name": "outcome", "role": "outcome", "keywords": ["rare marker"]}


def plan(*blocks, why="r"):
    return {"concepts": list(blocks), "rationale": why}


PROTOCOL = {
    "question": "Does exercise reduce depressive symptoms in older adults?",
    "title": "t", "rationale": "r",
    "pico": {"population": "adults aged 65 and over", "intervention": "exercise",
             "comparator": "no intervention, usual care or any control",
             "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": [], "exclusion_criteria": [],
    "study_designs": [],
    "databases": ["openalex", "pubmed"], "years": {"start": 2015, "end": None},
    "languages": ["English"],
}


def default_hits(query: str) -> int:
    """Fake database: P AND I is huge, + outcome is fine, + a rare term is tiny."""
    q = query.lower()
    if "rare marker" in q:
        return 5
    if "depress" in q:
        return 800
    return 100_000


class FakeCounts:
    """Answers count requests of the three databases in their own formats."""

    def __init__(self, hits=default_hits, fail: set[str] = frozenset()):
        self.hits, self.fail = hits, fail
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, params = request.url.host, dict(request.url.params)
        if host == "api.openalex.org":
            if "openalex" in self.fail:
                return httpx.Response(400, text="bad filter")
            assert params["per-page"] == "1"                       # count only, no records
            n = self.hits(params["filter"].split(",publication_year")[0])
            return httpx.Response(200, json={"meta": {"count": n}, "results": [{"id": "W1"}]})
        if host == "eutils.ncbi.nlm.nih.gov":
            assert params["retmax"] == "0"
            n = self.hits(params["term"])
            return httpx.Response(200, json={"esearchresult": {"count": str(n), "idlist": []}})
        if host == "export.arxiv.org":
            n = self.hits(params["search_query"])
            return httpx.Response(200, text=(
                '<feed xmlns="http://www.w3.org/2005/Atom" '
                'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
                f"<opensearch:totalResults>{n}</opensearch:totalResults></feed>"))
        raise AssertionError(f"unexpected host {host}")


def http_for(fake: FakeCounts) -> HttpClient:
    return HttpClient(cache_dir=None, transport=httpx.MockTransport(fake),
                      sleep=lambda s: None)


def run_stage(answers, fake=None, protocol=PROTOCOL, **kw):
    llm = ScriptedLLM({"search_strategy": answers})
    fake = fake or FakeCounts()
    kw.setdefault("max_records", 1000)
    node = make_search_strategy(llm, http_for(fake), min_hits=20, env={}, **kw)
    out = node({"protocol": protocol, "completed_stages": ["planner"]})
    return out, llm, fake


# --- Fix 1: outcome and design blocks ------------------------------------------------

def test_prompt_asks_for_outcome_block_and_never_a_design_block():
    prompt = build_prompt(PROTOCOL, max_records=500)
    assert "Add an outcome block when the primary outcome is specific" in prompt
    assert "Never write a study-design block" in prompt
    assert "At most 500 records per database" in prompt
    assert "Study designs: any\n" in prompt
    rct = build_prompt({**PROTOCOL, "study_designs": ["randomized controlled trial"]})
    assert "Study designs: randomized controlled trial (a validated design filter" in rct


def test_outcome_block_in_every_database_syntax():
    q = build_queries([Concept(**c) for c in (POP, INT, OUT)], ["openalex", "pubmed"])
    assert q["openalex"].count(" AND ") == 2 and "(depression OR " in q["openalex"]
    assert q["pubmed"].endswith('Depression[mh])') and "[pt]" not in q["pubmed"]


def test_specific_outcome_in_first_plan_needs_no_refinement():
    out, llm, fake = run_stage([plan(POP, INT, OUT)])
    s = out["search_strategy"]
    assert len(llm.calls) == 1 and len(s["attempts"]) == 1
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention", "outcome"]
    assert s["counts"] == {"openalex": 800, "pubmed": 800}
    assert s["attempts"][0]["reason"] == "all databases within range"
    assert len(fake.requests) == 2                    # one count request per database


# --- Fix 2: count and refine ------------------------------------------------------------

def test_too_broad_is_refined_to_narrower_plan():
    out, llm, _ = run_stage([plan(POP, INT), plan(POP, INT, OUT, why="outcome added")])
    s = out["search_strategy"]
    first, second = s["attempts"]
    assert first["counts"] == {"openalex": 100_000, "pubmed": 100_000}
    assert first["status"] == {"openalex": "too broad", "pubmed": "too broad"}
    assert "refinement requested" in first["reason"]
    assert second["status"] == {"openalex": "ok", "pubmed": "ok"}
    # the final strategy is the refined one
    assert s["chosen_attempt"] == 2 and s["rationale"] == "outcome added"
    assert "depression" in s["queries"]["openalex"]
    # the LLM saw its previous blocks, the counts and the target range
    refine = llm.calls[1][1]
    assert "100000 hits -> too broad" in refine and "between 20 and 1000" in refine
    assert "[population] population: older adults, elderly" in refine
    entry = out["audit_log"][0]
    assert entry["llm_calls"] == 2 and entry["attempts"] == 2 and entry["chosen_attempt"] == 2


def test_too_narrow_is_refined_to_broader_plan():
    out, llm, _ = run_stage([plan(POP, INT, RARE), plan(POP, INT, OUT)])
    first, second = out["search_strategy"]["attempts"]
    assert first["status"] == {"openalex": "too narrow", "pubmed": "too narrow"}
    assert "5 hits -> too narrow" in llm.calls[1][1]
    assert second["status"]["pubmed"] == "ok"
    assert out["search_strategy"]["chosen_attempt"] == 2


def shrinking_pop(n):
    """Population block with n keywords: each refinement narrows a little."""
    return {**POP, "keywords": ["older adults", "elderly", "seniors"][:n]}


def by_terms(base):
    """Fake database: fewer OR-ed terms -> fewer hits (always some progress)."""
    return FakeCounts(lambda q: base + base // 8 * q.count(" OR "))


def test_stops_after_two_refinements():
    out, llm, _ = run_stage([plan(shrinking_pop(n), INT) for n in (3, 2, 1, 1, 1)],
                            fake=by_terms(100_000))
    s = out["search_strategy"]
    assert not any(a.get("no_progress") for a in s["attempts"])
    assert len(llm.calls) == 3 and len(s["attempts"]) == 3   # first plan + 2 refinements
    assert s["attempts"][-1]["reason"].count("refinement limit reached") == 1
    assert s["chosen_attempt"] == 3                           # all equal: latest wins


def test_overshooting_refinement_does_not_replace_a_better_plan():
    def hits(q):  # attempt 2 is in range for pubmed only; attempt 3 overshoots everywhere
        q = q.lower()
        return 5 if "rare" in q else 800 if ("depress" in q and "[" in q) else 100_000
    out, _, _ = run_stage([plan(POP, INT), plan(POP, INT, OUT), plan(POP, INT, OUT, RARE)],
                          fake=FakeCounts(hits))
    s = out["search_strategy"]
    assert [sum(v != "ok" for v in a["status"].values()) for a in s["attempts"]] == [2, 1, 2]
    assert s["chosen_attempt"] == 2 and "depression" in s["queries"]["pubmed"]


def test_choose_attempt_prefers_fewest_out_of_range_then_latest():
    a = lambda n, *st: {"attempt": n, "status": dict(enumerate(st))}
    assert choose_attempt([a(1, "ok", "too broad"), a(2, "too narrow", "ok")])["attempt"] == 2
    assert choose_attempt([a(1, "ok", "ok"), a(2, "too narrow", "ok")])["attempt"] == 1


def test_count_error_is_reported_not_refined():
    out, llm, _ = run_stage([plan(POP, INT, OUT)], fake=FakeCounts(fail={"openalex"}))
    a = out["search_strategy"]["attempts"][0]
    assert a["status"] == {"pubmed": "ok", "openalex": "error"} and "400" in a["errors"]["openalex"]
    assert len(llm.calls) == 1


def test_no_refinement_when_disabled():
    out, llm, _ = run_stage([plan(POP, INT)], max_refinements=0)
    assert len(llm.calls) == 1
    assert out["search_strategy"]["attempts"][0]["reason"].count("refinement limit reached") == 1


# --- counts use the same filters as the real search ----------------------------------

def test_count_requests_carry_the_search_filters():
    fake = FakeCounts()
    queries = build_queries([Concept(**POP), Concept(**INT)], ["openalex", "pubmed", "arxiv"])
    filters = {"years": {"start": 2015, "end": 2024}, "languages": ["English"]}
    counts, errors = count_all(queries, filters, http_for(fake), env={})
    assert counts == {"openalex": 100_000, "pubmed": 100_000, "arxiv": 100_000} and not errors
    by_host = {r.url.host: dict(r.url.params) for r in fake.requests}
    assert by_host["api.openalex.org"]["filter"].endswith(
        ",publication_year:2015-2024,language:en")
    pm = by_host["eutils.ncbi.nlm.nih.gov"]
    assert pm["term"].endswith("AND (english[la])") and (pm["mindate"], pm["maxdate"]) == \
        ("2015", "2024")
    assert by_host["export.arxiv.org"]["search_query"].endswith(
        "submittedDate:[201501010000 TO 202412312359]")


def test_refine_settings_from_config_and_env():
    cfg = load_config(env={"SR_MIN_HITS": "50"})
    assert (cfg.max_records, cfg.min_hits, cfg.max_refinements) == (3000, 50, 2)
    assert not hasattr(cfg, "broad_factor")            # dropped in Part 4c
    with pytest.raises(ValueError):
        load_config(env={"SR_MAX_RECORDS": "20000"})   # above the APIs' paging limit


# --- after the first live run: repeated plans, mixed-up blocks -----------------------

def test_prompt_keeps_concepts_apart_and_terms_short():
    prompt = build_prompt(PROTOCOL)
    assert "each term in exactly one block" in prompt
    assert "never the condition being measured" in prompt
    assert "1 to 3 words each" in prompt and "Never phrases describing a result" in prompt


def test_repeated_plan_is_flagged_and_stops_the_loop():
    out, llm, _ = run_stage([plan(POP, INT), plan(POP, INT), plan(POP, INT, OUT)])
    first, second = out["search_strategy"]["attempts"]   # Part 4d: same counts = no progress
    assert first["same_as"] is None and second["same_as"] == 1
    assert second["no_progress"] and "no progress" in second["reason"]
    assert len(llm.calls) == 2                       # no third call for a repeated plan
    assert "must differ from every previous attempt" in llm.calls[1][1]
    assert "Attempt 2 (identical to attempt 1: you did not change it)" in \
        _describe_attempt(second)


def test_overlap_lists_terms_in_several_blocks():
    mixed = {**POP, "keywords": ["older adults", "Depression"]}
    assert overlap([mixed, INT, OUT]) == ["depression"]
    assert overlap([POP, INT, OUT]) == []
    out, _, _ = run_stage([plan(mixed, INT, OUT)])
    assert out["search_strategy"]["attempts"][0]["overlap"] == ["depression"]
