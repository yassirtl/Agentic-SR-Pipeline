"""Part 4: search strategy agent, database connectors, HTTP client.

Connector tests replay *recorded* real API exchanges (tests/fixtures/sources/,
recorded once from the live APIs). The replay transport fails on any request
that was not recorded, so these tests also check the exact requests sent.
"""

import json
import os
from pathlib import Path

import httpx
import pytest

from sr_pipeline.agents.planner import make_planner
from sr_pipeline.agents.search_strategy import SearchPlan, build_queries, make_search_strategy
from sr_pipeline.config import load_config
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.protocol import load_template
from sr_pipeline.sources import Concept, HttpClient, HttpError, search_all
from sr_pipeline.sources.arxiv import ArxivSource
from sr_pipeline.sources.base import normalize_doi
from sr_pipeline.sources.http import cache_key
from sr_pipeline.sources.openalex import OpenAlexSource, rebuild_abstract
from sr_pipeline.sources.pubmed import PubMedSource

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "sources"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


class Replay:
    """httpx transport serving recorded exchanges; identical requests are served in order."""

    def __init__(self, *fixtures: dict):
        self.exchanges = [e for f in fixtures for e in f["exchanges"]]
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        u = request.url
        key = (f"{u.scheme}://{u.host}{u.path}", dict(u.params.multi_items()))
        self.calls.append(key)
        for i, e in enumerate(self.exchanges):
            if (e["url"], e["params"]) == key:
                self.exchanges.pop(i)
                return httpx.Response(e["status"], text=e["body"])
        raise AssertionError(f"request not recorded: {key}")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class FakeTime:
    """Clock that only moves when sleep() is called: rate-limit tests take no time."""

    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def client(transport, tmp_path=None, **kw) -> tuple[HttpClient, FakeTime]:
    t = FakeTime()
    return HttpClient(cache_dir=tmp_path, transport=transport, sleep=t.sleep, clock=t.clock,
                      **kw), t


# --- connectors on recorded responses ---------------------------------------------

CONNECTORS = {
    "openalex": lambda http: OpenAlexSource(http, per_page=2),
    "pubmed": lambda http: PubMedSource(http),
    "arxiv": lambda http: ArxivSource(http, page_size=2),
}


def run_recorded(name: str):
    fx = load_fixture(name)
    replay = Replay(fx)
    http, t = client(replay.transport())
    source = CONNECTORS[name](http)
    concepts = [Concept(**c) for c in fx["concepts"]]
    query = source.build_query(concepts)
    result = source.search(query, years=fx["years"], languages=["English"],
                           max_records=fx["max_records"])
    return fx, replay, result, t


@pytest.mark.parametrize("name", list(CONNECTORS))
def test_connector_parses_recorded_response(name):
    fx, replay, result, _ = run_recorded(name)
    assert result.query.startswith(fx["query"]) or fx["query"] in result.query
    assert not replay.exchanges            # every recorded request was sent ...
    assert len(result.records) == 3        # ... and max_records was respected
    assert result.total_hits > 3 and result.capped
    ids = [r.id for r in result.records]
    assert len(set(ids)) == 3 and all(i.startswith(f"{name}:") for i in ids)
    for r in result.records:
        assert r.source == name and r.title and r.authors and r.url
        assert 2015 <= r.year <= 2025      # year filter applied by the database
    assert any(r.abstract for r in result.records)


def test_openalex_fields_and_retry_on_recorded_429():
    _, _, result, t = run_recorded("openalex")
    r = result.records[0]
    assert r.source_id.startswith("W") and r.venue
    assert all(d is None or d.startswith("10.") for d in (x.doi for x in result.records))
    # The recording contains a real 429 on page 2: retried once, then served.
    assert len(t.sleeps) >= 1


def test_pubmed_fields():
    _, replay, result, _ = run_recorded("pubmed")
    assert [c[0].rsplit("/", 1)[1] for c in replay.calls] == ["esearch.fcgi", "efetch.fcgi"]
    assert "english[la]" in result.query
    for r in result.records:
        assert r.pmid == r.source_id and r.url.endswith(f"/{r.pmid}/")
        assert (r.oa_pdf_url is not None) == (r.pmcid is not None)
    assert any(r.doi for r in result.records)


def test_arxiv_fields():
    _, _, result, _ = run_recorded("arxiv")
    assert "submittedDate:[201501010000 TO 202512312359]" in result.query
    for r in result.records:
        assert "v" not in r.source_id.split(".")[-1]   # version suffix dropped
        assert r.oa_pdf_url                            # arXiv is always open access


def test_rebuild_abstract_and_doi():
    assert rebuild_abstract({"exercise": [1], "Regular": [0], "helps.": [2]}) == \
        "Regular exercise helps."
    assert rebuild_abstract(None) is None
    assert normalize_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert normalize_doi("doi:10.1/X") == "10.1/x"


# --- query builders ---------------------------------------------------------------

CONCEPTS = [
    Concept(name="population", keywords=["older adults", "Elderly", "elderly"],
            mesh_terms=["Aged"]),
    Concept(name="intervention", keywords=['exercis*', '"physical, activity"']),
]


def test_query_builders_write_each_database_syntax():
    q = build_queries(CONCEPTS, ["openalex", "pubmed", "arxiv"])
    # duplicates removed (case-insensitive), wildcards / quotes / commas stripped
    assert q["openalex"] == '("older adults" OR Elderly) AND (exercis OR "physical activity")'
    assert q["pubmed"] == ('("older adults"[tiab] OR Elderly[tiab] OR Aged[mh]) AND '
                           '(exercis[tiab] OR "physical activity"[tiab])')
    assert q["arxiv"] == ('(ti:"older adults" OR abs:"older adults" OR ti:Elderly OR '
                          'abs:Elderly) AND (ti:exercis OR abs:exercis OR '
                          'ti:"physical activity" OR abs:"physical activity")')


# --- HTTP client: cache, retries, rate limit ------------------------------------------

def ok(text="ok"):
    return lambda request: httpx.Response(200, text=text)


def test_cache_hit_makes_no_request_and_hides_api_key(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text='{"n": 1}')

    http, _ = client(httpx.MockTransport(handler), tmp_path)
    assert http.get_json("https://x.org/a", {"q": "1", "api_key": "SECRET"}) == {"n": 1}
    # Same request without the key, from a new client: served from disk.
    def boom(request):
        raise AssertionError("network used")

    http2, _ = client(httpx.MockTransport(boom), tmp_path)
    assert http2.get_json("https://x.org/a", {"q": "1"}) == {"n": 1}
    assert len(calls) == 1 and http2.network_calls == 0
    assert cache_key("https://x.org/a", {"q": "1", "api_key": "k"}) == \
        cache_key("https://x.org/a", {"q": "1"})
    assert "SECRET" not in "".join(p.read_text() for p in tmp_path.iterdir())
    # refresh=True ignores the cache
    http3, _ = client(httpx.MockTransport(handler), tmp_path, refresh=True)
    http3.get_text("https://x.org/a", {"q": "1"})
    assert len(calls) == 2


def test_retry_after_429_then_success():
    answers = iter([httpx.Response(429, headers={"Retry-After": "2"}),
                    httpx.Response(503), httpx.Response(200, text="done")])
    http, t = client(httpx.MockTransport(lambda r: next(answers)), backoff=1.0)
    assert http.get_text("https://x.org/a") == "done"
    assert t.sleeps == [2.0, 2.0]      # Retry-After honoured, then backoff 1 * 2**1
    assert http.network_calls == 3


def test_client_errors_are_not_retried_and_retries_are_bounded():
    http, t = client(httpx.MockTransport(lambda r: httpx.Response(400, text="bad query")))
    with pytest.raises(HttpError, match="400"):
        http.get_text("https://x.org/a")
    assert http.network_calls == 1 and t.sleeps == []

    http, t = client(httpx.MockTransport(lambda r: httpx.Response(500)), max_retries=2)
    with pytest.raises(HttpError, match="gave up after 3 attempts"):
        http.get_text("https://x.org/a")
    assert http.network_calls == 3 and t.sleeps == [1.0, 2.0]


def test_rate_limit_spaces_requests_per_host():
    http, t = client(httpx.MockTransport(ok()), min_interval={"slow.org": 3.0})
    for _ in range(3):
        http.get_text("https://slow.org/a")
    http.get_text("https://fast.org/a")       # other host: no wait
    assert t.sleeps == [3.0, 3.0]


def test_search_all_keeps_going_when_one_database_fails():
    pubmed = load_fixture("pubmed")
    replay = Replay(pubmed)

    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(400, text="bad filter")
        return replay(request)

    http, _ = client(httpx.MockTransport(handler))
    concepts = [Concept(**c) for c in pubmed["concepts"]]
    strategy = {"queries": build_queries(concepts, ["openalex", "pubmed"]),
                "filters": {"years": pubmed["years"], "languages": ["English"]}}
    results, errors = search_all(strategy, http, max_records=3, env={})
    assert list(results) == ["pubmed"] and len(results["pubmed"].records) == 3
    assert "400" in errors["openalex"]


# --- search strategy agent ---------------------------------------------------------

PROTOCOL = {
    "question": "Does exercise reduce depressive symptoms in older adults?",
    "title": "Exercise for depression in older adults",
    "pico": {"population": "adults aged 65 and over", "intervention": "exercise",
             "comparator": "usual care", "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": ["Aged 65+"], "exclusion_criteria": [],
    "study_designs": ["randomized controlled trial"], "rationale": "r",
    "databases": ["pubmed", "arxiv"], "years": {"start": 2015, "end": None},
    "languages": ["English"],
}
PLAN = {"concepts": [c.model_dump() for c in CONCEPTS], "rationale": "two blocks"}


def test_search_strategy_node_builds_queries_for_protocol_databases():
    llm = ScriptedLLM({"search_strategy": [PLAN]})
    out = make_search_strategy(llm)({"protocol": PROTOCOL, "completed_stages": ["planner"]})
    s = out["search_strategy"]
    assert set(s["queries"]) == {"pubmed", "arxiv"}            # only the protocol's databases
    assert s["queries"]["pubmed"].endswith("Aged[mh]) AND (exercis[tiab] OR "
                                           '"physical activity"[tiab])')
    assert s["filters"] == {"years": {"start": 2015, "end": None}, "languages": ["English"]}
    assert out["completed_stages"] == ["search_strategy"]
    entry = out["audit_log"][0]
    assert entry["round"] == 1 and entry["queries"] == s["queries"]
    prompt = llm.calls[0][1]
    assert "adults aged 65 and over" in prompt and "high recall" in prompt


def test_search_plan_schema_rejects_empty_concepts():
    with pytest.raises(ValueError):
        SearchPlan(concepts=[], rationale="x")
    with pytest.raises(ValueError):
        Concept(name="x", keywords=[])


def test_planner_then_search_strategy_in_the_graph():
    draft = {k: PROTOCOL[k] for k in ("title", "pico", "inclusion_criteria",
                                      "exclusion_criteria", "study_designs", "rationale")}
    llm = ScriptedLLM({"planner": [draft], "search_strategy": [PLAN],
                       "coordinator": [{"next": s, "rationale": "r"} for s in STAGES]})
    template = load_template(ROOT / "protocols" / "template.yaml")
    graph = build_graph(llm, stages={"planner": make_planner(llm, template),
                                     "search_strategy": make_search_strategy(llm)})
    state = graph.invoke({"question": PROTOCOL["question"]}, run_config("t-search"))
    assert state["completed_stages"] == list(STAGES)
    # Template scope (all three databases) flows into the strategy.
    assert set(state["search_strategy"]["queries"]) == {"openalex", "pubmed", "arxiv"}


# --- config: .env loading ---------------------------------------------------------

def test_dotenv_is_loaded_and_shell_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SR_MAX_RECORDS=7\nSR_MAX_STEPS=3\n"
                                   "SR_CONTACT_EMAIL=me@example.org\n")
    fake_env = {k: v for k, v in os.environ.items() if not k.startswith("SR_")}
    fake_env.update(SR_PROJECT_ROOT=str(tmp_path), SR_MAX_STEPS="9")
    monkeypatch.setattr(os, "environ", fake_env)   # load_dotenv writes here, not the real env
    cfg = load_config()
    assert cfg.max_records == 7                    # from .env
    assert cfg.max_steps == 9                      # shell variable beats .env
    assert cfg.contact_email == "me@example.org"
    assert cfg.http_cache_dir == tmp_path.resolve() / ".cache" / "http"
