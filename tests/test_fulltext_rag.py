"""Part 5: open-access full text, chunking, hybrid index (FAISS + BM25 + RRF), re-ranker.

No network: PMC answers are replayed from a real efetch response, trimmed
(tests/fixtures/fulltext/pmc_efetch.xml: one open article, PMC9338474, and one
whose publisher forbids XML full text, PMC8282290). Embeddings come from
``FakeEmbedder`` (hashed words) or a fake OpenAI client.
"""

from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from sr_pipeline.llm import ScriptedLLM
from sr_pipeline.rag import FakeEmbedder, HybridIndex, OpenAIEmbedder, chunk_paper, rerank, rrf
from sr_pipeline.sources import HttpClient
from sr_pipeline.sources.fulltext import FullText, Section, fetch_fulltexts, parse_pmc_articles

FIXTURE = Path(__file__).parent / "fixtures" / "fulltext" / "pmc_efetch.xml"


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF with a text layer (offsets computed, so pypdf reads it)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1,
                                                                            xref)
    return bytes(out)


# --- PMC XML -------------------------------------------------------------------------

def test_jats_sections_tables_and_forbidden_full_text():
    parsed = parse_pmc_articles(FIXTURE.read_text())
    sections, reason = parsed["PMC9338474"]
    assert reason is None
    titles = [s.title for s in sections]
    assert titles[:3] == ["Background", "Objectives", "Methods > Study design"]
    assert "Methods > Intervention > Control group" in titles
    table = sections[-1]
    assert table.title == "Table 1"
    assert "Inclusion criteria | Exclusion criteria" in table.text
    # Lifted out of the body: the table's cells are not repeated in a paragraph.
    assert not any("Clinical exclusion of delirium" in s.text for s in sections[:-1])
    assert parsed["PMC8282290"] == ([], "PMC: publisher does not allow full text in XML form")


class Server:
    """Fake web: PMC efetch replays the fixture; PDF links answer as configured."""

    def __init__(self, pages: dict[str, httpx.Response]):
        self.pages, self.calls = pages, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        self.calls.append((url, dict(request.url.params)))
        if url.endswith("/efetch.fcgi"):
            assert request.url.params["db"] == "pmc"
            return httpx.Response(200, text=FIXTURE.read_text())
        return self.pages[url]


def http_for(server: Server, tmp_path) -> HttpClient:
    return HttpClient(tmp_path / "http", transport=httpx.MockTransport(server),
                      sleep=lambda s: None, max_retries=0)


def test_fetch_fulltexts_routes_and_reasons(tmp_path):
    pdf = make_pdf("Participants were aged 65 years or older")
    server = Server({
        "https://link.springer.com/ok.pdf": httpx.Response(
            200, content=pdf, headers={"content-type": "application/pdf"}),
        "https://publisher.org/login.pdf": httpx.Response(
            200, text="<!DOCTYPE html><p>Sign in</p>", headers={"content-type": "text/html"}),
        "https://onlinelibrary.wiley.com/x.pdf": httpx.Response(403, text="Forbidden"),
    })
    papers = [
        {"paper_id": "a", "pmcid": "PMC9338474"},
        {"paper_id": "b", "pmcid": "pmc8282290", "oa_pdf_url": "https://link.springer.com/ok.pdf"},
        {"paper_id": "c", "oa_pdf_url": "https://publisher.org/login.pdf"},
        {"paper_id": "d", "oa_pdf_url": "https://onlinelibrary.wiley.com/x.pdf"},
        {"paper_id": "e", "oa_pdf_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/pdf/"},
        {"paper_id": "f"},
    ]
    out = fetch_fulltexts(papers, http_for(server, tmp_path), env={})
    assert list(out) == ["a", "b", "c", "d", "e", "f"]
    assert out["a"].status == "pmc_xml" and out["a"].words > 2000
    # Forbidden in XML -> the publisher PDF is used instead.
    assert out["b"].status == "pdf"
    assert "aged 65 years or older" in out["b"].sections[0].text
    assert out["c"].status == "abstract_only" and out["c"].reason == \
        "PDF link returned text/html, not a PDF"
    assert out["d"].reason == "PDF: HTTP 403"
    assert out["e"].reason == "no open-access full text"  # PMC PDF links are never tried
    assert out["f"].reason == "no open-access full text"
    efetch = [params for url, params in server.calls if url.endswith("efetch.fcgi")]
    assert len(efetch) == 1 and efetch[0]["id"] == "PMC9338474,PMC8282290"  # one batch


def test_pmc_failure_reason_is_kept_when_the_pdf_also_fails(tmp_path):
    server = Server({"https://x.org/a.pdf": httpx.Response(403, text="no")})
    out = fetch_fulltexts([{"paper_id": "b", "pmcid": "PMC8282290",
                            "oa_pdf_url": "https://x.org/a.pdf"}],
                          http_for(server, tmp_path), env={})
    assert out["b"].reason == ("PMC: publisher does not allow full text in XML form; "
                               "PDF: HTTP 403")


def test_binary_cache_serves_the_second_request(tmp_path):
    server = Server({"https://x.org/a.pdf": httpx.Response(
        200, content=b"%PDF-1.4 x", headers={"content-type": "application/pdf"})})
    http = http_for(server, tmp_path)
    assert http.get_bytes("https://x.org/a.pdf") == (b"%PDF-1.4 x", "application/pdf")
    assert http.get_bytes("https://x.org/a.pdf") == (b"%PDF-1.4 x", "application/pdf")
    assert http.network_calls == 1


# --- chunking ----------------------------------------------------------------------

def test_chunks_abstract_first_sections_windows_and_merges():
    paper = {"paper_id": "p", "title": "Tai chi for mood", "abstract": "We tested tai chi."}
    long_text = " ".join(f"w{i}" for i in range(600))
    ft = FullText(paper_id="p", status="pdf", sections=[
        Section(title="Ethics", text="Approved by the board."),
        Section(title="Funding", text="None."),
        Section(title="Methods", text=long_text),
        Section(title="Table 1", text="Age | 70"),
    ])
    chunks = chunk_paper(paper, ft, size=250, overlap=40)
    assert chunks[0]["section"] == "Title and abstract"
    assert chunks[0]["text"] == "[Title and abstract] Tai chi for mood. We tested tai chi."
    assert chunks[1]["section"] == "Ethics + Funding + Methods"  # short sections merged forward
    windows = [c["text"].split("] ", 1)[1].split() for c in chunks if "Methods" in c["section"]]
    assert len(windows) == 3  # 5 + 600 words, windows of 250 starting every 210 words
    assert [len(w) for w in windows] == [250, 250, 185]
    assert windows[0][-40:] == windows[1][:40]  # 40 words of overlap
    assert windows[1][0] == "w205"
    assert chunks[-1]["section"] == "Table 1"  # tables stay alone
    assert [c["chunk_id"] for c in chunks] == [f"p::{i}" for i in range(len(chunks))]
    assert chunk_paper(paper)[0]["paper_id"] == "p" and len(chunk_paper(paper)) == 1


# --- hybrid index --------------------------------------------------------------------

def real_paper_chunks() -> list[dict]:
    sections, _ = parse_pmc_articles(FIXTURE.read_text())["PMC9338474"]
    ft = FullText(paper_id="pmc:PMC9338474", status="pmc_xml", sections=sections)
    other = FullText(paper_id="doi:10.1/other", status="pdf", sections=[
        Section(title="Methods", text="Participants walked in groups three times a week for "
                                      "twelve weeks. Mood was rated with the GDS-15 scale."),
    ])
    return (chunk_paper({"paper_id": "pmc:PMC9338474", "title": "Depression in geriatric "
                         "inpatients", "abstract": "Protocol of a trial."}, ft)
            + chunk_paper({"paper_id": "doi:10.1/other", "title": "Walking groups",
                           "abstract": "A walking trial."}, other))


def test_known_query_returns_the_right_chunk():
    index = HybridIndex.build(real_paper_chunks(), FakeEmbedder())
    hits = index.search("who performs the group allocation and how is randomization done", k=3)
    assert hits[0]["section"] == "Methods > Allocation"
    assert hits[0]["dense_rank"] is not None and hits[0]["sparse_rank"] is not None
    hits = index.search("inclusion and exclusion criteria delirium", k=1)
    assert hits[0]["section"] == "Table 1"


def test_paper_filter_searches_one_paper_only():
    index = HybridIndex.build(real_paper_chunks(), FakeEmbedder())
    hits = index.search("how was mood measured", k=5, paper_ids=["doi:10.1/other"])
    assert hits and {h["paper_id"] for h in hits} == {"doi:10.1/other"}
    assert index.search("anything", paper_ids=["unknown"]) == []


def test_bm25_matches_word_forms():
    from sr_pipeline.rag.index import tokenize
    assert tokenize("Participants were randomized") == tokenize("participant randomization")
    chunks = [{"chunk_id": "a::0", "paper_id": "a", "text": "Randomization used sealed envelopes"},
              {"chunk_id": "b::0", "paper_id": "b", "text": "Walking improved mood scores"},
              # BM25's idf is 0 for a word in half of a 2-document corpus: use three.
              {"chunk_id": "c::0", "paper_id": "c", "text": "Tai chi classes twice a week"}]
    index = HybridIndex.build(chunks, FakeEmbedder())
    assert [chunks[r]["chunk_id"] for r in index.sparse("how were they randomized")] == ["a::0"]


def test_rrf_rewards_agreement():
    fused = dict(rrf([[1, 2, 3], [3, 4, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 63)
    assert max(fused, key=fused.get) == 1  # ranked by both lists
    assert fused[3] > fused[2]  # 3rd + 1st beats 2nd alone


def test_save_and_load_give_the_same_results(tmp_path):
    index = HybridIndex.build(real_paper_chunks(), FakeEmbedder())
    index.save(tmp_path / "index")
    again = HybridIndex.load(tmp_path / "index", FakeEmbedder())
    q = "sample size calculation effect size"
    assert [h["chunk_id"] for h in again.search(q)] == [h["chunk_id"] for h in index.search(q)]
    other = FakeEmbedder()
    other.model = "text-embedding-3-small"
    with pytest.raises(ValueError, match="index built with fake"):
        HybridIndex.load(tmp_path / "index", other)


# --- re-ranker --------------------------------------------------------------------

def test_rerank_orders_by_llm_score_in_one_call():
    hits = [{"chunk_id": f"c{i}", "paper_id": "p", "text": f"passage {i}"} for i in range(4)]
    llm = ScriptedLLM({"reranker": [{"scores": [{"passage": 1, "score": 2},
                                                {"passage": 2, "score": 9},
                                                {"passage": 3, "score": 9}]}]})
    ranked, _ = rerank(llm, "query?", hits, top=3)
    assert [h["chunk_id"] for h in ranked] == ["c1", "c2", "c0", "c3"]  # ties keep fused order
    assert [h["rerank_score"] for h in ranked] == [9, 9, 2, None]
    assert len(llm.calls) == 1 and "[3] (p) passage 2" in llm.calls[0][1]
    assert "passage 3" not in llm.calls[0][1]  # beyond top: not sent


# --- OpenAI embedder (fake client) ------------------------------------------------

class FakeOpenAI:
    def __init__(self):
        self.requests = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, model, input):
        self.requests.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 1.0, 0.0])
                for i, t in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10 * len(input)))


def test_openai_embedder_caches_and_counts_cost(tmp_path):
    client = FakeOpenAI()
    emb = OpenAIEmbedder("text-embedding-3-small", cache_path=tmp_path / "e.sqlite",
                         client_factory=lambda: client)
    v = emb.embed(["ab", "abcd", "ab"])
    assert v.shape == (3, 3) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert client.requests == [["ab", "abcd"]]  # duplicates embedded once
    usage = emb.usage()
    assert usage["input_tokens"] == 20 and usage["cost_usd"] == pytest.approx(20 * 0.02 / 1e6)

    again = OpenAIEmbedder("text-embedding-3-small", cache_path=tmp_path / "e.sqlite",
                           client_factory=lambda: pytest.fail("cache miss"))
    assert np.allclose(again.embed(["abcd", "ab"]), v[[1, 0]])
    assert again.usage()["api_calls"] == 0 and again.usage()["texts_cached"] == 2


def test_oversized_word_is_split_under_the_embedding_limit():
    # First real run: a chunk of a few huge "words" (PDF text without spaces) exceeded
    # the embedding API's 8,192-token input limit.
    from sr_pipeline.rag.chunking import MAX_CHARS, chunk_paper
    from sr_pipeline.sources.fulltext import FullText, Section
    blob = "x" * 20000
    ft = FullText(paper_id="p1", status="pdf", sections=[Section(title="Results", text="a b " + blob)])
    chunks = chunk_paper({"paper_id": "p1", "title": "T", "abstract": "A"}, ft)
    assert all(len(c["text"]) <= MAX_CHARS for c in chunks)
    assert sum(len(c["text"]) for c in chunks) > 20000   # nothing dropped
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
