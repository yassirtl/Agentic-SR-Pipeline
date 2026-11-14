"""Hybrid passage index: dense (FAISS over embeddings) + sparse (BM25), fused by rank.

Two ways to find passages, with opposite strengths:

- **Dense** (meaning): the query and every chunk are embeddings; the closest
  vectors win (cosine similarity, exact search with FAISS ``IndexFlatIP``).
  Finds paraphrases ("elderly" for "older adults"); weak on exact names and
  numbers ("GDS-15", "HAM-D 17").
- **Sparse** (keywords): BM25 scores chunks by the query words they contain,
  rare words weighing more, long chunks penalized. Exact and transparent;
  blind to synonyms. Words are stemmed (Snowball English: "randomized",
  "randomization" -> "random"), otherwise a query and a passage using two
  forms of a word never match (seen live: BM25 put the "Randomization" section
  35th for "How were participants randomized?").

**Reciprocal rank fusion** (RRF) merges the two ranked lists using ranks only:
``score(chunk) = sum over lists of 1 / (k + rank)``, k = 60. The raw scores
(cosine vs BM25) are on scales that cannot be compared, ranks can. A chunk
ranked well by both methods comes first; one found by a single method still
gets in.

``paper_ids`` restricts a search to some papers (full-text screening asks one
question about one paper at a time): FAISS through an ID selector, BM25 by
scoring those papers' chunks only.

Saved per run as ``chunks.jsonl`` + ``faiss.index`` + ``meta.json``; BM25 is rebuilt on
load (fast, and its pickle would tie the files to a library version).
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
import threading

import snowballstemmer
from rank_bm25 import BM25Okapi

from .embed import Embedder

RRF_K = 60
POOL = 50  # candidates taken from each method before fusion
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that the
their there these this to was were what which with without""".split())


# A snowball stemmer is not thread-safe (seen in Part 12: parallel extraction threads
# compressing passages crashed it), so each thread gets its own.
_LOCAL = threading.local()


def _stemmer():
    if not hasattr(_LOCAL, "stemmer"):
        _LOCAL.stemmer = snowballstemmer.stemmer("english")
    return _LOCAL.stemmer


def tokenize(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
             if w not in STOPWORDS]
    return _stemmer().stemWords(words)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of ranked lists of item ids -> [(id, score)] best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridIndex:
    def __init__(self, chunks: list[dict], index: faiss.Index, embedder: Embedder):
        self.chunks = chunks
        self.faiss = index
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
        self.rows_of: dict[str, list[int]] = {}
        for row, c in enumerate(chunks):
            self.rows_of.setdefault(c["paper_id"], []).append(row)

    # --- build / save / load -------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[dict], embedder: Embedder) -> "HybridIndex":
        vectors = embedder.embed([c["text"] for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(chunks, index, embedder)

    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "chunks.jsonl").open("w") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
        faiss.write_index(self.faiss, str(folder / "faiss.index"))
        (folder / "meta.json").write_text(json.dumps({
            "embedding_model": self.embedder.model, "dim": self.faiss.d,
            "chunks": len(self.chunks), "papers": len(self.rows_of)}, indent=2))
        return folder

    @classmethod
    def load(cls, folder: str | Path, embedder: Embedder) -> "HybridIndex":
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text())
        if meta["embedding_model"] != embedder.model:
            raise ValueError(f"index built with {meta['embedding_model']}, "
                             f"query embedder is {embedder.model}")
        chunks = [json.loads(line) for line in (folder / "chunks.jsonl").open()]
        return cls(chunks, faiss.read_index(str(folder / "faiss.index")), embedder)

    # --- search -------------------------------------------------------------------
    def _rows(self, paper_ids: list[str] | None) -> list[int] | None:
        if paper_ids is None:
            return None
        return [r for pid in paper_ids for r in self.rows_of.get(pid, [])]

    def dense(self, query: str, pool: int = POOL, rows: list[int] | None = None) -> list[int]:
        if rows is not None and not rows:
            return []
        q = self.embedder.embed([query]).astype("float32")
        params = None
        if rows is not None:
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(np.array(rows, dtype="int64")))
        n = min(pool, self.faiss.ntotal if rows is None else len(rows))
        _, ids = self.faiss.search(q, n, params=params)
        return [int(i) for i in ids[0] if i >= 0]

    def sparse(self, query: str, pool: int = POOL, rows: list[int] | None = None) -> list[int]:
        tokens = tokenize(query)
        if self.bm25 is None or not tokens or (rows is not None and not rows):
            return []
        if rows is None:
            rows = list(range(len(self.chunks)))
            scores = np.asarray(self.bm25.get_scores(tokens))
        else:  # scores aligned with rows
            scores = np.asarray(self.bm25.get_batch_scores(tokens, rows))
        order = np.argsort(-scores, kind="stable")[:pool]
        return [rows[i] for i in order if scores[i] > 0]  # no query word at all: not a match

    def search(self, query: str, k: int = 10, *, paper_ids: list[str] | None = None,
               pool: int = POOL) -> list[dict]:
        """Top ``k`` chunks by RRF, each ``{**chunk, score, dense_rank, sparse_rank}``."""
        rows = self._rows(paper_ids)
        dense, sparse = self.dense(query, pool, rows), self.sparse(query, pool, rows)
        d_rank = {r: i for i, r in enumerate(dense, start=1)}
        s_rank = {r: i for i, r in enumerate(sparse, start=1)}
        return [{**self.chunks[r], "score": round(score, 5),
                 "dense_rank": d_rank.get(r), "sparse_rank": s_rank.get(r)}
                for r, score in rrf([dense, sparse])[:k]]
