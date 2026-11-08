"""Text -> vector ("embedding"), for meaning-based (dense) search.

An embedding model maps a passage to a point in a ~1,500-dimension space where
passages that *mean* the same thing are close, even without shared words
("older adults" ~ "elderly participants aged 65+"). Vectors are L2-normalized,
so the inner product of two vectors is their cosine similarity.

OpenAI ``text-embedding-3-small`` through the API (no torch on this Intel Mac,
see docs/PLAN.md), $0.02 per 1M tokens. Every vector is cached in one SQLite
file keyed on sha256(model, text): building the index again, or for another
run with the same papers, costs nothing.

``FakeEmbedder`` (tests) hashes words into a fixed-size vector: no network,
deterministic, and texts sharing words are close, which is enough to test the
index logic.
"""

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from ..llm import compute_cost

BATCH_TEXTS = 256         # inputs per API request (limit 2,048)
BATCH_WORDS = 120_000     # ~160k tokens per request (limit 300k tokens)


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)



MAX_EMBED_CHARS = 8000

class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class _Usage:
    """Running totals for the audit log."""

    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.api_calls = 0
        self.cached = 0
        self.embedded = 0

    def usage(self) -> dict:
        return {"model": self.model, "input_tokens": self.input_tokens,
                "api_calls": self.api_calls, "texts_embedded": self.embedded,
                "texts_cached": self.cached,
                "cost_usd": compute_cost(self.model, self.input_tokens, 0)}


def _default_client():
    from openai import OpenAI  # reads OPENAI_API_KEY from the environment

    return OpenAI()


class OpenAIEmbedder(_Usage):
    def __init__(self, model: str = "text-embedding-3-small", *, cache_path: str | Path,
                 client_factory: Callable[[], Any] = _default_client):
        super().__init__(model)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.cache_path)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self._client_factory = client_factory
        self._client = None

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()

    def _lookup(self, keys: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for start in range(0, len(keys), 500):  # SQLite variable limit
            part = keys[start:start + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part)
            out.update({k: np.frombuffer(v, dtype="float32") for k, v in rows})
        return out

    def _batches(self, texts: list[str]):
        batch, words = [], 0
        for t in texts:
            n = len(t.split())
            if batch and (len(batch) >= BATCH_TEXTS or words + n > BATCH_WORDS):
                yield batch
                batch, words = [], 0
            batch.append(t)
            words += n
        if batch:
            yield batch

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        # Last guard against the API's 8,192-token input limit (chunks are already
        # capped at 6,000 characters by the chunker; other callers may not be).
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        keys = [self._key(t) for t in texts]
        found = self._lookup(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        self.cached += len(texts) - len(missing)
        for batch in self._batches(missing):
            if self._client is None:
                self._client = self._client_factory()
            resp = self._client.embeddings.create(model=self.model, input=batch)
            self.api_calls += 1
            self.input_tokens += resp.usage.prompt_tokens
            rows = []
            for text, item in zip(batch, sorted(resp.data, key=lambda d: d.index)):
                vec = np.asarray(item.embedding, dtype="float32")
                found[self._key(text)] = vec
                rows.append((self._key(text), vec.tobytes()))
            with self._db:  # one transaction per batch: a crash never leaves half a batch
                self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.embedded += len(batch)
        return normalize(np.stack([found[k] for k in keys]))


class FakeEmbedder(_Usage):
    """Bag of hashed words (tests only): deterministic, no network, shared words -> close."""

    def __init__(self, dim: int = 256):
        super().__init__("fake")
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return normalize(out)
