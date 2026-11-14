"""Context compression (Part 12): keep only the sentences of a retrieved passage that bear
on the question, before the LLM reads it.

After hybrid retrieval and re-ranking, each passage (about 250 words) still holds sentences
that have nothing to do with the criterion or field asked about. ``compress_hits`` keeps the
``keep`` sentences that share the most distinct words with the query (``index.tokenize``, so
the same words BM25 matches), in their original order, joined by " … ". The kept sentences
are copied verbatim, so a quote the LLM takes from them is still found in the paper.

Off by default (config ``rag_compress``): screening and extraction were evaluated without
it, and their numbers stay valid; a run with it on is a different setting, said in the run's
config. Code only, no LLM call.
"""

from __future__ import annotations

import re

from .index import tokenize

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text or "") if s.strip()]


def compress_text(text: str, query: str, keep: int = 3) -> str:
    """The ``keep`` sentences of ``text`` most related to ``query``, in order."""
    sents = sentences(text)
    if keep <= 0 or len(sents) <= keep:
        return text
    q = set(tokenize(query))
    scored = [(len(q & set(tokenize(s))), -i, i) for i, s in enumerate(sents)]
    best = sorted(i for _, _, i in sorted(scored, reverse=True)[:keep])
    return " … ".join(sents[i] for i in best)


def compress_hits(hits: list[dict], query: str, keep: int = 3) -> list[dict]:
    """Copies of the retrieved passages with their text compressed (``keep`` 0 = as is).
    Each keeps its chunk id, so the passage stays traceable to the paper."""
    if keep <= 0:
        return hits
    out = []
    for h in hits:
        text = compress_text(h.get("text") or "", query, keep)
        out.append({**h, "text": text, "compressed_from": len(h.get("text") or "")}
                   if text != h.get("text") else h)
    return out
