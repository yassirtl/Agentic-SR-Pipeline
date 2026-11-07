"""Retrieval-augmented generation over full texts: chunking, embeddings, hybrid index, re-ranker.

Used from full-text screening (Part 6) and extraction (Part 7): for one paper and
one question (an eligibility criterion, a field to extract), find the passages
that answer it and give only those to the LLM, with their chunk ids as citations.
"""

from .chunking import chunk_paper
from .embed import FakeEmbedder, OpenAIEmbedder
from .index import HybridIndex, rrf
from .rerank import rerank

__all__ = ["FakeEmbedder", "HybridIndex", "OpenAIEmbedder", "chunk_paper", "rerank", "rrf"]
