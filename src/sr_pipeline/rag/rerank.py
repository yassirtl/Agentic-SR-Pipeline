"""LLM re-ranker: read the best fused passages and score how well each answers the query.

Retrievers are fast but shallow (vector closeness, word counts). The re-ranker
actually reads, but only the ~20 survivors of the hybrid search, in **one**
call: every passage gets a 0-10 relevance score. Passages are then sorted by
that score; ties (and passages the model forgot to score) keep their fused
order. The call goes through the LLM disk cache, so the same query on the same
passages is free the second time.
"""

from pydantic import BaseModel, Field

from ..llm import LLM, Usage

PASSAGE_CHARS = 1500  # ~250 words: the chunk size, so normally nothing is cut


class PassageScore(BaseModel):
    passage: int = Field(description="Passage number as shown in brackets.")
    score: int = Field(ge=0, le=10, description="0 = unrelated, 10 = answers the query directly.")


class Rerank(BaseModel):
    scores: list[PassageScore] = Field(description="One score per passage.")


def build_prompt(query: str, hits: list[dict]) -> str:
    passages = "\n\n".join(f"[{i}] ({h['paper_id']}) {h['text'][:PASSAGE_CHARS]}"
                           for i, h in enumerate(hits, start=1))
    return (
        "You judge passages from scientific papers for a systematic review.\n"
        f"Query: {query}\n\n"
        "Score every passage from 0 to 10 for how directly it answers the query:\n"
        "- 9-10: states the answer explicitly and completely;\n"
        "- 6-8: contains part of the answer, or states it indirectly;\n"
        "- 3-5: on the same topic but does not answer;\n"
        "- 0-2: unrelated.\n"
        "Use the whole scale and compare the passages with each other: give the same "
        "score to two passages only if they answer equally well. Judge only what the "
        "passage says; do not assume missing facts.\n\n"
        f"Passages:\n{passages}"
    )


def rerank(llm: LLM, query: str, hits: list[dict], *, top: int = 20
           ) -> tuple[list[dict], Usage | None]:
    """Re-order the first ``top`` hits by LLM score; returns (hits, usage).

    Each hit gains ``rerank_score`` (None if the model did not score it). Hits
    beyond ``top`` are not read and stay after the re-ranked ones.
    """
    head, tail = hits[:top], hits[top:]
    if not head:
        return hits, None
    result, usage = llm.call_with_usage(Rerank, build_prompt(query, head), agent="reranker")
    score = {s.passage: s.score for s in result.scores if 1 <= s.passage <= len(head)}
    scored = [{**h, "rerank_score": score.get(i)} for i, h in enumerate(head, start=1)]
    order = sorted(range(len(scored)),
                   key=lambda i: (-(scored[i]["rerank_score"] if scored[i]["rerank_score"]
                                    is not None else -1), i))
    return [scored[i] for i in order] + [{**h, "rerank_score": None} for h in tail], usage
