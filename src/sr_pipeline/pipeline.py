"""The full review graph with the real agents (Parts 3-9), built from the config.

One place wires every stage, so the CLI and the end-to-end test run the same graph. The
network-facing parts are injectable (the test passes fakes; a real run passes nothing):
- ``http``: database + full-text HTTP client (``None`` = search strategy without hit
  counts, one LLM call);
- ``search``: same signature as ``sources.search_all`` (retrieval);
- ``fetch``: same signature as ``sources.fulltext.fetch_fulltexts`` (full-text screening);
- ``embedder``: OpenAI embeddings with the disk cache, or ``rag.FakeEmbedder``.
"""

from pathlib import Path

from .agents.critic import make_critic
from .agents.extraction import make_extraction
from .agents.planner import make_planner
from .agents.retrieval import make_retrieval
from .agents.screening import make_ft_screening, make_ta_screening
from .agents.search_strategy import make_search_strategy
from .agents.synthesis import make_synthesis
from .config import Config
from .graph import build_graph
from .llm import LLM
from .protocol import load_template
from .reporting import make_report
from .sources import search_all
from .sources.fulltext import fetch_fulltexts

AGENTS = ("coordinator", "planner", "search_strategy", "screener_a", "screener_b",
          "adjudicator", "reranker", "extraction", "risk_of_bias", "synthesis", "critic")


def agent_models(cfg: Config) -> dict[str, str]:
    return {a: cfg.model_for(a) for a in AGENTS}


def build_stages(cfg: Config, llm: LLM, run_dir: str | Path, *, topic: dict | None = None,
                 http=None, embedder=None, search=search_all, fetch=fetch_fulltexts,
                 env=None) -> dict:
    run_dir = Path(run_dir)
    index = run_dir / "index"
    common = {"contact_email": cfg.contact_email, "env": env}
    return {
        "planner": make_planner(llm, load_template(cfg.protocol_template), topic=topic),
        "search_strategy": make_search_strategy(
            llm, http, max_records=cfg.max_records, min_hits=cfg.min_hits,
            max_refinements=cfg.max_refinements,
            broaden_below=cfg.search_broaden_below(), **common),
        "retrieval": make_retrieval(http, max_records=cfg.max_records, search=search,
                                    **common),
        "ta_screening": make_ta_screening(llm, workers=cfg.screening_workers,
                                          limit=cfg.screen_limit or None),
        "ft_screening": make_ft_screening(
            llm, http, embedder, index_dir=index, workers=cfg.screening_workers,
            passages=cfg.ft_passages, candidates=cfg.ft_candidates,
            chunk_words=cfg.chunk_words, chunk_overlap=cfg.chunk_overlap,
            use_rerank=cfg.ft_rerank, fetch=fetch, **common),
        "extraction": make_extraction(llm, embedder, index_dir=index,
                                      workers=cfg.screening_workers,
                                      passages=cfg.ft_passages, candidates=cfg.ft_candidates,
                                      use_rerank=cfg.ft_rerank),
        "synthesis": make_synthesis(llm, out_dir=run_dir / "figures"),
        "critic": make_critic(llm),
        "report": make_report(run_dir / "report", models=agent_models(cfg),
                              embedder=embedder),
    }


def build_review_graph(cfg: Config, llm: LLM, run_dir: str | Path, *, auto: bool,
                       checkpointer=None, **inject):
    """Compiled graph. ``auto=False``: pauses for protocol approval and unresolved
    screening conflicts (LangGraph ``interrupt``); ``auto=True`` never pauses."""
    return build_graph(llm, max_steps=cfg.max_steps,
                       stages=build_stages(cfg, llm, run_dir, **inject),
                       approve_protocol=not auto, human_screening=not auto,
                       checkpointer=checkpointer)
