"""Live smoke run of Part 5: retrieval + deduplication, open-access full text, hybrid RAG index.

    .venv/bin/python -m sr_pipeline.rag_smoke runs/20261107-160209-aec820
    .venv/bin/python -m sr_pipeline.rag_smoke runs/<search run> --papers 150
    .venv/bin/python -m sr_pipeline.rag_smoke runs/<search run> --query "How was depression measured?"

Starts from a ``search_smoke`` run (its protocol.json and search_strategy.json):
1. runs the retrieval stage (database answers come from the HTTP cache, so this
   is free and identical to the search run unless --refresh): records,
   duplicates by rule, PRISMA identification counts;
2. full text for the first ``--papers`` unique papers in relevance order (a
   development sample: from Part 6 on, the papers are those that pass
   title/abstract screening). Taking them in order, not only open-access ones,
   gives a realistic availability rate;
3. chunks, embeds (OpenAI, cached), builds and saves the hybrid index, then
   answers the test queries: hybrid search + LLM re-ranking of the top passages.

Cost: embeddings ~$0.02 per 1M tokens (~150 papers: ~$0.02-0.05), one re-rank
call per query (< $0.002 each); re-runs are free (caches). Writes to
runs/<new run_id>/: unique_records.jsonl, dedup.json, prisma_counts.json,
fulltext.jsonl, index/ (chunks.jsonl, faiss.index, meta.json), queries.json,
audit.jsonl.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .agents.retrieval import make_retrieval
from .audit import AuditWriter, new_run_id
from .config import load_config
from .rag import HybridIndex, chunk_paper, rerank
from .sources.fulltext import fetch_fulltexts

DEFAULT_QUERIES = [
    "How were participants randomized and allocated to groups?",
    "Which scale measured depressive symptoms as the primary outcome?",
    "What was the age of participants (older adults, 60 or 65 years and over)?",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("search_run", type=Path, help="runs/<id> of a search_smoke run")
    ap.add_argument("--papers", type=int, default=20, help="papers to fetch and index")
    ap.add_argument("--query", action="append", default=[], help="test query (repeatable)")
    ap.add_argument("--refresh", action="store_true", help="re-query databases and full texts")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).", file=sys.stderr)
        return 1
    src = args.search_run
    protocol = json.loads((src / "protocol.json").read_text())
    strategy = json.loads((src / "search_strategy.json").read_text())
    queries = args.query or DEFAULT_QUERIES

    run_id = new_run_id()
    audit = AuditWriter(cfg.runs_dir, run_id)
    out = audit.run_dir
    http = cfg.make_http(refresh=args.refresh)
    log: list[dict] = [{"node": "rag_smoke", "event": "start", "search_run": str(src),
                        "papers": args.papers}]

    # 1. Retrieval + deduplication ------------------------------------------------
    t0 = time.monotonic()
    state = {"question": protocol.get("question", ""), "protocol": protocol,
             "search_strategy": strategy, "completed_stages": ["planner", "search_strategy"]}
    update = make_retrieval(http, max_records=cfg.max_records,
                            contact_email=cfg.contact_email)(state)
    log += update["audit_log"]
    papers = update["unique_records"]
    ident = update["prisma_counts"]["identification"]
    with (out / "unique_records.jsonl").open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    (out / "dedup.json").write_text(json.dumps(update["retrieval"], indent=2))
    (out / "prisma_counts.json").write_text(json.dumps(update["prisma_counts"], indent=2))
    t_retrieval = time.monotonic() - t0

    # 2. Full text ---------------------------------------------------------------------
    t0 = time.monotonic()
    sample = papers[:args.papers]
    calls_before = http.network_calls
    fulltexts = fetch_fulltexts(sample, http, contact_email=cfg.contact_email)
    with (out / "fulltext.jsonl").open("w") as f:
        for ft in fulltexts.values():
            f.write(json.dumps({"paper_id": ft.paper_id, "status": ft.status,
                                "source_url": ft.source_url, "reason": ft.reason,
                                "sections": len(ft.sections), "words": ft.words}) + "\n")
    status = Counter(ft.status for ft in fulltexts.values())
    reasons = Counter(ft.reason for ft in fulltexts.values() if ft.reason)
    log.append({"node": "rag_smoke", "event": "full texts fetched", "papers": len(sample),
                "status": dict(status), "reasons": dict(reasons),
                "http_requests": http.network_calls - calls_before})
    t_fulltext = time.monotonic() - t0

    # 3. Chunks, embeddings, index ---------------------------------------------------------
    t0 = time.monotonic()
    chunks = [c for p in sample for c in chunk_paper(p, fulltexts[p["paper_id"]],
                                                       size=cfg.chunk_words,
                                                       overlap=cfg.chunk_overlap)]
    embedder = cfg.make_embedder()
    index = HybridIndex.build(chunks, embedder)
    index.save(out / "index")
    log.append({"node": "rag_smoke", "event": "index built", "chunks": len(chunks),
                **embedder.usage()})
    t_index = time.monotonic() - t0

    # 4. Test queries: hybrid search + LLM re-rank ----------------------------------------
    llm = cfg.make_llm()
    answers = []
    for q in queries:
        hits = index.search(q, k=cfg.rerank_top)
        ranked, usage = rerank(llm, q, hits, top=cfg.rerank_top)
        answers.append({"query": q, "hits": [{k: h.get(k) for k in (
            "chunk_id", "section", "score", "dense_rank", "sparse_rank", "rerank_score", "text")}
            for h in ranked[:5]]})
        log.append({"node": "rag_smoke", "event": "query", "query": q,
                    **(usage.as_dict() if usage else {})})
    (out / "queries.json").write_text(json.dumps(answers, indent=2))
    audit.sync(log)

    # Report ---------------------------------------------------------------------------
    cost = sum(e.get("cost_usd") or 0 for e in log)
    print(f"Run {run_id}  (retrieval {t_retrieval:.0f} s, full text {t_fulltext:.0f} s, "
          f"index {t_index:.0f} s; API ${cost:.4f})")
    print("\nIdentification (PRISMA):")
    for db, d in ident["databases"].items():
        print(f"  {db:<9} {d['identified']:>7} identified {d['downloaded']:>6} downloaded "
              f"{d['not_retrieved']:>6} not retrieved" + ("  (capped)" if d["capped"] else ""))
    for db, err in ident["errors"].items():
        print(f"  {db:<9} ERROR {err}")
    by_rule = ", ".join(f"{r} {n}" for r, n in ident["duplicates_by_rule"].items() if n)
    print(f"  downloaded {ident['downloaded']} - duplicates {ident['duplicates_removed']} "
          f"({by_rule}) = {ident['records_screened']} unique records to screen")
    print(f"  pairs kept apart (title match, guard): {len(update['retrieval']['conflicts'])}")

    print(f"\nFull text, first {len(sample)} papers: "
          + ", ".join(f"{s} {n}" for s, n in status.most_common()))
    for reason, n in reasons.most_common(6):
        print(f"  {n:>4} x {reason}")
    usage = embedder.usage()
    print(f"\nIndex: {len(chunks)} chunks from {len(sample)} papers; embeddings: "
          f"{usage['texts_embedded']} new, {usage['texts_cached']} cached, "
          f"{usage['input_tokens']} tokens, ${usage['cost_usd'] or 0:.4f}")
    for a in answers:
        print(f"\nQ: {a['query']}")
        for h in a["hits"][:3]:
            print(f"  [{h['rerank_score']}] {h['chunk_id']}  {h['section'][:50]}  "
                  f"(dense #{h['dense_rank']}, BM25 #{h['sparse_rank']})")
            print(f"      {h['text'][:160]}")
    print(f"\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
