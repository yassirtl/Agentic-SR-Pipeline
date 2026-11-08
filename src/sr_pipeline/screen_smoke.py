"""Live smoke run of Part 6: title/abstract then full-text screening on a search run.

    .venv/bin/python -m sr_pipeline.screen_smoke runs/20261107-160209-aec820 --limit 50
    .venv/bin/python -m sr_pipeline.screen_smoke runs/<search run> --limit 50 \\
        --gold evaluation/gold/T3_coverage.csv
    .venv/bin/python -m sr_pipeline.screen_smoke runs/<search run>          # every record
    .venv/bin/python -m sr_pipeline.screen_smoke runs/20261107-160209-aec820 --topic T3 \\
        --limit 50 --gold evaluation/gold/T3_coverage.csv

``--topic T1|T2|T3|D0`` re-seeds the search run's protocol with the gold review's
question, designs, years, languages and criteria (evaluation/topics.yaml) before
screening. Only the protocol changes: the records are those of the search run, so
the topic's years/languages were NOT applied to retrieval (logged in the audit).

Starts from a ``search_smoke`` run (protocol.json, search_strategy.json):
1. retrieval + deduplication (database answers from the HTTP cache: free);
2. title/abstract screening of the first ``--limit`` unique records (relevance
   order), dual screening + adjudicator, unresolved records auto-included
   (``--auto`` behaviour; the human path is exercised by the tests);
3. full-text screening of the includes: open-access full text, hybrid index in
   runs/<id>/index, passages per criterion, re-ranker, dual screening;
4. ``--gold``: every gold article present among the screened records, with its
   decisions. A gold article missing from the screened records was not found by
   the search (or is beyond ``--limit``): that is recall, measured in Part 10.

Cost (gpt-4o-mini): ~$0.0005 per abstract (2-3 calls); full text ~$0.003 per paper
(re-ranking per criterion + screeners). Re-runs are free (caches).
Writes runs/<new id>/: ta_decisions.jsonl, ft_decisions.jsonl, prisma_counts.json,
gold.json (with --gold), index/, audit.jsonl.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .agents.retrieval import make_retrieval
from .agents.screening import make_conflict_review, make_ft_screening, make_ta_screening
from .audit import AuditWriter, new_run_id
from .config import load_config
from .evaluation.gold import load_gold, match_gold
from .evaluation.topics import load_topic, seed_protocol
from .state import latest

APPEND = {"ta_decisions", "ft_decisions", "completed_stages", "audit_log", "records"}


def merge(state: dict, update: dict) -> dict:
    """Apply a node's update the way LangGraph does (append-only fields concatenated)."""
    return {**state, **{k: state.get(k, []) + v if k in APPEND else v
                        for k, v in update.items()}}


def _dump(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("search_run", type=Path, help="runs/<id> of a search_smoke run")
    ap.add_argument("--limit", type=int, default=None, help="screen only the first N records")
    ap.add_argument("--gold", type=Path, default=None, help="gold CSV (a *_coverage.csv is best)")
    ap.add_argument("--no-full-text", action="store_true", help="stop after title/abstract")
    ap.add_argument("--topic", default=None,
                    help="evaluation topic (T1, T2, T3, D0): screen with the gold review's criteria")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).", file=sys.stderr)
        return 1
    protocol = json.loads((args.search_run / "protocol.json").read_text())
    strategy = json.loads((args.search_run / "search_strategy.json").read_text())
    seeded = None
    if args.topic:
        search_scope = {"years": protocol.get("years"), "languages": protocol.get("languages")}
        protocol, seeded = seed_protocol(protocol, load_topic(args.topic))
        seeded |= {"node": "screen_smoke", "records_from": str(args.search_run),
                   "retrieval_scope": search_scope,
                   "note": "records come from the search run: the topic's years and "
                           "languages were not applied to retrieval"}

    run_id = new_run_id()
    audit = AuditWriter(cfg.runs_dir, run_id)
    out = audit.run_dir
    http = cfg.make_http()
    llm = cfg.make_llm()
    times = {}

    t0 = time.monotonic()
    state = {"question": protocol.get("question", ""), "protocol": protocol,
             "search_strategy": strategy, "completed_stages": ["planner", "search_strategy"],
             "audit_log": [{"node": "screen_smoke", "event": "start",
                            "search_run": str(args.search_run), "limit": args.limit}]
                           + ([seeded] if seeded else [])}
    state = merge(state, make_retrieval(http, max_records=cfg.max_records,
                                        contact_email=cfg.contact_email)(state))
    if args.limit:  # development run: the screened sample is the whole input downstream
        state["unique_records"] = state["unique_records"][:args.limit]
    times["retrieval"] = time.monotonic() - t0

    t0 = time.monotonic()
    state = merge(state, make_ta_screening(llm, workers=cfg.screening_workers)(state))
    state = merge(state, make_conflict_review("ta", human=False)(state))
    times["title/abstract"] = time.monotonic() - t0

    if not args.no_full_text:
        t0 = time.monotonic()
        node = make_ft_screening(llm, http, cfg.make_embedder(), index_dir=out / "index",
                                 workers=cfg.screening_workers, passages=cfg.ft_passages,
                                 candidates=cfg.ft_candidates, chunk_words=cfg.chunk_words,
                                 chunk_overlap=cfg.chunk_overlap, use_rerank=cfg.ft_rerank,
                                 contact_email=cfg.contact_email)
        state = merge(state, node(state))
        state = merge(state, make_conflict_review("ft", human=False)(state))
        times["full text"] = time.monotonic() - t0

    ta = latest(state.get("ta_decisions", []))
    ft = latest(state.get("ft_decisions", []))
    _dump(out / "ta_decisions.jsonl", ta)
    _dump(out / "ft_decisions.jsonl", ft)
    pc = state["prisma_counts"]
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "prisma_counts.json").write_text(json.dumps(pc, indent=2))

    gold_report = None
    if args.gold:
        rows = load_gold(args.gold)
        found = match_gold(rows, state["unique_records"])
        ta_by, ft_by = {d["paper_id"]: d for d in ta}, {d["paper_id"]: d for d in ft}
        gold_report = []
        for r in rows:
            pid = found.get(r["row"])
            gold_report.append({
                "row": r["row"], "study": f"{r['first_author']} {r['year']}", "paper_id": pid,
                "ta": ta_by[pid]["final"] if pid in ta_by else None,
                "ta_criterion": ta_by[pid]["criterion"] if pid in ta_by else None,
                "ft": ft_by[pid]["final"] if pid in ft_by else None,
                "ft_criterion": ft_by[pid]["criterion"] if pid in ft_by else None,
                "fulltext": ft_by[pid]["fulltext_status"] if pid in ft_by else None})
        (out / "gold.json").write_text(json.dumps(gold_report, indent=2))
    audit.sync(state["audit_log"])

    # Report ------------------------------------------------------------------------
    cost = sum(e.get("cost_usd") or 0 for e in state["audit_log"])
    print(f"Run {run_id}  (" + ", ".join(f"{k} {v:.0f} s" for k, v in times.items())
          + f"; API ${cost:.4f})")
    print(f"Criteria from: {protocol.get('criteria_source', 'Planner (LLM draft)')}")
    t = pc["title_abstract"]
    print(f"\nTitle/abstract: {t['records_screened']} screened, {t['included']} included, "
          f"{t['records_excluded']} excluded  (no abstract: {t['no_abstract']})")
    print(f"  A/B agreement {t['agreement']}, kappa {t['kappa']}; resolved by {t['resolved_by']}")
    print(f"  excluded by criterion {t['excluded_by_criterion']}; quotes not found "
          f"{t['quotes_not_found']}; LLM errors {t['llm_errors']}")
    if "full_text" in pc:
        f = pc["full_text"]
        print(f"\nFull text: {f['reports_sought']} sought, {f['reports_not_retrieved']} not "
              f"retrieved (screened on abstract), {f['reports_assessed']} assessed")
        for reason, n in Counter(f["not_retrieved_reasons"]).most_common(5):
            print(f"  {n:>3} x {reason}")
        print(f"  {f['studies_included']} included ({f['included_on_abstract_only']} on "
              f"abstract only), {f['reports_excluded']} excluded {f['excluded_by_criterion']}")
        print(f"  A/B agreement {f['agreement']}, kappa {f['kappa']}; resolved by "
              f"{f['resolved_by']}; quotes not found {f['quotes_not_found']}")
    if gold_report is not None:
        present = [g for g in gold_report if g["paper_id"]]
        print(f"\nGold ({args.gold.name}): {len(present)}/{len(gold_report)} among the "
              f"screened records")
        for g in gold_report:
            if g["paper_id"]:
                print(f"  row {g['row']:>2} {g['study']:<24} TA {g['ta']} ({g['ta_criterion']})"
                      f"  FT {g['ft']} ({g['ft_criterion']}, {g['fulltext']})")
    print(f"\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
