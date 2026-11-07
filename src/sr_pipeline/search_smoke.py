"""Live smoke run: question -> protocol -> search strategy -> records from every database.

    .venv/bin/python -m sr_pipeline.search_smoke "Does exercise reduce depression in older adults?"
    .venv/bin/python -m sr_pipeline.search_smoke "..." --max-records 50 --refresh
    .venv/bin/python -m sr_pipeline.search_smoke "..." --check-word depress

Two to four LLM calls (planner + search strategy with up to 2 refinements,
< $0.01) and a few free database requests (hit counts per attempt, then the
records). ``--check-word`` prints, per database, the share of downloaded
records whose title or abstract contains one of the words: a rough precision
measure to compare search strategies. The protocol is not paused for approval (like --auto). Writes to
runs/<run_id>/: protocol.json, search_strategy.json, search_results.json,
records.jsonl, audit.jsonl. Re-running the same question is free and identical
(LLM and HTTP caches) unless --refresh is given. The full CLI comes in Part 9.
"""

import argparse
import json
import os
import sys
import time

from .agents.planner import make_planner
from .agents.search_strategy import make_search_strategy
from .audit import AuditWriter, new_run_id
from .config import load_config
from .protocol import load_template
from .sources import search_all


def mention_count(records, words: list[str]) -> int:
    """Records whose title or abstract contains one of ``words`` (case-insensitive)."""
    words = [w.lower() for w in words]
    return sum(any(w in f"{r.title} {r.abstract or ''}".lower() for w in words)
               for r in records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question")
    ap.add_argument("--max-records", type=int, default=None, help="per database (default: config)")
    ap.add_argument("--refresh", action="store_true", help="re-query databases, ignore HTTP cache")
    ap.add_argument("--check-word", action="append", default=[],
                    help="report the share of records mentioning this word (repeatable)")
    args = ap.parse_args(argv)

    cfg = load_config()  # also loads .env
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).", file=sys.stderr)
        return 1
    max_records = args.max_records or cfg.max_records
    llm = cfg.make_llm()
    run_id = new_run_id()
    audit = AuditWriter(cfg.runs_dir, run_id)
    t0 = time.monotonic()

    http = cfg.make_http(refresh=args.refresh)
    state: dict = {"question": args.question, "completed_stages": [], "audit_log": []}
    for node in (make_planner(llm, load_template(cfg.protocol_template)),
                 make_search_strategy(llm, http, max_records=max_records,
                                      broad_factor=cfg.broad_factor, min_hits=cfg.min_hits,
                                      max_refinements=cfg.max_refinements,
                                      contact_email=cfg.contact_email)):
        update = node(state)
        for k, v in update.items():  # same merge rule as the graph's reducers
            state[k] = state.get(k, []) + v if k in ("completed_stages", "audit_log") else v

    results, errors = search_all(state["search_strategy"], http, max_records=max_records,
                                 contact_email=cfg.contact_email)
    state["audit_log"].append({"node": "search_smoke", "event": "databases searched",
                               "http_requests": http.network_calls, "errors": errors,
                               **{f"{db}_hits": r.total_hits for db, r in results.items()},
                               **{f"{db}_downloaded": len(r.records) for db, r in results.items()}})

    out = audit.run_dir
    (out / "protocol.json").write_text(json.dumps(state["protocol"], indent=2))
    (out / "search_strategy.json").write_text(json.dumps(state["search_strategy"], indent=2))
    (out / "search_results.json").write_text(json.dumps(
        {db: r.model_dump(exclude={"records"}) | {"downloaded": len(r.records)}
         for db, r in results.items()} | {"errors": errors}, indent=2))
    with (out / "records.jsonl").open("w") as f:
        for r in results.values():
            for rec in r.records:
                f.write(rec.model_dump_json() + "\n")
    audit.sync(state["audit_log"])

    cost = sum(e.get("cost_usd") or 0 for e in state["audit_log"])
    print(f"Run {run_id}  ({time.monotonic() - t0:.0f} s, LLM ${cost:.4f}, "
          f"{http.network_calls} HTTP requests)")
    protocol, strategy = state["protocol"], state["search_strategy"]
    print(f"Protocol: {protocol['title']}")
    print(f"Outcomes: {', '.join(protocol['pico']['outcomes'])} | "
          f"comparator: {protocol['pico']['comparator']}")
    print(f"Databases: {', '.join(protocol['databases'])} ({protocol['database_reason']})")
    t = strategy["targets"]
    print(f"\nSearch attempts (target {t['min_hits']}-{t['max_hits']} hits per database):")
    for a in strategy["attempts"]:
        blocks = " AND ".join(c.get("role", c["name"]) for c in a["concepts"])
        counts = ", ".join(f"{db} {n}" for db, n in a["counts"].items())
        mark = "  <- used" if a["attempt"] == strategy["chosen_attempt"] else ""
        notes = (f" | same as attempt {a['same_as']}" if a.get("same_as") else "") + \
            (f" | terms in several blocks: {', '.join(a['overlap'])}" if a.get("overlap") else "")
        print(f"  {a['attempt']}. {blocks}: {counts or '-'}{mark}\n     {a['reason']}{notes}")
    for db, query in state["search_strategy"]["queries"].items():
        print(f"\n[{db}] {query}")
        if db in results:
            r = results[db]
            print(f"  -> {r.total_hits} hits, {len(r.records)} downloaded"
                  + (" (capped by max_records)" if r.capped else ""))
            for rec in r.records[:3]:
                print(f"     - {rec.year} {rec.title[:90]}")
            if args.check_word and r.records:
                n = mention_count(r.records, args.check_word)
                print(f"     {n}/{len(r.records)} records ({100 * n / len(r.records):.0f}%) "
                      f"mention {' / '.join(args.check_word)}")
        else:
            print(f"  -> ERROR {errors.get(db)}")
    print(f"\nOutputs in {out}")
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
