"""Live smoke run of Part 7: extraction + risk of bias on a screening run's includes.

    .venv/bin/python -m sr_pipeline.extract_smoke runs/20261107-205453-5eb2ed

Starts from a ``screen_smoke`` run (protocol.json, ft_decisions.jsonl, index/):
1. the paper records are rebuilt by replaying retrieval on that run's search run
   (database answers from the HTTP cache: free);
2. ``included`` = papers whose latest full-text decision is include; their full-text
   status and the arm pair judged at screening come from ft_decisions.jsonl;
3. extraction + risk of bias over the saved index (passages per field group / domain,
   re-ranked, then one extraction call and one risk-of-bias call per study).

Cost (gpt-4o-mini): ~$0.002-0.003 per study (re-ranking + two calls). Re-runs are free.
Writes runs/<new id>/: extractions.jsonl, risk_of_bias.jsonl, audit.jsonl.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .agents.extraction import make_extraction
from .agents.retrieval import make_retrieval
from .audit import AuditWriter, new_run_id
from .config import load_config
from .guardrail import STAGES
from .screen_smoke import _dump, merge


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("screen_run", type=Path, help="runs/<id> of a screen_smoke run")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).", file=sys.stderr)
        return 1
    run = args.screen_run
    start = json.loads((run / "audit.jsonl").open().readline())
    search_run = cfg.project_root / start["search_run"]
    protocol = json.loads((run / "protocol.json").read_text())
    strategy = json.loads((search_run / "search_strategy.json").read_text())
    ft = [json.loads(line) for line in (run / "ft_decisions.jsonl").open()]
    included = [d["paper_id"] for d in ft if d["final"] == "include"]

    run_id = new_run_id()
    audit = AuditWriter(cfg.runs_dir, run_id)
    out = audit.run_dir
    http = cfg.make_http()
    t0 = time.monotonic()
    state = {"question": protocol.get("question", ""), "protocol": protocol,
             "search_strategy": strategy, "completed_stages": ["planner", "search_strategy"],
             "audit_log": [{"node": "extract_smoke", "event": "start",
                            "screen_run": str(run), "search_run": str(search_run),
                            "included": len(included)}]}
    state = merge(state, make_retrieval(http, max_records=cfg.max_records,
                                        contact_email=cfg.contact_email)(state))
    state |= {"ft_decisions": ft, "included": included,
              "completed_stages": list(STAGES[:STAGES.index("extraction")])}
    missing = set(included) - {p["paper_id"] for p in state["unique_records"]}
    if missing:
        print(f"warning: {len(missing)} included papers not found by the replayed retrieval",
              file=sys.stderr)

    node = make_extraction(cfg.make_llm(), cfg.make_embedder(), index_dir=run / "index",
                           workers=cfg.screening_workers, passages=cfg.ft_passages,
                           candidates=cfg.ft_candidates, use_rerank=cfg.ft_rerank)
    state = merge(state, node(state))
    seconds = time.monotonic() - t0
    _dump(out / "extractions.jsonl", state["extractions"])
    _dump(out / "risk_of_bias.jsonl", state["risk_of_bias"])
    audit.sync(state["audit_log"])

    # Report ------------------------------------------------------------------------
    s = state["audit_log"][-1]
    cost = sum(e.get("cost_usd") or 0 for e in state["audit_log"])
    print(f"Run {run_id}  ({seconds:.0f} s; API ${cost:.4f}; "
          f"{s['llm_calls']} LLM calls, {s['llm_calls_cached']} cached)")
    print(f"Studies {s['studies']}; fields reported {s['fields_reported']}/"
          f"{s['fields_requested']} (by text: {s['reported_by_text']}); outcome results "
          f"{s['results_reported']}")
    print(f"Risk of bias: tools {s['tools']} chosen by {s['tool_chosen_by']}; "
          f"overall {s['overall']}")
    print(f"Flags {s['flags']}")
    rob = {r["paper_id"]: r for r in state["risk_of_bias"]}
    print()
    for e in state["extractions"]:
        f = e["fields"]
        r = rob[e["paper_id"]]
        res = next((x for x in e["results"] if x["status"] == "reported"), None)
        arms = ("; ".join(f"{a['arm']} n={a['n']} {a['mean']}({a['sd']})" for a in res["arms"])
                if res else "-")
        print(f"- {(e['title'] or '')[:60]}  [{e['fulltext_status']}]")
        print(f"    design: {f['design']['value'][:50]} | n: {f['n_enrolled']['value'][:40]} | "
              f"measure: {f.get('O1_measure', {}).get('value', '-')[:30]}")
        print(f"    reported {e['reported']}/{e['requested']}; O1: {arms[:90]}")
        print(f"    {r['tool']} ({r['tool_chosen_by']}): "
              + " ".join(f"{d['domain']}={d['judgment']}" for d in r["domains"])
              + f" -> {r['overall']}")
    print(f"\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
