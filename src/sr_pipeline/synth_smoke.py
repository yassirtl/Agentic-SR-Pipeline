"""Live smoke run of Part 8: synthesis + critic (with its loop) on an extraction run.

    .venv/bin/python -m sr_pipeline.synth_smoke runs/<extract_smoke run>

Starts from an ``extract_smoke`` run (extractions.jsonl, risk_of_bias.jsonl; its audit
names the screening run, which holds protocol.json and prisma_counts.json):
1. paper records (authors, years) rebuilt by replaying retrieval from the HTTP cache (free);
2. the protocol is re-seeded from its topic in evaluation/topics.yaml (e.g. T3), which now
   carries ``outcome_directions``;
3. synthesis, then critic; while the critic says ``loop_back``, synthesis and critic run
   again (what the Coordinator does in a full run; at most 2 loops).

Writes runs/<new id>/: synthesis.json, synthesis.md, qa_report.json, figures/, audit.jsonl.
Cost (gpt-4o-mini): a few tenths of a cent per round.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .agents.critic import make_critic
from .agents.retrieval import make_retrieval
from .agents.synthesis import make_synthesis
from .audit import AuditWriter, new_run_id
from .config import load_config
from .evaluation.topics import load_topic, seed_protocol
from .guardrail import STAGES
from .screen_smoke import merge


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("extract_run", type=Path, help="runs/<id> of an extract_smoke run")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set (put it in .env at the project root).", file=sys.stderr)
        return 1
    run = args.extract_run
    start = json.loads((run / "audit.jsonl").open().readline())
    screen_run = cfg.project_root / start["screen_run"]
    search_run = cfg.project_root / start["search_run"]
    protocol = json.loads((screen_run / "protocol.json").read_text())
    seeded = {}
    if protocol.get("topic"):
        protocol, seeded = seed_protocol(protocol, load_topic(protocol["topic"]))
    prisma = json.loads((screen_run / "prisma_counts.json").read_text())
    ext = [json.loads(line) for line in (run / "extractions.jsonl").open()]
    rob = [json.loads(line) for line in (run / "risk_of_bias.jsonl").open()]
    strategy = json.loads((search_run / "search_strategy.json").read_text())

    run_id = new_run_id()
    audit = AuditWriter(cfg.runs_dir, run_id)
    out = audit.run_dir
    t0 = time.monotonic()
    state = {"question": protocol.get("question", ""), "protocol": protocol,
             "search_strategy": strategy, "completed_stages": ["planner", "search_strategy"],
             "audit_log": [{"node": "synth_smoke", "event": "start", "extract_run": str(run),
                            "screen_run": str(screen_run), "seeded": seeded}]}
    state = merge(state, make_retrieval(cfg.make_http(), max_records=cfg.max_records,
                                        contact_email=cfg.contact_email)(state))
    state |= {"extractions": ext, "risk_of_bias": rob,
              "included": [e["paper_id"] for e in ext], "prisma_counts": prisma,
              "completed_stages": list(STAGES[:STAGES.index("synthesis")])}

    llm = cfg.make_llm()
    synthesis = make_synthesis(llm, out_dir=out / "figures")
    critic = make_critic(llm)
    while True:
        state = merge(state, synthesis(state))
        state = merge(state, critic(state))
        qa = state["qa_report"]
        print(f"Round {state['synthesis']['round']}: blockers {qa['blockers']}, warnings "
              f"{qa['warnings']}, loop_back {qa['loop_back']}")
        if not qa["loop_back"]:
            break
    seconds = time.monotonic() - t0

    s = state["synthesis"]
    (out / "synthesis.json").write_text(json.dumps(s, indent=2, default=str))
    (out / "synthesis.md").write_text(s["text"] + "\n\n## Evidence tables\n\n```\n"
                                      + s["evidence"] + "\n```\n")
    (out / "qa_report.json").write_text(json.dumps(qa, indent=2))
    audit.sync(state["audit_log"])

    cost = sum(e.get("cost_usd") or 0 for e in state["audit_log"])
    print(f"\nRun {run_id}  ({seconds:.0f} s; API ${cost:.4f})")
    for g in s["outcomes"]:
        print(f"{g['id']} {g['outcome']} (better: {g['better']}): {g['method']}; "
              f"{g['reason']}")
        print(f"   votes {g['vote_count']}")
        if g["meta"]:
            m = g["meta"]
            print(f"   SMD {m['pooled']:.2f} ({m['ci_low']:.2f} to {m['ci_high']:.2f}), "
                  f"I2 {100 * m['i2']:.0f}%, tau2 {m['tau2']:.3f}; plot {g['plot']}")
    print(f"Narrative: {len(s['narrative'])} sentences; checks {s['checks'] or 'clean'}")
    print(f"QA: passed {qa['passed']}, loops used {qa['loops_used']}; issues:")
    for i in qa["issues"]:
        print(f"  [{i['severity']}] {i['kind']}: {i['detail'][:150]}")
    print(f"\n{s['text']}\n\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
