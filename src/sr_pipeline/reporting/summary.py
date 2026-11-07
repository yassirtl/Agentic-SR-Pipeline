"""Run summary: time, steps, LLM calls, tokens and cost, read from the audit log.

Stage entries carry summed usage (``llm_calls``, ``llm_calls_cached``, ``input_tokens``,
``output_tokens``, ``cost_usd``); coordinator and planner entries carry one call
(``model``, ``cached``, tokens, cost). A cost of None means a model missing from the price
table: the total is then a lower bound and says so. Cached calls cost $0 and are counted
apart: they are what makes a re-run reproducible (same prompt -> same answer from disk).

Embeddings are not in the stage entries (the embedder keeps its own counter): the report
stage passes ``embedder.usage()`` in.
"""

from collections import defaultdict
from datetime import datetime, timezone


def _calls(entry: dict) -> tuple[int, int]:
    if "llm_calls" in entry:
        return entry["llm_calls"] or 0, entry.get("llm_calls_cached") or 0
    if "model" in entry and "input_tokens" in entry:     # one call (Usage.as_dict)
        return 1, int(bool(entry.get("cached")))
    return 0, 0


def run_summary(state: dict, *, finished_at: datetime | None = None,
                embeddings: dict | None = None) -> dict:
    audit = state.get("audit_log", [])
    run = state.get("run") or {}
    finished = finished_at or datetime.now(timezone.utc)
    started = run.get("started_at")
    seconds = None
    if started:
        seconds = round((finished - datetime.fromisoformat(started)).total_seconds(), 1)

    per_node: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cached": 0,
                                                     "input_tokens": 0, "output_tokens": 0,
                                                     "cost_usd": 0.0, "unpriced": 0})
    for e in audit:
        calls, cached = _calls(e)
        if not calls:
            continue
        n = per_node[e.get("node", "?")]
        n["calls"] += calls
        n["cached"] += cached
        n["input_tokens"] += e.get("input_tokens") or 0
        n["output_tokens"] += e.get("output_tokens") or 0
        if e.get("cost_usd") is None:
            n["unpriced"] += 1
        else:
            n["cost_usd"] += e["cost_usd"]
    totals = {k: sum(n[k] for n in per_node.values())
              for k in ("calls", "cached", "input_tokens", "output_tokens", "cost_usd",
                        "unpriced")}
    if embeddings:
        totals["cost_usd"] += embeddings.get("cost_usd") or 0
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    for n in per_node.values():
        n["cost_usd"] = round(n["cost_usd"], 6)

    coord = [e for e in audit if e.get("node") == "coordinator"]
    outcomes = defaultdict(int)
    for e in coord:
        if "outcome" in e:
            outcomes[e["outcome"]] += 1
    return {
        "run_id": run.get("run_id"),
        "question": state.get("question") or (state.get("protocol") or {}).get("question"),
        "started_at": started,
        "finished_at": finished.isoformat(timespec="seconds"),
        "seconds": seconds,
        "steps_taken": state.get("steps_taken", 0),
        "stages_run": list(state.get("completed_stages", [])),
        "coordinator": {"decisions": dict(outcomes),
                        "refusals": [{"step": e.get("step"), "proposed": e.get("proposed"),
                                      "reason": e.get("reason") or e.get("error"),
                                      "substituted": e.get("substituted")}
                                     for e in coord if e.get("outcome") in ("refused",
                                                                            "llm_error")]},
        "budget_exhausted": any(e.get("event") == "budget exhausted" for e in audit),
        "llm": {"per_node": dict(per_node), **totals,
                "cost_is_lower_bound": totals["unpriced"] > 0},
        "embeddings": embeddings or {},
    }


def summary_markdown(s: dict) -> str:
    llm = s["llm"]
    secs = s["seconds"]
    when = f"{secs / 60:.1f} min" if secs is not None else "unknown"
    lines = [
        f"- Run `{s['run_id']}`, {when} wall-clock (includes time waiting for a human at "
        "the approval steps).",
        f"- Coordinator steps: {s['steps_taken']}; decisions "
        + (", ".join(f"{k} {v}" for k, v in sorted(s["coordinator"]["decisions"].items()))
           or "none") + ".",
        f"- LLM calls: {llm['calls']} ({llm['cached']} from the cache), "
        f"{llm['input_tokens']:,} input and {llm['output_tokens']:,} output tokens.",
        f"- API cost: ${llm['cost_usd']:.4f}"
        + (" (lower bound: some models have no price in the table)"
           if llm["cost_is_lower_bound"] else "")
        + (f", of which embeddings ${s['embeddings'].get('cost_usd') or 0:.4f}"
           if s["embeddings"] else "") + ".",
        "",
        "| Agent / node | Calls | Cached | Input tokens | Output tokens | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for node, n in sorted(llm["per_node"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        lines.append(f"| {node} | {n['calls']} | {n['cached']} | {n['input_tokens']:,} | "
                     f"{n['output_tokens']:,} | {n['cost_usd']:.4f} |")
    if s["coordinator"]["refusals"]:
        lines += ["", "Coordinator proposals refused by the guardrail (deterministic "
                  "substitute run instead):", ""]
        lines += [f"- step {r['step']}: proposed `{r['proposed']}`, ran "
                  f"`{r['substituted']}` ({r['reason']})" for r in s["coordinator"]["refusals"]]
    return "\n".join(lines)
