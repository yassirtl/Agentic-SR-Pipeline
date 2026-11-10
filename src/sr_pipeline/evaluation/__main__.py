"""Evaluation command line (Part 10). No LLM, no network: reads a finished run folder.

  python -m sr_pipeline.evaluation sample   RUN [--n 30] [--seed 10]
      blind labelling sheet: evaluation/labels/<run_id>/labels.csv + criteria.md
      (tracked in git: the labels are the user's work, runs/ is git-ignored)
  python -m sr_pipeline.evaluation evaluate RUN [--topic T3] [--labels FILE] [--metrics a,b]
      runs/<id>/evaluation/evaluation.json + .md, copied to evaluation/results/<run_id>.*
  python -m sr_pipeline.evaluation compare  RUN RUN [...] [--out FILE]
      side-by-side table (e.g. --limit 100 vs a larger limit on one topic)

RUN is a run id (folder under runs/) or a path. The topic defaults to the one in the
run's run.json; its gold list comes from topics.yaml (``*_coverage.csv`` when present).
"""

import argparse
import json
import sys
from pathlib import Path

from ..config import load_config
from .labels import make_sheet, read_labels
from .metrics import load_manual, run_metrics
from .recall import load_gold_rows
from .topics import TOPICS_FILE, load_topic

EVAL_DIR = TOPICS_FILE.parent          # evaluation/ (tracked)


def labels_dir(run_id: str) -> Path:
    return EVAL_DIR / "labels" / run_id


def run_dir(arg: str, runs: Path) -> Path:
    p = Path(arg)
    folder = p if p.is_dir() else runs / arg
    if not (folder / "state.json").exists():
        raise SystemExit(f"{folder}: no state.json (not a finished run?)")
    return folder


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def context(folder: Path, topic_id: str | None = None, labels: Path | None = None,
            manual: dict | None = None) -> dict:
    state = _json(folder / "state.json")
    run = _json(folder / "run.json")
    summary = _json(folder / "report" / "run_summary.json")
    given = topic_id
    topic_id = topic_id or run.get("topic") or (state.get("protocol") or {}).get("topic")
    topic = load_topic(topic_id) if topic_id else None
    question = (state.get("protocol") or {}).get("question")
    if topic and not given and question and question != topic["question"]:
        # Part 10: topic ids were reassigned (old T3 = D0); a run keeps its old id.
        raise SystemExit(f"run question differs from topics.yaml {topic_id} "
                         f"({question!r} vs {topic['question']!r}): pass --topic")
    # A topic without a published review (T2) has no gold list: labels only.
    gold_rows = (load_gold_rows(TOPICS_FILE.parent / topic["gold"])
                 if topic and topic.get("gold") else None)
    if labels is None:
        labels = labels_dir(folder.name) / "labels.csv"
        old = folder / "evaluation" / "labels.csv"   # first sheets were written there
        labels = labels if labels.exists() or not old.exists() else old
    human, problems = read_labels(labels) if labels.exists() else ({}, [])
    return {"folder": folder, "state": state, "run": run, "summary": summary,
            "topic": topic, "gold_rows": gold_rows, "labels": human or None,
            "label_problems": problems, "manual": manual}


# --- Markdown -----------------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(ctx: dict, results: dict) -> str:
    run = ctx["run"].get("run_id") or ctx["folder"].name
    topic = ctx["topic"]
    lines = [f"# Evaluation of run {run}", ""]
    if topic:
        lines += [f"Topic {topic['id']}: {topic['question']}  ",
                  f"Gold standard: {topic.get('review')}", ""]
    c, a, k, t = (results[x] for x in ("completeness", "accuracy", "consistency", "time"))

    if "gold_recall" in c:
        g = c["gold_recall"]
        n = g["counts"]
        lines += ["## Completeness: gold studies, stage by stage", "",
                  "| Stage | Gold studies still in |", "|---|---|"]
        names = {"gold": "gold list", "covered": "in OpenAlex or PubMed (coverage)",
                 "found": "found by the search", "screened": "inside the screening limit"
                 + (f" ({g['screen_limit']})" if g["screen_limit"] else ""),
                 "ta_included": "kept at title/abstract", "included": "kept at full text"}
        lines += [f"| {names[s]} | {n[s]} |" for s in names]
        r = g["recall"]
        lines += ["", f"Recall of the search: {_pct(r['search_vs_all'])} of all gold, "
                      f"{_pct(r['search_vs_covered'])} of covered.  ",
                  f"Final recall: {_pct(r['final_vs_all'])} of all gold, "
                  f"{_pct(r['final_vs_covered'])} of covered.  ",
                  f"Screening alone (gold studies the screeners saw): "
                  f"{_pct(r['screening_ta'])} kept at title/abstract, "
                  f"{_pct(r['screening_ta_ft'])} after full text.", "",
                  "| Gold study | Reached | Lost because | Matched by | Best rank |",
                  "|---|---|---|---|---|"]
        for s in g["studies"]:
            lines.append(f"| {s['first_author']} {s['year']} | {s['reached']} | "
                         f"{s['reason'] or '(included)'} | {s['matched_by'] or ''} | "
                         f"{s['rank'] or ''} |")
        lines.append("")
    if "extraction_completeness" in c:
        e = c["extraction_completeness"]
        parts = ", ".join(f"{k} {_pct(v['share'])} of {v['fields']}" for k, v in
                          e["by_text"].items())
        if e.get("scope") == "topic fields":
            a = e["all_fields"]
            lines += [f"Extraction: {e['reported']}/{e['fields']} topic fields "
                      f"({len(e['topic_fields'])} per study: the topic's own extraction "
                      f"fields only) reported with a verified quote "
                      f"({_pct(e['share_reported'])}) over {e['studies']} studies; {parts}. "
                      f"All fields, incl. the generic clinical ones: {a['reported']}/"
                      f"{a['fields']} ({_pct(a['share'])}).", ""]
        else:
            lines += [f"Extraction: {e['reported']}/{e['fields']} fields reported with a "
                      f"verified quote ({_pct(e['share_reported'])}) over {e['studies']} "
                      f"studies; {parts}.", ""]
    if "prisma_consistency" in c:
        p = c["prisma_consistency"]
        lines += [f"PRISMA arithmetic checks: {'all pass' if p['consistent'] else 'FAIL'} "
                  f"({', '.join(f'{k}: {v}' for k, v in p['checks'].items())}).", ""]
    if "critic" in c:
        q = c["critic"]
        lines += [f"Critic: {'passed' if q['passed'] else 'not passed'}, {q['blockers']} "
                  f"blockers, {q['warnings']} warnings, {q['loops_used']} loops.", ""]

    lines += ["## Accuracy", ""]
    if "gold_precision" in a:
        p = a["gold_precision"]
        lines += [f"Gold share of our includes: {p['gold_included']}/{p['included']} "
                  f"({_pct(p['precision_lower_bound'])}), a lower bound on precision.", ""]
    if "human_labels" in a:
        h = a["human_labels"]
        cf = h["confusion"]
        lines += [f"Title/abstract screening vs {h['labelled']} human-labelled records "
                  f"({h['human_includes']} human includes): accuracy {_pct(h['accuracy'])}, "
                  f"sensitivity {_pct(h['sensitivity'])}, specificity {_pct(h['specificity'])}, "
                  f"kappa {h['kappa']} (tp {cf['tp']}, fn {cf['fn']}, fp {cf['fp']}, "
                  f"tn {cf['tn']}).", ""]
        for d in h["disagreements"]:
            lines.append(f"- human {d['human']}, pipeline {d['pipeline']}"
                         + (f" on {d['criterion']} ({d['label']})" if d["criterion"] else "")
                         + f": {d['title']}")
        lines.append("")
    else:
        lines += ["No human labels yet (`python -m sr_pipeline.evaluation sample RUN`).", ""]

    lines += ["## Inter-agent consistency", ""]
    for stage, v in (k.get("screener_agreement") or {}).items():
        lines.append(f"- {stage.upper()}: screener A vs B kappa {v['kappa']}, agreement "
                     f"{_pct(v['agreement'])} over {v['pairs']} pairs; adjudicated "
                     f"{_pct(v['adjudicated_share'])}; resolved by {v['resolved_by']}")
    if "coordinator" in k:
        co = k["coordinator"]
        lines.append(f"- Coordinator: {co['steps']} steps, decisions {co['decisions']}, "
                     f"budget exhausted: {co['budget_exhausted']}")
        for r in co.get("refusals") or []:
            lines.append(f"  - step {r['step']}: proposed {r['proposed']} → "
                         f"{r['substituted']} ({r['reason']})")
    lines.append("")

    if "time_cost" in t:
        tc = t["time_cost"]
        lines += ["## Time and cost", "",
                  f"Run: {tc['hours']} h wall clock ({tc['seconds_note']}), "
                  f"{tc['llm_calls']} LLM calls ({tc['llm_calls_cached']} from cache), "
                  f"${tc['cost_usd']} (${tc['cost_per_screened_record']} per screened record).",
                  ""]
        m = tc.get("manual")
        if isinstance(m, dict):
            per = ", ".join(f"{k} {v}" for k, v in m["days_per_task"].items())
            lines += [f"Manual estimate for the same workload ({m['source']}): "
                      f"{m['person_days']} person-days ({per}); screening, extraction "
                      f"and risk of bias only (no protocol, search, synthesis or writing).",
                      ""]
        elif m:
            lines += [m, ""]
        if tc.get("benchmark"):
            lines += [f"Benchmark: {tc['benchmark']}", ""]
    return "\n".join(lines)


def compare(folders: list[Path]) -> str:
    rows = []
    for f in folders:
        ctx = context(f, manual=None)
        res = run_metrics(ctx, ["gold_recall", "gold_precision", "time_cost"])
        g = res["completeness"].get("gold_recall") or {}
        tc = res["time"].get("time_cost") or {}
        n = g.get("counts") or {}
        rows.append([f.name, str(ctx["run"].get("screen_limit")),
                     str((tc.get("counts") or {}).get("screened", "")),
                     str((tc.get("counts") or {}).get("included", "")),
                     " → ".join(str(n.get(s, "")) for s in
                                ("gold", "covered", "found", "screened", "ta_included",
                                 "included")),
                     _pct((g.get("recall") or {}).get("final_vs_covered")),
                     str(tc.get("hours", "")), str(tc.get("cost_usd", ""))])
    head = ["Run", "Limit", "Screened", "Included",
            "Gold: all → covered → found → screened → TA → FT", "Final recall (covered)",
            "Hours", "Cost $"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sr_pipeline.evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write a blind labelling sheet")
    s.add_argument("run")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=10)
    s.add_argument("--out", type=Path, help="folder (default evaluation/labels/<run_id>)")
    e = sub.add_parser("evaluate", help="compute the metrics of one run")
    e.add_argument("run")
    e.add_argument("--topic")
    e.add_argument("--labels", type=Path)
    e.add_argument("--metrics", help="comma-separated names (default: topic's, else all)")
    e.add_argument("--manual", type=Path, help="manual effort file "
                   "(default evaluation/manual_effort.yaml)")
    e.add_argument("--results", type=Path, help="tracked copy folder "
                   "(default evaluation/results)")
    c = sub.add_parser("compare", help="side-by-side table of several runs")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    runs = load_config().runs_dir

    if args.cmd == "sample":
        folder = run_dir(args.run, runs)
        state = _json(folder / "state.json")
        out = make_sheet(state, args.out or labels_dir(folder.name), args.n, args.seed)
        print(f"{out['records']} records (seed {out['seed']}) → {out['sheet']}\n"
              f"criteria → {out['criteria']}\nWrite include/exclude in the 'human' column, "
              f"then: python -m sr_pipeline.evaluation evaluate {args.run}")
        return 0
    if args.cmd == "compare":
        table = compare([run_dir(r, runs) for r in args.runs])
        if args.out:
            args.out.write_text(table)
        print(table)
        return 0

    folder = run_dir(args.run, runs)
    ctx = context(folder, args.topic, args.labels, load_manual(args.manual))
    names = args.metrics.split(",") if args.metrics else \
        (ctx["topic"] or {}).get("metrics")
    results = run_metrics(ctx, names)
    out = folder / "evaluation"
    out.mkdir(exist_ok=True)
    payload = {"run_id": folder.name, "topic": (ctx["topic"] or {}).get("id"),
               "label_problems": ctx["label_problems"], **results}
    (out / "evaluation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    md = render(ctx, results)
    (out / "evaluation.md").write_text(md)
    for p in ctx["label_problems"]:
        print(f"labels: {p}", file=sys.stderr)
    results_dir = args.results or EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("json", "md"):
        (results_dir / f"{folder.name}.{ext}").write_text(
            (out / f"evaluation.{ext}").read_text())
    print(md)
    print(f"→ {out / 'evaluation.json'}, {out / 'evaluation.md'}; "
          f"copied to {results_dir}/{folder.name}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
