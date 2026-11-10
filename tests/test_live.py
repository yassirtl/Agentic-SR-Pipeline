"""Part 11b, pipeline side: each node's audit entries carry its start / end time and are on
disk as soon as the node ends; the dollar budget cap pauses a run at a stage boundary and the
run resumes when the cap is raised. Scripted LLM, fake search / full text / embedder."""

import json

import pytest

from sr_pipeline import cli
from sr_pipeline.coordinator import budget_stop, parse_cap, spent_usd
from sr_pipeline.graph import build_graph, run_config, timed
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import ScriptedLLM, Usage

from test_report import INJECT, QUESTION, Answers, check_outputs, e2e_config, e2e_llm

CALL_COST = 0.001


class PricedLLM(ScriptedLLM):
    """Scripted answers, but every call is billed CALL_COST (the fake LLM costs $0)."""

    def call_with_usage(self, schema, prompt, *, agent):
        parsed, _ = super().call_with_usage(schema, prompt, agent=agent)
        return parsed, Usage("gpt-4o-mini", 100, 10, cost_usd=CALL_COST)


def priced_llm():
    return PricedLLM(e2e_llm().script)


def audit_lines(cfg, run_id):
    return [json.loads(x) for x in (cfg.runs_dir / run_id / "audit.jsonl").read_text()
            .splitlines()]


# --- Time stamps -----------------------------------------------------------------------------

class Meter:
    def __init__(self):
        self.cost = 0.0

    def usage(self):
        return {"cost_usd": self.cost}


def test_timed_stamps_entries_and_embedding_spend():
    meter = Meter()

    def stage(state):
        meter.cost += 0.002
        return {"audit_log": [{"node": "x", "event": "ran"}, {"node": "x", "event": "more"}]}

    out = timed(stage, meter)({})
    for e in out["audit_log"]:
        assert e["started_at"] <= e["ended_at"] and e["duration_s"] >= 0
        assert e["embed_cost_usd"] == pytest.approx(0.002)
    assert out["audit_log"][1]["event"] == "more"
    assert timed(lambda s: {})({}) == {}                 # no entry: nothing to stamp
    kept = timed(lambda s: {"audit_log": [{"started_at": "own"}]})({})
    assert kept["audit_log"][0]["started_at"] == "own"   # a node's own field wins


def test_placeholder_graph_entries_are_timed():
    def coord(prompt):
        import re
        done = re.search(r"^Stages run \(in order\): (.*)$", prompt, re.M).group(1).split(", ")
        return {"next": next((s for s in STAGES if s not in done), "report"), "rationale": "r"}

    graph = build_graph(ScriptedLLM({"coordinator": coord}))
    state = graph.invoke({"question": "q"}, run_config("t"))
    assert state["completed_stages"] == list(STAGES)
    assert all("started_at" in e and "ended_at" in e for e in state["audit_log"])


def test_audit_is_on_disk_while_the_run_is_paused(tmp_path):
    """At the protocol pause the planner's entry is already in audit.jsonl, with its time."""
    cfg = e2e_config(tmp_path)
    seen = {}

    def answer(payload):
        run_id = next(p.name for p in cfg.runs_dir.iterdir())
        seen["nodes"] = [e["node"] for e in audit_lines(cfg, run_id)]
        raise cli.AwaitingHuman(payload)

    with pytest.raises(cli.AwaitingHuman) as exc:
        cli.run(cfg, question=QUESTION, topic_id=None, auto=False, answer=answer,
                out=lambda *a: None, llm=e2e_llm(), **INJECT)
    assert exc.value.payload["kind"] == "protocol_approval"
    assert seen["nodes"] == ["coordinator", "planner"]


# --- Budget cap ------------------------------------------------------------------------------

def test_spent_and_cap_rules():
    log = [{"cost_usd": 0.01}, {"cost_usd": None}, {"embed_cost_usd": 0.005}, {}]
    assert spent_usd(log) == pytest.approx(0.015)
    assert budget_stop({"audit_log": log}) is None                          # no cap
    assert budget_stop({"audit_log": log, "run": {"budget_usd": 0.02}}) is None
    stop = budget_stop({"audit_log": log, "run": {"budget_usd": 0.015}})
    assert stop["event"] == "stopped: budget cap" and stop["spent_usd"] == pytest.approx(0.015)
    assert parse_cap(0.5, 0.015) == 0.5 and parse_cap({"budget_usd": 1}, 0) == 1.0
    assert parse_cap(None, 0.2) is None and parse_cap({"budget_usd": None}, 0.2) is None
    for bad in (0.01, True, "1", {"cap": 1}):
        with pytest.raises(ValueError):
            parse_cap(bad, 0.015)


def test_budget_cap_stops_cleanly_then_resumes_to_the_report(tmp_path):
    cfg = e2e_config(tmp_path)
    llm = priced_llm()
    # --auto never pauses for a human, but the cap still pauses it. Enter = stop here.
    with pytest.raises(KeyboardInterrupt):
        cli.run(cfg, question=QUESTION, topic_id=None, auto=True, budget_usd=0.004,
                ask=Answers([""]), out=lambda *a: None, llm=llm, **INJECT)
    run_id = next(p.name for p in cfg.runs_dir.iterdir())
    audit = audit_lines(cfg, run_id)
    stop = [e for e in audit if e.get("event") == "stopped: budget cap"]
    assert len(stop) == 1 and stop[0]["spent_usd"] >= 0.004
    # Stopped at a stage boundary: the last entry is the coordinator's stop, every stage that
    # started also ended, and the spend can overshoot by at most the stage that was running.
    assert audit[-1]["event"] == "stopped: budget cap"
    assert "report" not in [e["node"] for e in audit]
    calls_at_stop = len(llm.calls)

    # Resume without raising the cap: a cap not above the spend is refused, asked again;
    # Enter stops again, still cleanly, and no LLM call was made meanwhile.
    ask = Answers(["0.001", ""])
    with pytest.raises(KeyboardInterrupt):
        cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id, ask=ask,
                out=lambda *a: None, llm=llm, **INJECT)
    assert len(ask.prompts) == 2 and len(llm.calls) == calls_at_stop

    # Resume and raise it: the run goes on from the stage boundary to the report.
    said = []
    _, state = cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id,
                       ask=Answers(["1.0"]), out=said.append, llm=llm, **INJECT)
    assert state["completed_stages"] == list(STAGES)
    assert state["run"]["budget_usd"] == 1.0
    raised = [e for e in state["audit_log"] if e.get("event") == "budget cap raised"]
    assert len(raised) == 1 and raised[0]["previous_cap_usd"] == 0.004
    assert raised[0]["attempts"] == 2                     # 0.001 refused first
    check_outputs(cfg, run_id, state)
    # No stage ran twice: the pause sat between two stages, so the capped run made exactly
    # the calls of an uncapped one.
    base = priced_llm()
    cli.run(e2e_config(tmp_path / "base"), question=QUESTION, topic_id=None, auto=True,
            ask=Answers([]), out=lambda *a: None, llm=base, **INJECT)
    assert sorted(llm.calls) == sorted(base.calls)


def test_budget_cap_never_reached_and_none_removes_it(tmp_path):
    cfg = e2e_config(tmp_path)
    _, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True, budget_usd=5.0,
                       ask=Answers([]), out=lambda *a: None, llm=priced_llm(), **INJECT)
    assert state["completed_stages"][-1] == "report"
    assert not [e for e in state["audit_log"] if "budget cap" in str(e.get("event"))]
    meta = json.loads((cfg.runs_dir / state["run"]["run_id"] / "run.json").read_text())
    assert meta["budget_usd"] == 5.0

    cfg2 = e2e_config(tmp_path / "b")
    _, state = cli.run(cfg2, question=QUESTION, topic_id=None, auto=True, budget_usd=0.002,
                       ask=Answers(["none"]), out=lambda *a: None, llm=priced_llm(), **INJECT)
    assert state["completed_stages"][-1] == "report" and state["run"]["budget_usd"] is None
    assert [e["event"] for e in state["audit_log"] if "budget cap" in str(e.get("event"))] \
        == ["stopped: budget cap", "budget cap removed"]


def test_cli_budget_argument_errors(capsys):
    for argv in (["q", "--budget", "0"], ["--resume", "x", "--limit", "1"],
                 ["--report-only", "x", "--budget", "1"]):
        with pytest.raises(SystemExit):
            cli.main(argv)


def test_cap_set_when_resuming_a_stopped_uncapped_run(tmp_path):
    """A run started without a cap (e.g. before Part 11b) stops mid-run; resuming it with a
    cap stores the cap in the checkpoint and run.json, and the coordinator then pauses at
    the next stage boundary."""
    from test_report import extractor
    cfg = e2e_config(tmp_path).model_copy(update={"screening_workers": 1})
    llm = priced_llm()
    left = {"n": 1}

    def flaky(prompt):
        if left["n"]:
            left["n"] -= 1
            raise KeyboardInterrupt
        return extractor(prompt)

    llm.script["extraction"] = flaky
    with pytest.raises(KeyboardInterrupt):
        cli.run(cfg, question=QUESTION, topic_id=None, auto=True, ask=Answers([]),
                out=lambda *a: None, llm=llm, **INJECT)
    run_id = next(p.name for p in cfg.runs_dir.iterdir())
    with pytest.raises(KeyboardInterrupt):   # Enter at the budget pause = stop there
        cli.run(cfg, question=None, topic_id=None, auto=False, resume=run_id, budget_usd=0.001,
                ask=Answers([""]), out=lambda *a: None, llm=llm, **INJECT)
    audit = audit_lines(cfg, run_id)
    events = [e.get("event") for e in audit]
    i = events.index("budget cap set at resume")
    assert audit[i]["previous_cap_usd"] is None and audit[i]["budget_usd"] == 0.001
    # The interrupted stage (extraction) finished first, then the cap paused the run.
    assert events[i + 1] == "extraction and risk of bias" and events[-1] == "stopped: budget cap"
    assert json.loads((cfg.runs_dir / run_id / "run.json").read_text())["budget_usd"] == 0.001
