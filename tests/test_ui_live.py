"""Part 11b, UI side: a fake-LLM run goes start -> protocol approval -> screening conflicts ->
report from the Streamlit page (AppTest, no browser, no network); the budget cap pauses and is
raised from the page; the live view reads audit.jsonl alone; a stopped run resumes from the
Runs page. The worker thread is real; tests wait for it with ``live.join``."""

import json
from datetime import datetime, timezone

import pytest

from sr_pipeline.guardrail import STAGES
from sr_pipeline.ui import data as D
from sr_pipeline.ui import live as L

from test_live import priced_llm
from test_report import INJECT, QUESTION, check_outputs, e2e_config, e2e_llm

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def ui(tmp_path):
    cfg = e2e_config(tmp_path)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    L.CONFIG, L.INJECT = cfg, {**INJECT, "llm": e2e_llm()}
    yield cfg
    for rid in list(L._WORKERS):
        L.join(rid, 30)
    L.CONFIG, L.INJECT = None, {}
    L._WORKERS.clear()


def page(cfg, fn="page_live", **session):
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              f"P.RUNS_DIR = Path({str(cfg.runs_dir)!r})\nP.setup()\nP.{fn}()\n")
    at = AppTest.from_string(script, default_timeout=60)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def wait(at, cfg):
    rid = at.session_state["live_run"]
    L.join(rid, 60)
    assert not L.is_alive(rid)
    return rid, at.run()


def start(at, *, budget=0.0, auto=False):
    at.radio(key="new_mode").set_value("Free question").run()
    at.text_area(key="new_question").input(QUESTION).run()
    at.number_input(key="new_limit").set_value(100)
    at.number_input(key="new_budget").set_value(budget)
    at.toggle(key="new_auto").set_value(auto)
    return at.button(key="start_btn").click().run()


def submit(at):
    btn = next(b for b in at.button if "FormSubmitter" in str(b.proto.id) or
               b.proto.is_form_submitter)
    return btn.click().run()


def test_start_approve_resolve_report_from_the_ui(ui):
    at = start(page(ui))
    assert not at.exception
    rid, at = wait(at, ui)
    assert L.read_status(ui.runs_dir / rid)["status"] == "waiting"
    assert any("Approve the protocol" in s.value for s in at.subheader)
    # Edit one PICO field and accept the suggested design in the form.
    at.text_input[1].set_value("adults aged 60 or older")       # population
    at.checkbox(key="accept_designs").check()
    at = submit(at)
    rid, at = wait(at, ui)
    assert any("Resolve screening conflicts" in s.value for s in at.subheader)
    radios = [r for r in at.radio if r.key and r.key.startswith("conf_")]
    assert len(radios) == 1
    radios[0].set_value("exclude")
    at = submit(at)
    rid, at = wait(at, ui)
    assert not at.exception
    assert L.read_status(ui.runs_dir / rid)["status"] == "finished"
    assert any("Run finished" in s.value for s in at.success)

    state = json.loads((ui.runs_dir / rid / "state.json").read_text())
    assert state["completed_stages"] == list(STAGES)
    p = state["protocol"]
    assert p["study_designs"] == ["randomized controlled trial"]
    assert p["pico"]["population"] == "adults aged 60 or older"
    approval = next(e for e in state["audit_log"] if e["node"] == "protocol_approval")
    # Only the field the human changed is sent (+ the designs from the accepted suggestion).
    assert approval["edits"] == {"pico": {"population": "adults aged 60 or older"},
                                 "study_designs": ["randomized controlled trial"]}
    conflict = next(e for e in state["audit_log"] if e["node"] == "ta_conflicts")
    assert conflict["resolved_by"] == "human" and list(conflict["decisions"].values()) == ["exclude"]
    check_outputs(ui, rid, state)

    # The live view of the finished run matches the state: every card done, time logged.
    v = D.live_view(ui.runs_dir / rid, alive=False)
    assert v.status == "finished" and v.current is None
    assert [c.status for c in v.cards] == ["done"] * len(STAGES)
    assert all(c.seconds is not None for c in v.cards)
    assert v.counters == {"unique": 5, "screened": 5, "ta_included": 3, "ft_included": 3}
    meta = json.loads((ui.runs_dir / rid / "run.json").read_text())
    assert meta["started_from"] == "ui" and meta["budget_usd"] is None


def test_budget_cap_pauses_and_is_raised_from_the_ui(ui):
    L.INJECT["llm"] = priced_llm()                                # $0.001 per call
    at = start(page(ui), budget=0.004, auto=True)
    rid, at = wait(at, ui)
    status = L.read_status(ui.runs_dir / rid)
    assert status["status"] == "waiting" and status["pending"]["kind"] == "budget_cap"
    v = D.live_view(ui.runs_dir / rid, alive=False)
    assert v.cap_usd == 0.004 and v.spent_usd >= 0.004
    assert "report" not in [c.stage for c in v.cards if c.status == "done"]
    assert any("Budget cap reached" in s.value for s in at.subheader)
    at.number_input(key="new_cap").set_value(1.0)
    at = submit(at)
    rid, at = wait(at, ui)
    assert L.read_status(ui.runs_dir / rid)["status"] == "finished"
    v = D.live_view(ui.runs_dir / rid, alive=False)
    assert v.cap_usd == 1.0 and v.spent_usd > 0.004


def test_stopped_run_resumes_from_the_runs_page(ui):
    """A worker that dies mid-run (here: the LLM fails once in extraction) leaves an error
    status; the Runs page lists it under Continue and Resume finishes it."""
    llm = e2e_llm()
    real = llm.script["extraction"]
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SystemExit("network gone")    # not caught by the stages: the run stops
        return real(prompt)

    llm.script["extraction"] = flaky
    L.INJECT["llm"] = llm
    cfg = ui.model_copy(update={"screening_workers": 1})
    L.CONFIG = cfg
    rid = L.start(question=QUESTION, topic=None, auto=True, limit=100, budget_usd=None)
    L.join(rid, 60)
    status = L.read_status(cfg.runs_dir / rid)
    assert status["status"] == "error" and "network gone" in status["error"]
    assert D.resumable(cfg.runs_dir / rid)
    with pytest.raises(RuntimeError):     # never two workers on one run
        L._WORKERS[rid] = type("T", (), {"is_alive": lambda self: True})()
        L.resume(rid)
    L._WORKERS.pop(rid)

    at = page(cfg, "page_runs")
    assert not at.exception
    assert at.selectbox(key="resume_pick").value == rid
    assert at.button(key="resume_btn").disabled                # needs the confirmation
    assert at.number_input(key="resume_cap").value > 0           # uncapped run: a cap offered
    at.number_input(key="resume_cap").set_value(5.0)
    at.checkbox(key="resume_confirm").check().run()
    at = at.button(key="resume_btn").click().run()
    L.join(rid, 60)
    assert L.read_status(cfg.runs_dir / rid)["status"] == "finished"
    state = json.loads((cfg.runs_dir / rid / "state.json").read_text())
    assert state["completed_stages"] == list(STAGES)
    assert state["run"]["budget_usd"] == 5.0
    assert any(e.get("event") == "budget cap set at resume" for e in state["audit_log"])


def test_live_view_while_a_stage_is_running(tmp_path):
    """Mid-run: the stage the coordinator just chose and that has not written yet is
    'running'; a live.json 'running' with no worker means the app stopped: 'stopped'."""
    run = tmp_path / "20261110-120000-abcdef"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"question": "q", "screen_limit": 100,
                                              "budget_usd": 1.0, "started_at":
                                              "2026-11-10T12:00:00+00:00",
                                              "estimate": {"low": 0.1, "mid": 0.2, "high": 0.3}}))
    stamp = {"started_at": "2026-11-10T12:00:01+00:00", "ended_at": "2026-11-10T12:00:03+00:00",
             "duration_s": 2.0}
    audit = [
        {"node": "coordinator", "step": 1, "proposed": "planner", "outcome": "accepted",
         "cost_usd": 0.01, **stamp},
        {"node": "planner", "event": "protocol drafted", "cost_usd": 0.02, "model": "m",
         "input_tokens": 10, "output_tokens": 5, **stamp},
        {"node": "coordinator", "step": 2, "proposed": "retrieval", "outcome": "refused",
         "substituted": "search_strategy", "reason": "prerequisites not fresh", "cost_usd": 0.01,
         **stamp},
    ]
    (run / "audit.jsonl").write_text("\n".join(json.dumps(e) for e in audit))
    v = D.live_view(run, status={"status": "running"}, alive=True,
                    now=datetime(2026, 11, 10, 12, 0, 30, tzinfo=timezone.utc).timestamp())
    assert v.current == "search_strategy"
    cards = {c.stage: c for c in v.cards}
    assert cards["search_strategy"].status == "running"
    assert cards["planner"].status == "done" and cards["planner"].seconds == 2.0
    assert cards["retrieval"].refused == 1
    assert v.spent_usd == pytest.approx(0.04) and v.cap_usd == 1.0 and v.steps == 2
    assert v.elapsed_s == pytest.approx(30, abs=1)
    assert D.live_view(run, status={"status": "running"}, alive=False).status == "stopped"


def test_live_page_renders_a_waiting_run_without_a_worker(ui):
    """The pause payload lives in live.json: the form shows after an app restart too."""
    rid = "20261110-120000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "screen_limit": 100}))
    L.write_status(run, "waiting", pending={
        "kind": "screening_conflicts", "stage": "title/abstract", "error": None,
        "papers": [{"paper_id": "p1", "title": "<b>T</b>", "screener_a": {"decision": "include"},
                    "screener_b": {"decision": "exclude", "criterion": "E1", "quote": "q"},
                    "adjudicator": {"decision": "include", "criterion": "all", "reason": "unsure",
                                    "unresolved": True}, "flags": []}]})
    at = page(ui, live_run=rid)
    assert not at.exception
    assert [r.key for r in at.radio if r.key and r.key.startswith("conf_")] == ["conf_p1"]
    html = "".join(m.value for m in at.markdown)
    assert "could not decide" in html and "Adjudicator: **include" not in html


def seeded_protocols():
    from sr_pipeline.evaluation.topics import load_topic, seed_protocol
    from sr_pipeline.protocol import Protocol, ProtocolDraft
    from test_report import DRAFT

    base = Protocol(question=QUESTION, databases=DRAFT["databases"],
                    **{k: v for k, v in DRAFT.items()
                       if k in ProtocolDraft.model_fields and k != "databases"}
                    ).model_dump(mode="json")
    out = {"free question": base}
    for t in ("T1", "T2", "T3", "D0"):
        out[t] = seed_protocol(dict(base), load_topic(t))[0]
    odd = json.loads(json.dumps(base))       # what a form would silently "change"
    odd["pico"]["comparator"] = None
    odd["inclusion_criteria"][0] += "  \n continued on a second line "
    odd["databases"] = list(reversed(odd["databases"]))
    out["edge cases"] = odd
    return out


@pytest.mark.parametrize("name", ["free question", "T1", "T2", "T3", "D0", "edge cases"])
def test_untouched_approval_form_sends_true(ui, monkeypatch, name):
    protocol = seeded_protocols()[name]
    rid = "20261110-120000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "screen_limit": 100}))
    L.write_status(run, "waiting", pending={
        "kind": "protocol_approval", "protocol": protocol, "error": None,
        "suggested_designs": protocol.get("suggested_designs", []),
        "design_reason": protocol.get("design_reason", ""),
        "criteria_source": protocol.get("criteria_source")})
    sent = {}
    monkeypatch.setattr(L, "resume", lambda run_id, **kw: sent.update(kw) or run_id)
    at = page(ui, live_run=rid)
    assert not at.exception
    submit(at)
    assert sent["kind"] == "protocol_approval" and sent["answer"] is True


def test_running_run_renders_through_the_refresh_fragment(ui):
    """A run whose worker is alive renders inside the st.fragment(run_every=2) that polls the
    audit; the stage the coordinator chose is highlighted as running."""
    rid = "20261110-120000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "screen_limit": 100,
                                              "started_at": "2026-11-10T12:00:00+00:00"}))
    (run / "audit.jsonl").write_text(json.dumps(
        {"node": "coordinator", "step": 1, "proposed": "planner", "outcome": "accepted"}))
    L.write_status(run, "running")
    L._WORKERS[rid] = type("Alive", (), {"is_alive": lambda self: True})()
    at = page(ui, live_run=rid)
    assert not at.exception
    html = "".join(m.value for m in at.markdown)
    assert 'class="stage running"' in html and "now: <b>Protocol</b>" in html
    assert at.get("progress")                                    # spend bar
    L._WORKERS.pop(rid)
