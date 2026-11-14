"""Part 11e: guided flow. The demo goes from step 1 to "All done" by its main buttons and
writes nothing tracked; a fake-LLM real run goes start -> plan -> disagreement -> results ->
check through the same screens; the replay and step logic are tested directly."""

import json
from pathlib import Path

import pytest

from sr_pipeline.evaluation import stage_checks as SC
from sr_pipeline.guardrail import STAGES
from sr_pipeline.ui import flow as F
from sr_pipeline.ui import live as L

from test_report import INJECT, QUESTION, e2e_config, e2e_llm

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

DEMO = F.demo_run_dir()


def _audit(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "audit.jsonl").read_text().splitlines() if x.strip()]


# --- Logic --------------------------------------------------------------------------------

def test_demo_snapshot_is_tracked_and_complete():
    assert DEMO is not None, "docs/demo snapshot missing"
    demo = json.loads((DEMO / "demo.json").read_text())
    r = demo["results"]
    assert r["included"] == len(r["kept"]) == 55
    assert (r["found"], r["unique"], r["screened"]) == (618, 555, 100)
    assert demo["practice"]["a"]["decision"] != demo["practice"]["b"]["decision"]
    assert demo["protocol"]["inclusion_criteria"]
    assert (DEMO / "report" / "prisma_flow.png").exists()


def test_replay_never_goes_back_and_runs_the_next_stage():
    audit = _audit(DEMO)
    frames = F.replay_frames(audit)
    assert len(frames) == len(STAGES) + 1 and frames[-1] == (len(audit), None)
    done_before = -1
    for cut, running in frames:
        done = F.done_stages(audit[:cut])
        assert len(done) > done_before or cut == frames[0][0]
        done_before = len(done)
        if running:      # the stage shown running is the next one not done yet
            assert running not in done
            assert running == next(s for s in STAGES if s not in done)
    # every cut position, not only frame boundaries: done count is monotonic
    counts = [len(F.done_stages(audit[:k])) for k in range(len(audit) + 1)]
    assert counts == sorted(counts)
    assert frames[F.pause_frame(audit)][1] == "ft_screening"   # pause right after TA


def test_step_lines_are_plain_and_match_the_log():
    audit = _audit(DEMO)
    lines = F.step_lines(audit, None, F.done_stages(audit))
    assert [l["status"] for l in lines] == ["done"] * len(STAGES)
    text = {l["stage"]: l["text"] for l in lines}
    assert text["retrieval"] == "618 papers found · 555 after removing duplicates"
    assert text["ft_screening"].startswith("70 read · 55 kept")


@pytest.mark.parametrize("status,pending,audit,auto,step", [
    ("starting", None, [], False, 2),
    ("running", None, [{"node": "planner", "event": "x"}], False, 2),
    ("waiting", {"kind": "protocol_approval"}, [], False, 2),
    ("running", None, [{"node": "protocol_approval", "event": "protocol approved"}], False, 3),
    ("running", None, [], True, 3),
    ("waiting", {"kind": "screening_conflicts"}, [], False, 3),
    ("waiting", {"kind": "budget_cap"}, [], True, 3),
    ("error", None, [], False, 3),
    ("stopped", None, [], False, 3),
    ("finished", None, [], False, 4),
])
def test_real_step_comes_from_the_status(status, pending, audit, auto, step):
    assert F.real_step(status, pending, audit, auto) == step


def test_practice_checks_copy_samples_without_marks(tmp_path):
    rid = DEMO.name
    before = {p: p.read_bytes() for p in (SC.CHECKS_DIR / rid).glob("*")}
    F.practice_checks(rid, tmp_path)
    for stage, _ in F.CHECKS:
        rows = SC.load_check(rid, stage, tmp_path)["rows"]
        assert len(rows) == 10 and not any(r["mark"] for r in rows)
    SC.set_mark(rid, "ta", rows[0]["item_id"] if False else
                SC.load_check(rid, "ta", tmp_path)["rows"][0]["item_id"], "wrong", root=tmp_path)
    F.practice_checks(rid, tmp_path)                       # a second call keeps the marks
    assert SC.load_check(rid, "ta", tmp_path)["rows"][0]["mark"] == "wrong"
    assert {p: p.read_bytes() for p in (SC.CHECKS_DIR / rid).glob("*")} == before


# --- Demo, through the screens -----------------------------------------------------------

def guide(fn="page_guide", **session):
    script = ("from sr_pipeline.ui import guide as G\n"
              f"G.{fn}()\n")
    at = AppTest.from_string(script, default_timeout=60)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def click(at, key):
    btn = at.button(key=key)
    assert not btn.disabled, key
    at = btn.click().run()
    assert not at.exception, at.exception
    return at


def test_home_offers_the_two_choices():
    at = guide("page_home")
    assert not at.exception
    assert not at.button(key="home_demo").disabled
    assert at.button(key="home_real")


def test_demo_runs_start_to_end_by_its_main_buttons(tmp_path, monkeypatch):
    from sr_pipeline.ui import guide as G
    monkeypatch.setattr(G, "DEMO_CHECKS", tmp_path)
    tracked = {p: p.read_bytes() for p in SC.CHECKS_DIR.rglob("*") if p.is_file()}
    demo_files = {p: p.read_bytes() for p in DEMO.rglob("*") if p.is_file()}

    at = guide(flow_mode="demo", flow_step=1)
    assert not at.exception
    at = click(at, "demo_start")
    assert any("Approve the plan" in s.value for s in at.subheader)
    at = click(at, "demo_approve")
    assert at.session_state["flow_step"] == 3
    # The replay pauses for the practice disagreement after title/abstract screening, then
    # plays on; "Skip to the end" finishes it.
    pause = F.pause_frame(_audit(DEMO))
    at.session_state["replay_frame"] = pause
    at = at.run()
    assert at.button(key="practice_keep") and not at.success
    at = click(at, "practice_keep")
    assert any("the AI judge chose" in i.value for i in at.info)
    at = click(at, "demo_skip")
    assert any("Review finished" in s.value for s in at.success)
    at = click(at, "demo_results")
    assert any("Results" in s.value for s in at.subheader)
    at = click(at, "to_checks")
    for stage, _ in F.CHECKS:
        for i in range(10):
            at = click(at, f"ok_{stage}_{i}")
        assert any("Check finished: 10 of 10" in s.value for s in at.success)
        at = click(at, f"next_{stage}")
    assert any("All done" in s.value for s in at.success)
    # Nothing tracked was written: practice marks went to the scratch folder.
    assert {p: p.read_bytes() for p in SC.CHECKS_DIR.rglob("*") if p.is_file()} == tracked
    assert {p: p.read_bytes() for p in DEMO.rglob("*") if p.is_file()} == demo_files
    assert SC.summarise(DEMO.name, tmp_path)["ta"]["correct"] == 10


# --- Real run (fake LLM), through the same screens ------------------------------------------

@pytest.fixture
def ui(tmp_path, monkeypatch):
    from sr_pipeline.ui import pages as P
    cfg = e2e_config(tmp_path)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    L.CONFIG, L.INJECT = cfg, {**INJECT, "llm": e2e_llm()}
    monkeypatch.setattr(P, "RUNS_DIR", cfg.runs_dir)
    monkeypatch.setattr(P, "CHECKS_DIR", tmp_path / "checks")
    yield cfg
    for rid in list(L._WORKERS):
        L.join(rid, 30)
    L.CONFIG, L.INJECT = None, {}
    L._WORKERS.clear()


def real(cfg, **session):
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              "from sr_pipeline.ui import guide as G\n"
              f"P.RUNS_DIR = Path({str(cfg.runs_dir)!r})\n"
              f"P.CHECKS_DIR = Path({str(cfg.runs_dir.parent / 'checks')!r})\n"
              "G.page_guide()\n")
    at = AppTest.from_string(script, default_timeout=60)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def settle(at):
    rid = at.session_state["flow_run"]
    L.join(rid, 60)
    assert not L.is_alive(rid)
    at = at.run()
    assert not at.exception, at.exception
    return rid, at


def submit(at):
    btn = next(b for b in at.button if b.proto.is_form_submitter)
    return btn.click().run()


def test_real_run_start_plan_disagreement_results_check(ui):
    at = real(ui, flow_mode="real")
    assert not at.exception
    at.radio(key="g_kind").set_value("My own question").run()
    at.text_area(key="g_question").input(QUESTION)
    at = at.button(key="g_start").click().run()
    assert not at.exception, at.exception
    rid, at = settle(at)
    # Step 2: the plan, untouched -> approved as is (True), exactly like the CLI.
    assert any("Approve the plan" in s.value for s in at.subheader)
    at = submit(at)
    rid, at = settle(at)
    # Step 3: each disagreement the judge could not settle (title/abstract, then full text).
    asked = 0
    while [r for r in at.radio if r.key and r.key.startswith("conf_")]:
        assert any("AI at work" in s.value for s in at.subheader)
        at = submit(at)
        rid, at = settle(at)
        asked += 1
    assert asked >= 1
    # Step 4: results of the finished run.
    assert any("Results" in s.value for s in at.subheader)
    state = json.loads((ui.runs_dir / rid / "state.json").read_text())
    approval = next(e for e in state["audit_log"] if e["node"] == "protocol_approval")
    assert not approval.get("edits")
    assert state["completed_stages"][-1] == "report"
    # Step 5: draw a check and mark the first item; saved in the checks folder.
    at = click(at, "to_checks")
    at = click(at, "draw_ta")
    at = click(at, "ok_ta_0")
    rows = SC.load_check(rid, "ta", ui.runs_dir.parent / "checks")["rows"]
    assert rows[0]["mark"] == "correct"


def test_stopped_real_run_shows_a_continue_button(ui):
    rid = "20261114-120000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "screen_limit": 100}))
    (run / "audit.jsonl").write_text(json.dumps(
        {"node": "protocol_approval", "event": "protocol approved"}) + "\n")
    L.write_status(run, "error", error="no connection")
    at = real(ui, flow_mode="real", flow_run=rid)
    assert not at.exception
    assert any("no connection" in w.value for w in at.warning)
    assert at.button(key="g_resume")


# --- Review fixes ---------------------------------------------------------------------------

def test_snapshot_has_no_local_paths():
    for f in DEMO.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".jsonl", ".md", ".csv", ".bib"):
            text = f.read_text()
            assert "/Users/" not in text and str(Path.home()) not in text, f


def test_demo_twice_starts_unmarked(tmp_path, monkeypatch):
    from sr_pipeline.ui import guide as G
    monkeypatch.setattr(G, "DEMO_CHECKS", tmp_path)
    for _ in range(2):
        at = guide(flow_mode="demo", flow_step=1)
        at = click(at, "demo_start")
        at.session_state["flow_step"] = 5
        at = at.run()
        assert any("0 of 10 marked" in str(p.proto) for p in at.get("progress"))
        at = click(at, "ok_ta_0")
        assert SC.load_check(DEMO.name, "ta", tmp_path)["rows"][0]["mark"] == "correct"


@pytest.mark.parametrize("est,fallback,cap", [
    ({"high": 0.6}, {"high": 4.37}, 1.0),          # quick run: from its own estimate, min $1
    ({"high": 2.0}, {"high": 4.37}, 3.0),
    (None, {"high": 4.37}, 6.6),                   # everything: the largest known estimate
    (None, None, None),
])
def test_default_cap_follows_the_chosen_size(est, fallback, cap):
    assert F.default_cap(est, fallback) == cap


def test_home_offers_an_unfinished_review_found_on_disk(ui):
    rid = "20261114-130000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "started_from": "ui"}))
    L.write_status(run, "stopped")
    done = ui.runs_dir / "20261114-120000-fedcba"          # finished: not offered
    (done / "report").mkdir(parents=True)
    (done / "run.json").write_text(json.dumps({"question": "x", "started_from": "ui"}))
    (done / "report" / "run_summary.json").write_text("{}")
    assert F.unfinished_ui_runs(ui.runs_dir) == [run]
    script = ("from pathlib import Path\nfrom sr_pipeline.ui import pages as P\n"
              "from sr_pipeline.ui import guide as G\n"
              f"P.RUNS_DIR = Path({str(ui.runs_dir)!r})\nG.page_home()\n")
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception
    assert any(QUESTION in i.value for i in at.info)
    at = at.button(key="home_continue").click().run()
    assert at.session_state["flow_run"] == rid and at.session_state["flow_mode"] == "real"


@pytest.mark.parametrize("name", ["free question", "T1", "T2", "T3", "D0", "edge cases"])
def test_guided_plan_form_untouched_sends_true(ui, monkeypatch, name):
    from test_ui_live import seeded_protocols
    protocol = seeded_protocols()[name]
    rid = "20261114-140000-abcdef"
    run = ui.runs_dir / rid
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"question": QUESTION, "screen_limit": 100,
                                              "started_from": "ui"}))
    L.write_status(run, "waiting", pending={
        "kind": "protocol_approval", "protocol": protocol, "error": None,
        "suggested_designs": protocol.get("suggested_designs", []),
        "design_reason": protocol.get("design_reason", ""),
        "criteria_source": protocol.get("criteria_source")})
    sent = {}
    monkeypatch.setattr(L, "resume", lambda run_id, **kw: sent.update(kw) or run_id)
    at = real(ui, flow_mode="real", flow_run=rid)
    assert not at.exception
    assert any("Keep papers that" in a.label for a in at.text_area)
    submit(at)
    assert sent["kind"] == "protocol_approval" and sent["answer"] is True
