"""Home page and guided review (Part 11e): one path, a step bar, one main button per screen.

Logic is in ``flow.py``; the live runner and the three pause forms are reused from
``live.py`` / ``pages.py`` (same answers as the CLI). Demo mode reads the tracked snapshot in
docs/demo and writes only to a scratch folder (practice marks); real mode starts a billed run.

Session keys: ``flow_mode`` (demo / real), ``flow_run`` (real run id), ``flow_step`` (demo
step; real: 5 when the person moved on from the results), ``replay_frame``,
``practice_answer``, ``chk_idx`` (which check), ``chk_item`` (item shown, None = first
unmarked). The URL keeps mode, step and run, so a reload stays on the same screen.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from sr_pipeline.evaluation import stage_checks as SC
from sr_pipeline.ui import data as D
from sr_pipeline.ui import flow as F
from sr_pipeline.ui import live as L
from sr_pipeline.ui import pages as P

e = P.e
DEMO_DIR: Path | None = None          # tests; otherwise docs/demo
DEMO_CHECKS: Path | None = None       # tests; otherwise a scratch folder (never tracked files)
REPLAY_SECONDS = 1.5                  # one stage per tick

CSS = """
<style>
.stepper { display:flex; gap:6px; flex-wrap:wrap; margin: 0 0 22px; }
.stepper .s { flex:1 1 150px; border-radius:10px; padding:8px 12px; font-size:14px;
              border:1px solid var(--line); background:var(--card); color:var(--ink3); }
.stepper .s b { display:block; font-size:12px; letter-spacing:.05em; text-transform:uppercase; }
.stepper .s.done { color:#15803d; border-color:#15803d55; background:#f0fdf4; }
.stepper .s.now { color:#fff; background:var(--blue); border-color:var(--blue); font-weight:650; }
.banner { background:#fff7e6; border:1px solid #f5c26b; color:#7a4b00; border-radius:10px;
          padding:8px 14px; font-size:14px; margin-bottom:14px; }
.choice { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:20px 22px; min-height:150px; }
.choice .t { font-size:22px; font-weight:700; color:var(--ink); }
.choice .s { font-size:15px; color:var(--ink2); margin-top:6px; }
.todo { font-size:18px; color:var(--ink); margin: 0 0 14px; }
.check { display:grid; grid-template-columns: 34px 1fr; gap:10px; padding:9px 4px;
         border-bottom:1px solid var(--line); align-items:start; }
.check .i { font-size:20px; text-align:center; }
.check .l { font-size:16px; color:var(--ink); }
.check .x { font-size:14px; color:var(--ink2); }
.check.waiting .l { color:var(--ink3); }
.check.running { background:#2a78d60f; border-radius:8px; }
.check.running .l { font-weight:700; }
.funnel { display:flex; flex-wrap:wrap; align-items:center; gap:10px; font-size:18px; margin:8px 0 18px; }
.funnel .b { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 16px; }
.funnel .b b { font-size:26px; display:block; }
.funnel .a { color:var(--ink3); font-size:22px; }
.item { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 22px; }
.item .t { font-size:19px; font-weight:700; color:var(--ink); }
.item .d { font-size:17px; margin:10px 0 6px; }
</style>
"""

ICON = {"done": "✅", "running": "⏳", "waiting": "◻️"}


# --- Small parts ------------------------------------------------------------------------

def setup() -> None:
    P.setup()
    st.markdown(CSS, unsafe_allow_html=True)


def demo_dir() -> Path | None:
    return F.demo_run_dir(DEMO_DIR)


def demo_checks_root() -> Path:
    return DEMO_CHECKS or Path(tempfile.gettempdir()) / "sr-demo-checks"


def stepper(now: int) -> None:
    cells = []
    for i, name in enumerate(F.STEPS, start=1):
        cls = "done" if i < now else "now" if i == now else ""
        mark = "✓ " if i < now else ""
        cells.append(f'<div class="s {cls}"><b>Step {i}</b>{mark}{e(name)}</div>')
    st.markdown('<div class="stepper">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def todo(text: str) -> None:
    """The one sentence that says what to do on this screen."""
    st.markdown(f'<div class="todo">👉 {text}</div>', unsafe_allow_html=True)


def demo_banner() -> None:
    st.markdown('<div class="banner">▶ <b>Demo</b>: a replay of a real review that already ran '
                '(topic T2, 8 Nov 2026). Free: no AI calls, nothing is changed.</div>',
                unsafe_allow_html=True)


def go(**changes) -> None:
    for k, v in changes.items():
        st.session_state[k] = v
    sync_url()
    st.rerun()


def sync_url() -> None:
    qp = st.query_params
    mode = st.session_state.get("flow_mode")
    if mode:
        qp["mode"] = mode
    if mode == "demo":
        qp["step"] = str(st.session_state.get("flow_step", 1))
    if mode == "real" and st.session_state.get("flow_run"):
        qp["run"] = st.session_state["flow_run"]


def read_url() -> None:
    """A reload (or a shared link) lands on the same screen."""
    qp = st.query_params
    if "flow_mode" not in st.session_state and qp.get("mode") in ("demo", "real"):
        st.session_state["flow_mode"] = qp["mode"]
        if qp.get("mode") == "demo" and str(qp.get("step", "")).isdigit():
            st.session_state["flow_step"] = max(1, min(5, int(qp["step"])))
        if qp.get("mode") == "real" and qp.get("run"):
            st.session_state["flow_run"] = qp["run"]


def reset_flow(mode: str | None) -> None:
    for k in ("flow_step", "flow_run", "replay_frame", "practice_answer", "chk_idx",
              "chk_item"):
        st.session_state.pop(k, None)
    st.session_state["flow_mode"] = mode
    st.query_params.clear()


def _open_guide(mode: str) -> None:
    reset_flow(mode)
    if mode == "demo" and demo_dir() is not None:
        F.reset_practice(demo_dir().name, demo_checks_root())
    st.session_state["flow_step"] = 1
    sync_url()
    if "guide" in P.PAGES:
        st.switch_page(P.PAGES["guide"])
    st.rerun()


# --- Home -------------------------------------------------------------------------------

def page_home() -> None:
    setup()
    st.title("AI Systematic Review")
    st.markdown('<p class="todo">Turn one research question into a full literature review. '
                'Every decision the AI makes comes with the sentence from the paper that '
                'proves it.</p>', unsafe_allow_html=True)
    choices()
    open_runs = F.unfinished_ui_runs(P.runs_dir())
    if open_runs:
        run = open_runs[0]
        v = D.live_view(run, alive=L.is_alive(run.name))
        q = v.meta.get("question") or TOPIC_NAMES.get(v.meta.get("topic"), v.meta.get("topic"))
        st.info(f"You have an unfinished review: “{q}” (started {run.name[:8]}, {v.status}).")
        if st.button("Continue your review →", key="home_continue", type="primary"):
            reset_flow("real")
            st.session_state["flow_run"] = run.name
            sync_url()
            if "guide" in P.PAGES:
                st.switch_page(P.PAGES["guide"])
            st.rerun()
    st.divider()
    c1, c2 = st.columns(2)
    if "runs" in P.PAGES:
        c1.page_link(P.PAGES["runs"], label="Open a past review")
    c2.markdown('<p class="note">⚙ <b>Advanced tools</b> (left menu): every detail of a run, '
                'the cost estimator and the full human check.</p>', unsafe_allow_html=True)


def choices() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="choice"><div class="t">▶ Watch a demo</div><div class="s">'
                    'Free · about 3 minutes. Replays a finished review step by step: the plan, '
                    'the AI at work, the results, and checking the AI.</div></div>',
                    unsafe_allow_html=True)
        if st.button("Watch the demo →", key="home_demo", type="primary", width="stretch",
                     disabled=demo_dir() is None):
            _open_guide("demo")
        if demo_dir() is None:
            st.caption("The demo files (docs/demo) are missing from this copy.")
    with c2:
        st.markdown('<div class="choice"><div class="t">＋ Start a new review</div><div class="s">'
                    'A real run with the OpenAI API: about $0.50 and 15 minutes for 100 papers. '
                    'You approve the plan and settle disagreements; a budget cap stops it '
                    'if it costs more than you allow.</div></div>', unsafe_allow_html=True)
        if st.button("Start a new review →", key="home_real", width="stretch"):
            _open_guide("real")


# --- Guided review ------------------------------------------------------------------------

def page_guide() -> None:
    setup()
    read_url()
    mode = st.session_state.get("flow_mode")
    if mode == "demo":
        demo_flow()
    elif mode == "real":
        real_flow()
    else:
        st.title("Guided review")
        todo("Choose one:")
        choices()


def nav(back: int | None = None, next_label: str | None = None, next_key: str = "next",
        on_next=None, disabled: bool = False) -> None:
    """Bottom buttons: one main button on the right, an optional Back on the left."""
    c1, _, c3 = st.columns([1, 2, 1])
    if back is not None and c1.button("← Back", key=f"back_{next_key}", width="stretch"):
        go(flow_step=back)
    if next_label and c3.button(next_label, key=next_key, type="primary", width="stretch",
                                disabled=disabled):
        on_next()


# --- Demo -------------------------------------------------------------------------------

def demo_flow() -> None:
    run = demo_dir()
    if run is None:
        st.error("The demo files (docs/demo) are missing from this copy of the project.")
        return
    step = st.session_state.get("flow_step", 1)
    stepper(step)
    demo_banner()
    snap = json.loads((run / "demo.json").read_text())
    res = snap["results"]
    if step == 1:
        st.subheader("Step 1 · The question")
        todo("This is the research question the review answers. Press <b>Start</b>.")
        st.markdown(f'<div class="hero"><div class="eyebrow">Research question</div>'
                    f'<div class="q">{e(res.get("question") or "")}</div></div>',
                    unsafe_allow_html=True)
        def begin():
            F.reset_practice(run.name, demo_checks_root())
            go(flow_step=2, replay_frame=0, practice_answer=None, chk_idx=0, chk_item=None)
        nav(next_label="Start →", next_key="demo_start", on_next=begin)
    elif step == 2:
        st.subheader("Step 2 · Approve the plan")
        todo("The AI turned the question into rules for which papers count. Read them, "
             "then press <b>Approve</b>.")
        plan_card(snap.get("protocol") or {})
        st.caption("In this replay the plan was approved automatically. In your own review "
                   "you can edit every rule here before anything is searched.")
        nav(back=1, next_label="Approve the plan →", next_key="demo_approve",
            on_next=lambda: go(flow_step=3, replay_frame=0, practice_answer=None))
    elif step == 3:
        demo_work(run, snap)
    elif step == 4:
        results_screen(run, res, demo=True)
    else:
        checks_screen(run.name, demo_checks_root(), demo=True, run_dir=run)


def plan_card(p: dict) -> None:
    years = p.get("years") or {}
    inc = "".join(f"<li>{e(x)}</li>" for x in p.get("inclusion_criteria") or [])
    exc = "".join(f"<li>{e(x)}</li>" for x in p.get("exclusion_criteria") or [])
    st.markdown(
        f'<div class="hero"><div class="eyebrow">Question</div>'
        f'<div class="q" style="font-size:19px">{e(p.get("question") or "")}</div>'
        f'<div class="eyebrow" style="margin-top:12px">✅ Keep papers that</div><ul>{inc}</ul>'
        + (f'<div class="eyebrow">❌ Throw out papers that</div><ul>{exc}</ul>' if exc else "")
        + f'<div class="meta">Search in: {e(", ".join(p.get("databases") or []))} · years '
        f'{e(str(years.get("start") or "any"))} to {e(str(years.get("end") or "today"))}</div>'
        '</div>', unsafe_allow_html=True)


def checklist(lines: list[dict]) -> None:
    html = ""
    for i, l in enumerate(lines, start=1):
        html += (f'<div class="check {l["status"]}"><div class="i">{ICON[l["status"]]}</div>'
                 f'<div><div class="l">{i}. {e(l["label"])}</div>'
                 f'<div class="x">{e(l["text"])}</div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def _demo_audit(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "audit.jsonl").read_text().splitlines() if x.strip()]


def demo_work(run: Path, snap: dict) -> None:
    audit = _demo_audit(run)
    frames = F.replay_frames(audit)
    pause = F.pause_frame(audit)
    frame = min(st.session_state.get("replay_frame", 0), len(frames) - 1)
    paused = (pause is not None and frame >= pause and st.session_state.get("practice_answer")
              is None and snap.get("practice"))
    if paused:
        frame = pause
    end = frame == len(frames) - 1
    st.subheader("Step 3 · The AI at work")
    if end:
        todo("The review is finished. Press <b>See the results</b>.")
    elif paused:
        todo("The two AI screeners disagreed about a paper. <b>What would you decide?</b>")
    else:
        todo("Watch the AI go through the 9 steps. Nothing to do yet.")
    _after_practice(snap)
    if not end and not paused:
        _replay_fragment(run.name, frame)
        if st.button("Skip to the end ⏭", key="demo_skip"):
            go(replay_frame=len(frames) - 1, practice_answer=st.session_state.get(
                "practice_answer") or "skipped")
        return
    if paused:
        practice_card(snap["practice"])
        st.markdown("&nbsp;", unsafe_allow_html=True)
        _replay_view(audit, frames, frame)
        return
    _replay_view(audit, frames, frame)
    r = snap["results"]
    st.success(f"✓ Review finished. In real time it took {r['minutes']:.0f} minutes and cost "
               f"${r['cost_usd']:.2f}.")
    nav(back=2, next_label="See the results →", next_key="demo_results",
        on_next=lambda: go(flow_step=4))


def _replay_view(audit: list[dict], frames, frame: int) -> None:
    cut, running = frames[frame]
    shown = audit[:cut]
    done = F.done_stages(shown)
    st.progress(len(done) / len(F.PLAIN), text=f"{len(done)} of {len(F.PLAIN)} steps done")
    checklist(F.step_lines(shown, running, done))


@st.fragment(run_every=REPLAY_SECONDS)
def _replay_fragment(run_id: str, start: int) -> None:
    run = demo_dir()
    audit = _demo_audit(run)
    frames = F.replay_frames(audit)
    pause = F.pause_frame(audit)
    frame = st.session_state.get("replay_frame", start)
    _replay_view(audit, frames, frame)
    nxt = frame + 1
    st.session_state["replay_frame"] = nxt
    if nxt >= len(frames) - 1 or (pause is not None and nxt >= pause
                                  and st.session_state.get("practice_answer") is None):
        st.session_state["replay_frame"] = min(nxt, len(frames) - 1)
        st.rerun()           # full page: practice question or the finished screen


def practice_card(p: dict) -> None:
    names = {"include": "Keep it", "exclude": "Throw it out"}
    st.markdown(
        f'<div class="item"><div class="t">{e(p.get("title") or "")}</div>'
        f'<div class="note" style="margin-top:8px">{e(p.get("abstract") or "")}</div></div>',
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, who in ((c1, "a"), (c2, "b")):
        v = p.get(who) or {}
        col.markdown(f"**Screener {who.upper()} says: {names.get(v.get('decision'), v.get('decision'))}**"
                     + (f"  \n_“{v['quote']}”_" if v.get("quote") else ""))
    st.caption("Practice: in a real review you are asked only when the AI judge cannot decide. "
               "Your answer here changes nothing.")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("✓ Keep it", key="practice_keep", type="primary", width="stretch"):
        go(practice_answer="include")
    if c2.button("✗ Throw it out", key="practice_drop", width="stretch"):
        go(practice_answer="exclude")


def _after_practice(snap: dict) -> None:
    p, ans = snap.get("practice") or {}, st.session_state.get("practice_answer")
    if ans in ("include", "exclude"):
        judge = (p.get("judge") or {}).get("decision")
        same = "the same as" if ans == judge else "different from"
        st.info(f"You chose **{'keep' if ans == 'include' else 'throw out'}**; the AI judge "
                f"chose **{'keep' if judge == 'include' else 'throw out'}**: {same} you.")


# --- Real run ---------------------------------------------------------------------------

def real_flow() -> None:
    rid = st.session_state.get("flow_run")
    if not rid:
        stepper(1)
        real_start()
        return
    run_dir = P.runs_dir() / rid
    v = D.live_view(run_dir, alive=L.is_alive(rid))
    step = F.real_step(v.status, v.pending, D.read_audit(run_dir), bool(v.meta.get("auto")))
    if step == 4 and st.session_state.get("flow_step") == 5:
        step = 5
    stepper(step)
    if step == 2:
        real_plan(rid, v)
    elif step == 3:
        real_work(rid, v)
    elif step == 4:
        res = F.results(run_dir)
        if res is None:
            st.error("The run finished but its results file is missing.")
            return
        results_screen(run_dir, res, demo=False)
    else:
        checks_screen(rid, P.checks_dir(), demo=False, run_dir=run_dir)


TOPIC_NAMES = {"T1": "Smartwatches that detect heart rhythm problems",
               "T2": "Small AI models that read ECGs on cheap devices",
               "T3": "AI ECG classifiers on embedded chips",
               "D0": "Exercise and depression in older adults"}
SIZES = {"Quick: the 100 most relevant papers": 100,
         "Thorough: 500 papers": 500,
         "Everything the search finds": 0}


def _topic_questions() -> dict[str, str]:
    import yaml

    from sr_pipeline.evaluation.topics import TOPICS_FILE
    try:
        data = yaml.safe_load(TOPICS_FILE.read_text()) or {}
    except OSError:
        return {}
    return {k: (v or {}).get("question", "") for k, v in data.items()
            if not str(k).startswith("_")}


def _size_label(name: str, est: dict | None) -> str:
    if not est:
        return name
    return f"{name} · about ${est['mid']:.2f}, {est['minutes']:.0f} min"


def real_start() -> None:
    st.subheader("Step 1 · Your question")
    todo("Pick an example question or type your own, choose the size, then press "
         "<b>Start the review</b>.")
    kind = st.radio("Question", ["An example question", "My own question"], horizontal=True,
                    key="g_kind", label_visibility="collapsed")
    ests = {n: (P.quick_estimate(v) if v else None) for n, v in SIZES.items()}
    # A form: what is typed is sent with the Start button (no "press Ctrl+Enter to apply").
    with st.form("g_start_form", border=False):
        topic = question = None
        if kind == "An example question":
            qs = _topic_questions()
            topic = st.selectbox("Example", list(qs), key="g_topic",
                                 format_func=lambda k: f"{TOPIC_NAMES.get(k, k)}: {qs[k]}")
        else:
            question = st.text_area("Your research question", key="g_question",
                                    placeholder="e.g. Does exercise reduce depressive "
                                                "symptoms in older adults?")
        size = st.radio("How many papers should the AI read?", list(SIZES), key="g_size",
                        format_func=lambda n: _size_label(n, ests[n]))
        biggest = max((x for x in ests.values() if x), key=lambda x: x["high"], default=None)
        with st.expander("Options"):
            cap = st.number_input("Stop if the cost reaches ($)", min_value=0.1,
                                  max_value=100.0, value=None, step=0.5, key="g_cap",
                                  placeholder="automatic: 1.5 × the highest expected cost",
                                  help="Checked between steps: the run pauses safely and you "
                                       "can continue with a higher limit.")
            auto = st.toggle("Don't ask me anything (approve the plan and settle "
                             "disagreements automatically)", value=False, key="g_auto")
        st.caption("This makes real, paid calls to the OpenAI API (key in the .env file).")
        c1, _, c3 = st.columns([1, 2, 1])
        go_home = c1.form_submit_button("← Home", width="stretch")
        started = c3.form_submit_button("Start the review →", type="primary", width="stretch",
                                        key="g_start")
    if go_home:
        reset_flow(None)
        st.switch_page(P.PAGES["home"]) if "home" in P.PAGES else st.rerun()
    if started:
        question = (question or "").strip() or None
        if not (topic or question):
            st.warning("Type your question first (or pick an example).")
            return
        cap = cap or F.default_cap(ests[size], biggest)
        rid = L.start(question=question, topic=topic, auto=auto, limit=SIZES[size],
                      budget_usd=cap, estimate=ests[size])
        go(flow_run=rid, flow_step=2)


@st.fragment(run_every=2)
def _watch(rid: str) -> None:
    """Refresh every 2 s while the run works; the full page reruns when it stops or asks."""
    v = D.live_view(P.runs_dir() / rid, alive=L.is_alive(rid))
    if v.status not in ("starting", "running"):
        st.rerun()
    audit = D.read_audit(P.runs_dir() / rid)
    lines = F.step_lines(audit, v.current, F.done_stages(audit))
    n = sum(1 for l in lines if l["status"] == "done")
    st.progress(n / len(lines), text=f"{n} of {len(lines)} steps done")
    checklist(lines)
    P.spend_bar(v)


def real_plan(rid: str, v: D.LiveView) -> None:
    st.subheader("Step 2 · Approve the plan")
    if v.status == "waiting" and v.pending:
        todo("The AI turned your question into rules for which papers count. Change anything "
             "you want, then press <b>Approve and continue</b> at the bottom.")
        P.pending_form(rid, v.pending, guided=True)
        return
    todo("The AI is writing the plan. This takes about a minute; the page updates by itself.")
    _watch(rid)


def real_work(rid: str, v: D.LiveView) -> None:
    st.subheader("Step 3 · The AI at work")
    audit = D.read_audit(P.runs_dir() / rid)
    if v.status in ("starting", "running"):
        todo("The AI is working through the 9 steps. Nothing to do: the page updates by "
             "itself, and asks you if it needs you.")
        _watch(rid)
        return
    lines = F.step_lines(audit, None, F.done_stages(audit))
    checklist(lines)
    P.spend_bar(v)
    kind = (v.pending or {}).get("kind")
    if v.status == "waiting" and kind == "screening_conflicts":
        todo("The AI could not decide about some papers. <b>Decide for each one</b>, then "
             "submit.")
        P.pending_form(rid, v.pending, guided=True)
    elif v.status == "waiting" and kind == "budget_cap":
        todo("The run reached your cost limit and paused safely. <b>Raise the limit to "
             "continue</b>, or leave it and come back later.")
        P.pending_form(rid, v.pending, guided=True)
    else:
        why = {"error": "The run stopped on an error" + (f": {v.error}" if v.error else "."),
               "stopped": "The run stopped (the app was closed or the connection dropped).",
               "ended": "The run ended before writing the report."}.get(
                   v.status, f"The run is {v.status}.")
        st.warning(why + " Everything done so far is saved.")
        todo("Press <b>Continue the review</b> to go on from where it stopped.")

        def resume():
            L.resume(rid)
            st.rerun()
        nav(next_label="Continue the review →", next_key="g_resume", on_next=resume)


# --- Results (both modes) ---------------------------------------------------------------

def results_screen(run_dir: Path, r: dict, *, demo: bool) -> None:
    st.subheader("Step 4 · Results")
    todo("Here is what the review found. Download the report, or press <b>Check the AI's "
         "accuracy</b>.")
    st.markdown(f'<div class="hero"><div class="eyebrow">Question</div>'
                f'<div class="q" style="font-size:19px">{e(r.get("question") or "")}</div>'
                f'<div class="meta">Finished in {r.get("minutes", 0):.0f} minutes · cost '
                f'${r.get("cost_usd", 0):.2f}</div></div>', unsafe_allow_html=True)

    def box(n, label):
        return f'<div class="b"><b>{P._num(n)}</b>{e(label)}</div>'
    st.markdown('<div class="funnel">' + box(r.get("found"), "papers found")
                + '<span class="a">→</span>' + box(r.get("unique"), "after removing duplicates")
                + '<span class="a">→</span>' + box(r.get("screened"), "read by the AI")
                + '<span class="a">→</span>' + box(r.get("included"), "kept in the review")
                + "</div>", unsafe_allow_html=True)
    if r.get("not_screened"):
        st.caption(f"{P._num(r['not_screened'])} less relevant papers were not read (size "
                   f"chosen at the start). {P._num(r.get('abstract_only'))} papers had no free "
                   "full text and were judged on their summary.")
    files = F.report_files(run_dir)
    docx = next((f for f in files if f.suffix == ".docx"), files[0] if files else None)
    c1, c2, c3 = st.columns([1, 1, 1])
    if back := (3 if demo else None):
        if c1.button("← Back", key="back_to_checks", width="stretch"):
            go(flow_step=back)
    if docx:
        c2.download_button("⬇ Download the report", docx.read_bytes(), file_name=docx.name,
                           key="dl_main", width="stretch")
    if c3.button("Check the AI's accuracy →", key="to_checks", type="primary", width="stretch"):
        go(flow_step=5, chk_idx=0, chk_item=None)
    kept = r.get("kept") or []
    with st.expander(f"See the {len(kept)} papers kept, with the proof for each"):
        for i, k in enumerate(kept, start=1):
            st.markdown(f"**{i}. {k.get('title') or ''}**  \nJudged on the {k.get('how')}."
                        + (f" [Open the paper ↗]({k['link']})" if k.get("link") else ""))
            if k.get("why"):
                st.markdown(f'<div class="quote" style="margin-bottom:12px">Why kept: '
                            f'“{e(k["why"])}”</div>', unsafe_allow_html=True)
    png = run_dir / "report" / "prisma_flow.png"
    if png.exists():
        with st.expander("See the funnel diagram (PRISMA 2020)"):
            st.image(str(png))
    if len(files) > 1:
        with st.expander("Other files (tables, references, diagram)"):
            for f in files:
                if f != docx:
                    st.download_button(f"⬇ {f.name}", f.read_bytes(), file_name=f.name,
                                       key=f"dl_{f.name}")


# --- Check accuracy (both modes) -------------------------------------------------------

def checks_screen(run_id: str, root: Path, *, demo: bool, run_dir: Path) -> None:
    st.subheader("Step 5 · Check the AI's accuracy")
    if demo:
        F.practice_checks(run_id, root)
    idx = st.session_state.get("chk_idx", 0)
    if idx >= len(F.CHECKS):
        all_done(run_id, root, demo)
        return
    stage, question = F.CHECKS[idx]
    st.markdown(f"**Check {idx + 1} of {len(F.CHECKS)}** · {question}")
    check = SC.load_check(run_id, stage, root)
    if check is None:
        todo("You will see 10 random examples of the AI's work, one at a time, and say if each "
             "one is right. About 5 minutes.")
        state_file = run_dir / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        n = len(SC.stage_items(state, stage))
        if not n:
            st.info("This review has nothing to check at this step.")
            nav(next_label="Next check →", next_key=f"skip_{stage}",
                on_next=lambda: go(chk_idx=idx + 1, chk_item=None))
            return

        def draw():
            chunks = SC.load_chunks(run_dir) if stage in ("ft", "extraction") else None
            SC.create_check(state, run_id, stage, SC.DEFAULT_N, SC.DEFAULT_SEED, root=root,
                            chunks=chunks)
            go(chk_item=None)
        nav(back=4, next_label="Start this check →", next_key=f"draw_{stage}", on_next=draw)
        return
    rows = check["rows"]
    s = SC.score(rows)
    st.progress(s["marked"] / max(1, len(rows)), text=f"{s['marked']} of {len(rows)} marked")
    i = st.session_state.get("chk_item")
    i = F.next_unmarked(rows) if i is None else i
    if i >= len(rows):
        share = f"{s['correct']} of {s['judged']}" if s["judged"] else "none"
        st.success(f"✓ Check finished: {share} judged correct"
                   + (f" ({s['unsure']} not sure)." if s["unsure"] else "."))
        last = idx + 1 == len(F.CHECKS)
        c1, _, c3 = st.columns([1, 2, 1])
        if c1.button("← Review my answers", key=f"rev_{stage}", width="stretch"):
            go(chk_item=0)
        if c3.button("Finish ✓" if last else "Next check →", key=f"next_{stage}",
                     type="primary", width="stretch"):
            go(chk_idx=idx + 1, chk_item=None)
        return
    todo("Is the AI right? Read the card, then press <b>Correct</b>, <b>Wrong</b> or "
         "<b>Not sure</b>.")
    item_card(stage, rows[i], i, len(rows))
    note = st.text_input("Optional note (why wrong or not sure)",
                         value=rows[i].get("note") or "", key=f"note_{stage}_{i}")

    def mark(m):
        SC.set_mark(run_id, stage, rows[i]["item_id"], m, note, root=root)
        first_open = F.next_unmarked(SC.load_check(run_id, stage, root)["rows"])
        if st.session_state.get("chk_item") is None:      # normal path: next unmarked item
            go(chk_item=None)
        target = i + 1                                    # reviewing: walk on, then back to gaps
        go(chk_item=first_open if target >= len(rows) and first_open < len(rows) else target)
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("✓ Correct", key=f"ok_{stage}_{i}", type="primary", width="stretch"):
        mark("correct")
    if c2.button("✗ Wrong", key=f"bad_{stage}_{i}", width="stretch"):
        mark("wrong")
    if c3.button("? Not sure", key=f"unsure_{stage}_{i}", width="stretch"):
        mark("unsure")
    if i > 0 and c4.button("← Previous", key=f"prev_{stage}_{i}", width="stretch"):
        go(chk_item=i - 1)
    if st.button("Skip this check", key=f"skipchk_{stage}"):
        go(chk_idx=idx + 1, chk_item=None)


def item_card(stage: str, r: dict, i: int, n: int) -> None:
    if stage in ("ta", "ft"):
        dec = {"include": "✅ Kept the paper", "exclude": "❌ Threw the paper out"}.get(
            r.get("decision"), r.get("decision") or "")
    else:
        dec = f"Copied this fact: <b>{e(r.get('decision') or '')}</b>"
    detail = r.get("detail") or ""
    done = SC.normalise_mark(r.get("mark"))
    st.markdown(
        f'<div class="item"><div class="note">Example {i + 1} of {n}'
        + (f' · your answer: <b>{e(done)}</b>' if done else "") + '</div>'
        f'<div class="t">{e(r.get("title") or "")}</div>'
        f'<div class="d">The AI: {dec if stage not in ("ta", "ft") else e(dec)}</div>'
        f'<div class="note">{e(detail)}</div>'
        + (f'<div class="quote" style="margin-top:8px">Proof from the paper: “{e(r["quote"])}”</div>'
           if r.get("quote") else "")
        + (f'<div class="note" style="margin-top:6px"><a href="{e(r["link"])}" '
           f'target="_blank">open the paper ↗</a></div>' if r.get("link") else "")
        + "</div>", unsafe_allow_html=True)
    if r.get("context"):
        with st.expander("Read the summary / passage"):
            st.markdown(f'<div class="note">{e(r["context"])}</div>', unsafe_allow_html=True)


def all_done(run_id: str, root: Path, demo: bool) -> None:
    st.success("✓ All done. This is the end of the review.")
    summary = SC.summarise(run_id, root)
    rows = [{"Check": v["name"], "Marked": f"{v['marked']} / {v['sampled']}",
             "Judged correct": f"{v['correct']} of {v['correct'] + v['wrong']}"
             if v["correct"] + v["wrong"] else "–"} for v in summary.values()]
    if rows:
        st.markdown("**Your answers**")
        st.dataframe(rows, hide_index=True, width="stretch")
    if demo:
        official = SC.summarise(run_id)
        if official:
            bits = ", ".join(f"{v['name'].lower()} {v['correct']}/{v['correct'] + v['wrong']}"
                             for v in official.values() if v["correct"] + v["wrong"])
            st.caption(f"For comparison, the author's check of this run (in the technical "
                       f"report): {bits}. Your practice answers are not saved in the project.")
    c1, _, c3 = st.columns([1, 2, 1])
    if c1.button("← Back to the checks", key="done_back", width="stretch"):
        go(chk_idx=0, chk_item=None)
    if c3.button("Back to home", key="done_home", type="primary", width="stretch"):
        reset_flow(None)
        if "home" in P.PAGES:
            st.switch_page(P.PAGES["home"])
        st.rerun()
