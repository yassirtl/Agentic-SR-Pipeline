"""Pages of the run viewer and cost estimator (Part 11a). ``app.py`` wires them together.

Files only: no LLM, database or embedding call.
"""

from __future__ import annotations

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from sr_pipeline.config import load_config
from sr_pipeline.llm import PRICES
from sr_pipeline.ui import data as D

PAGES: dict = {}   # filled by app.py (st.Page objects, for switch_page)

BLUE = "#2a78d6"
STATUS = {  # icon + label + colour: colour never carries the meaning alone
    "done": ("✓", "done", "#0ca30c", "#e8f6e8"),
    "stale": ("↻", "stale", "#b07800", "#fdf3dc"),
    "not run": ("○", "not run", "#6b6a66", "#f1f0ed"),
    "refused": ("⊘", "refused", "#c2410c", "#fdeee7"),
    "finished": ("✓", "finished", "#0ca30c", "#e8f6e8"),
    "budget exhausted": ("!", "budget exhausted", "#b07800", "#fdf3dc"),
    "incomplete": ("○", "incomplete", "#6b6a66", "#f1f0ed"),
    "accepted": ("✓", "accepted", "#0ca30c", "#e8f6e8"),
    "llm_error": ("!", "LLM error", "#d03b3b", "#fbeaea"),
    "info": ("", "", "#52514e", "#f1f0ed"),
}

CSS = """
<style>
:root { --ink:#0b0b0b; --ink2:#52514e; --ink3:#8a8984; --line:#e6e4df; --card:#ffffff;
        --bg:#f7f6f3; --blue:#2a78d6; }
html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stAppViewContainer"] { font-size: 17px; }
.block-container { padding-top: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.hero { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
        padding: 20px 24px; margin-bottom: 18px; }
.hero .eyebrow { color: var(--ink3); font-size: 13px; text-transform: uppercase;
                 letter-spacing: .06em; font-weight: 600; }
.hero .q { font-size: 24px; font-weight: 650; color: var(--ink); margin: 6px 0 10px; line-height: 1.3; }
.hero .meta { color: var(--ink2); font-size: 15px; }
.pill { display: inline-flex; align-items: center; gap: 5px; padding: 2px 10px;
        border-radius: 999px; font-size: 13px; font-weight: 650; border: 1px solid; white-space: nowrap; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px;
        margin: 6px 0 18px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 16px; }
.kpi .l { color: var(--ink2); font-size: 13px; font-weight: 600; }
.kpi .v { color: var(--ink); font-size: 28px; font-weight: 700; line-height: 1.2;
          font-variant-numeric: tabular-nums; }
.kpi .s { color: var(--ink3); font-size: 12.5px; }
.flow { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px;
        counter-reset: stage; }
.stage { background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--c);
         border-radius: 12px; padding: 12px 14px; position: relative; }
.stage .n { color: var(--ink3); font-size: 12px; font-weight: 700; }
.stage .t { font-size: 17px; font-weight: 700; color: var(--ink); margin: 2px 0 6px; }
.stage .io { font-size: 15px; color: var(--ink); font-variant-numeric: tabular-nums; margin: 6px 0; }
.stage .io b { font-weight: 700; }
.stage .io .arrow { color: var(--ink3); margin: 0 4px; }
.stage .u { font-size: 13px; color: var(--ink2); font-variant-numeric: tabular-nums;
            border-top: 1px dashed var(--line); padding-top: 6px; margin-top: 6px; }
.stage .badges { display: flex; gap: 6px; flex-wrap: wrap; }
.step { display: grid; grid-template-columns: 48px 1fr; gap: 12px; padding: 10px 0;
        border-bottom: 1px solid var(--line); }
.step .no { font-size: 20px; font-weight: 700; color: var(--ink3); text-align: right;
            font-variant-numeric: tabular-nums; }
.step .head { font-size: 16px; color: var(--ink); }
.step .why { font-size: 14px; color: var(--ink2); margin-top: 3px; }
.step .guard { font-size: 14px; color: #c2410c; margin-top: 3px; font-weight: 600; }
.note { color: var(--ink2); font-size: 14px; }
.quote { border-left: 3px solid var(--line); padding-left: 10px; color: var(--ink2); font-style: italic; }
</style>
"""

e = html.escape


def pill(kind: str, text: str | None = None) -> str:
    icon, label, fg, bg = STATUS.get(kind, ("•", kind, "#52514e", "#f1f0ed"))
    content = f"{icon} {e(text or label)}" if icon else e(text or label)
    return (f'<span class="pill" style="color:{fg};background:{bg};border-color:{fg}33">'
            f'{content}</span>')


def kpis(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(f'<div class="kpi"><div class="l">{e(l)}</div><div class="v">{e(v)}</div>'
                    f'<div class="s">{e(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def money(x: float | None) -> str:
    return "–" if x is None else f"${x:,.2f}" if x >= 0.1 else f"${x:,.3f}" if x >= 0.001 else (
        "$0" if x == 0 else "<$0.001")


def duration(sec: float | None) -> str:
    if sec is None:
        return "–"
    m = sec / 60
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


# --- Data access (cached, files only) ------------------------------------------------

RUNS_DIR: Path | None = None   # set by tests; otherwise the config's runs_dir (SR_RUNS_DIR)


def runs_dir() -> Path:
    global RUNS_DIR
    if RUNS_DIR is None:
        RUNS_DIR = load_config().runs_dir
    return RUNS_DIR


@st.cache_data(show_spinner=False, ttl=30)
def all_runs(path: str) -> list[D.RunInfo]:
    return D.discover_runs(Path(path))


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading run…", max_entries=4)
def get_run(path: str, mtime: float) -> D.Run:  # mtime: reload when the run changes
    return D.load_run(Path(path))


def open_run(info: D.RunInfo) -> D.Run:
    return get_run(str(info.path), _mtime(info.path / "state.json") + _mtime(info.path / "audit.jsonl"))


# --- Page 1: runs -----------------------------------------------------------------------

def page_runs() -> None:
    st.title("Runs")
    st.markdown(f'<p class="note">Folder <code>{e(str(runs_dir()))}</code> · read-only, '
                f'no API call.</p>', unsafe_allow_html=True)
    runs = all_runs(str(runs_dir()))
    finished = [r for r in runs if r.finished]
    kpis([
        ("Finished runs", str(len(finished)), f"{len(runs) - len(finished)} incomplete / smoke"),
        ("Topics", str(len({r.topic for r in finished if r.topic})),
         ", ".join(sorted({r.topic for r in finished if r.topic}))),
        ("Billed, all finished runs", money(sum(r.cost_usd or 0 for r in finished)),
         "cache hits cost $0"),
        ("Studies included (latest run)", f"{finished[0].included:,}" if finished and
         finished[0].included is not None else "–", finished[0].run_id if finished else ""),
    ])
    show_all = st.toggle("Show incomplete and smoke-test runs", value=False)
    shown = runs if show_all else finished
    if not shown:
        st.info("No run found. Run `sr-review --topic T1 --auto --limit 100` first.")
        return
    df = pd.DataFrame([{
        "Run": r.run_id, "Topic": r.topic or "–", "Status": r.status,
        "Date": (r.started_at or r.run_id)[:16].replace("T", " "),
        "Limit": r.screen_limit, "Screened": r.screened, "Included": r.included,
        "Cost ($)": r.cost_usd, "Time (min)": round(r.seconds / 60, 1) if r.seconds else None,
        "Question": r.question or "",
    } for r in shown])
    event = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "Cost ($)": st.column_config.NumberColumn(format="$%.3f"),
            "Limit": st.column_config.NumberColumn(format="%d"),
            "Screened": st.column_config.NumberColumn(format="%d"),
            "Included": st.column_config.NumberColumn(format="%d"),
            "Question": st.column_config.TextColumn(width="large"),
        })
    st.caption("Select a row to open the run.")
    rows = event.selection.rows if event and event.selection else []
    if rows:
        st.session_state["run_id"] = shown[rows[0]].run_id
        if "detail" in PAGES:
            st.switch_page(PAGES["detail"])


# --- Page 2: run detail --------------------------------------------------------------------

def stage_card_html(i: int, c: D.StageCard) -> str:
    color = STATUS[c.status][2]
    badges = pill(c.status)
    if c.runs > 1:
        badges += pill("info", f"ran ×{c.runs}")
    if c.refused:
        badges += pill("refused", f"refused ×{c.refused}")
    usage = (f"{c.calls:,} LLM call{'s' if c.calls != 1 else ''} ({c.cached:,} cached) · "
             f"{(c.input_tokens + c.output_tokens) / 1000:,.0f}k tokens · {money(c.cost_usd)}"
             if c.calls else "no LLM call")
    return (f'<div class="stage" style="--c:{color}"><div class="n">STAGE {i}</div>'
            f'<div class="t">{e(c.label)}</div><div class="badges">{badges}</div>'
            f'<div class="io">{e(c.count_in)}<span class="arrow">→</span><b>{e(c.count_out)}</b></div>'
            f'<div class="u">{e(usage)}</div></div>')


def tab_timeline(run: D.Run) -> None:
    steps = D.coordinator_steps(run)
    if not steps:
        st.info("No coordinator decision in the audit log.")
        return
    refused = sum(s["outcome"] == "refused" for s in steps)
    st.markdown(f'<p class="note">The LLM coordinator proposes the next stage after every step; '
                f'a code guardrail checks PRISMA prerequisites. {len(steps)} decisions, '
                f'{refused} refused.</p>', unsafe_allow_html=True)
    out = []
    for s in steps:
        head = f'proposed <b>{e(D.STAGE_LABELS.get(s["proposed"], s["proposed"] or "?"))}</b> '
        head += pill(s["outcome"])
        if s["outcome"] != "accepted":
            head += f' → ran <b>{e(D.STAGE_LABELS.get(s["ran"], s["ran"] or "?"))}</b>'
        guard = (f'<div class="guard">Guardrail: {e(s["reason"])}</div>' if s["reason"] else "")
        out.append(f'<div class="step"><div class="no">{e(str(s["step"]))}</div><div>'
                   f'<div class="head">{head}</div>{guard}'
                   f'<div class="why">Coordinator: “{e(s["rationale"])}”</div></div></div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def tab_prisma(run: D.Run) -> None:
    img = run.path / "report" / "prisma_flow.png"
    left, right = st.columns([3, 2])
    with left:
        if img.exists():
            st.image(str(img), width="stretch")
        else:
            st.info("No PRISMA diagram (report stage not run).")
    with right:
        ta = run.prisma.get("title_abstract") or {}
        ft = run.prisma.get("full_text") or {}
        st.subheader("Exclusion reasons")
        for name, block in (("Title/abstract", ta), ("Full text", ft)):
            reasons = block.get("excluded_by_criterion") or {}
            if reasons:
                st.markdown(f"**{name}** ({sum(reasons.values())} excluded)")
                st.dataframe(pd.DataFrame(sorted(reasons.items(), key=lambda x: -x[1]),
                                          columns=["Criterion", "Records"]),
                             hide_index=True, width="stretch")
        if ft.get("reports_not_retrieved"):
            st.markdown(f"**Full text not retrieved:** {ft['reports_not_retrieved']} of "
                        f"{ft.get('reports_sought')} (screened on abstract, flagged)")


def tab_screening(run: D.Run) -> None:
    ta = run.prisma.get("title_abstract") or {}
    ft = run.prisma.get("full_text") or {}
    kpis([
        ("Title/abstract κ (A vs B)", f"{ta['kappa']:.2f}" if ta.get("kappa") is not None else "–",
         f"agreement {ta.get('agreement', 0):.0%} on {ta.get('pairs', 0)} pairs"),
        ("Full-text κ (A vs B)", f"{ft['kappa']:.2f}" if ft.get("kappa") is not None else "–",
         f"agreement {ft.get('agreement', 0):.0%} on {ft.get('pairs', 0)} pairs"),
        ("TA included", f"{ta.get('included', 0):,}", f"of {ta.get('records_screened', 0):,} screened"),
        ("FT included", f"{ft.get('studies_included', 0):,}", f"of {ft.get('reports_sought', 0):,} sought"),
    ])
    stage = st.radio("Stage", ["Title/abstract", "Full text"], horizontal=True)
    rows = D.screening_rows(run, "ta" if stage == "Title/abstract" else "ft")
    if not rows:
        st.info("Decisions are read from state.json, which this run folder does not have.")
        return
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    dec = c1.multiselect("Decision", sorted(df["decision"].dropna().unique()))
    res = c2.multiselect("Resolved by", sorted(df["resolved_by"].dropna().unique()))
    q = c3.text_input("Search title")
    if dec:
        df = df[df["decision"].isin(dec)]
    if res:
        df = df[df["resolved_by"].isin(res)]
    if q:
        df = df[df["title"].fillna("").str.contains(q, case=False, regex=False)]
    st.caption(f"{len(df):,} records (latest decision per record)")
    st.dataframe(df, hide_index=True, width="stretch", height=480,
                 column_config={"title": st.column_config.TextColumn(width="large"),
                                "quote": st.column_config.TextColumn("deciding quote", width="large")})


def tab_extraction(run: D.Run) -> None:
    studies = run.table("studies")
    if not studies:
        st.info("No extraction table (report stage not run).")
        return
    df = pd.DataFrame(studies)
    st.markdown("**Included studies**")
    st.dataframe(df, hide_index=True, width="stretch", height=320)
    ext = pd.DataFrame(run.table("extraction"))
    if not ext.empty:
        st.markdown("**Extracted fields with supporting quotes**")
        label = st.selectbox("Study", df["label"] + " · " + df["title"].str.slice(0, 90))
        lab = label.split(" · ")[0]
        sub = ext[ext["label"] == lab][["field", "status", "value", "quote"]]
        st.dataframe(sub, hide_index=True, width="stretch",
                     column_config={"quote": st.column_config.TextColumn(width="large")})


def tab_rob(run: D.Run) -> None:
    rob = pd.DataFrame(run.table("risk_of_bias"))
    if rob.empty:
        st.info("No risk-of-bias table.")
        return
    overall = rob.drop_duplicates("label")["overall"].value_counts().rename_axis("Overall").reset_index(name="Studies")
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Overall judgement**")
        st.dataframe(overall, hide_index=True, width="stretch")
        st.markdown("**Tools used**")
        st.dataframe(rob.drop_duplicates("label")["tool"].value_counts().rename_axis("Tool")
                     .reset_index(name="Studies"), hide_index=True, width="stretch")
    with right:
        st.markdown("**Domain judgements** (quote verified against the text)")
        st.dataframe(rob[["label", "tool", "name", "judgment", "verified", "quote"]],
                     hide_index=True, width="stretch", height=420,
                     column_config={"quote": st.column_config.TextColumn(width="large")})


def tab_synthesis(run: D.Run) -> None:
    synth = (run.state or {}).get("synthesis") or {}
    qa = (run.state or {}).get("qa_report") or {}
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Narrative synthesis**")
        text = synth.get("text")
        if text:
            with st.container(height=520):
                st.markdown(text)
        else:
            st.info("No synthesis text in state.json.")
    with right:
        st.markdown("**Critic**")
        if qa:
            st.markdown(pill("accepted" if qa.get("passed") else "refused",
                             "passed" if qa.get("passed") else "not passed")
                        + f' &nbsp;{qa.get("blockers", 0)} blockers · {qa.get("warnings", 0)} warnings · '
                          f'{qa.get("loops_used", 0)} loop(s) used', unsafe_allow_html=True)
            for i in qa.get("issues") or []:
                with st.expander(f'{i.get("severity", "?")} · {i.get("kind", "")}'):
                    st.write(i.get("detail", ""))
                    if i.get("evidence"):
                        st.markdown(f'<div class="quote">{e(i["evidence"])}</div>', unsafe_allow_html=True)
                    if i.get("check"):
                        st.caption(f'code check: {i["check"]}')
        else:
            st.info("No critic report.")


def tab_evaluation(run: D.Run) -> None:
    ev = run.evaluation
    gr = ((ev or {}).get("completeness") or {}).get("gold_recall")
    if not gr:
        st.info("No gold standard for this run (evaluation without gold, or not evaluated).")
        return
    c = gr["counts"]
    rec = gr.get("recall") or {}
    kpis([
        ("Gold studies", str(c.get("gold")), "from the published review"),
        ("Found by the search", str(c.get("found")), f"{rec.get('search_vs_all', 0):.0%} of gold"),
        ("Included at the end", str(c.get("included")), f"{rec.get('final_vs_all', 0):.0%} of gold"),
        ("Screening recall", f"{rec.get('screening_ta_ft', 0):.0%}", "of gold studies that were screened"),
    ])
    steps = [("Gold", "gold"), ("Indexed in our databases", "covered"), ("Found by search", "found"),
             ("Screened", "screened"), ("Passed title/abstract", "ta_included"),
             ("Included", "included")]
    df = pd.DataFrame([{"Stage": s, "Gold studies": c.get(k, 0), "order": i}
                       for i, (s, k) in enumerate(steps)])
    chart = (alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=26)
             .encode(y=alt.Y("Stage:N", sort=alt.SortField("order"), title=None),
                     x=alt.X("Gold studies:Q", title="Gold studies still in the funnel",
                             scale=alt.Scale(domain=[0, c.get("gold") or 1])),
                     tooltip=["Stage", "Gold studies"])
             .properties(height=260))
    labels = chart.mark_text(align="left", dx=6, color="#0b0b0b", fontSize=14).encode(text="Gold studies:Q")
    st.altair_chart((chart + labels).configure_axis(labelFontSize=14, titleFontSize=13, grid=False, labelLimit=260, tickCount=6)
                    .configure_view(stroke=None), width="stretch")
    lost = [s for s in gr.get("studies") or [] if s.get("lost_at")]
    if lost:
        st.markdown("**Gold studies lost, and where**")
        st.dataframe(pd.DataFrame([{"Study": f'{s["first_author"]} {s["year"]}',
                                    "Lost at": s["lost_at"], "Reason": s.get("reason")}
                                   for s in lost]), hide_index=True, width="stretch")


def page_detail() -> None:
    runs = all_runs(str(runs_dir()))
    if not runs:
        st.info("No run found.")
        return
    ids = [r.run_id for r in runs]
    current = st.session_state.get("run_id")
    if current not in ids:
        current = next((r.run_id for r in runs if r.finished), ids[0])
    pick = st.sidebar.selectbox("Run", ids, index=ids.index(current),
                                format_func=lambda i: f"{i} · {next(r.topic or '–' for r in runs if r.run_id == i)}")
    st.session_state["run_id"] = pick
    info = next(r for r in runs if r.run_id == pick)
    run = open_run(info)
    s = run.summary
    llm = s.get("llm") or {}
    coord = (s.get("coordinator") or {}).get("decisions") or {}
    st.markdown(
        f'<div class="hero"><div class="eyebrow">Run {e(info.run_id)} · topic {e(info.topic or "–")} · '
        f'limit {e(str(info.screen_limit or "–"))}</div>'
        f'<div class="q">{e(info.question or "(no question)")}</div>'
        f'<div class="meta">{pill(info.status)} &nbsp;started {e((info.started_at or "")[:16].replace("T", " "))} UTC'
        f'</div></div>', unsafe_allow_html=True)
    kpis([
        ("Wall-clock time", duration(s.get("seconds")), "total; per-stage time not logged"),
        ("Coordinator steps", str(s.get("steps_taken", "–")),
         f'{coord.get("accepted", 0)} accepted · {coord.get("refused", 0)} refused'),
        ("LLM calls", f'{llm.get("calls", 0):,}', f'{llm.get("cached", 0):,} answered from cache'),
        ("Billed cost", money(llm.get("cost_usd")), f"fresh (no cache): {money(D.fresh_cost(run))}"),
        ("Studies included", f'{info.included:,}' if info.included is not None else "–",
         f'of {info.screened:,} screened' if info.screened else ""),
    ])
    st.subheader("Pipeline, in PRISMA order")
    cards = D.stage_cards(run)
    st.markdown('<div class="flow">' + "".join(stage_card_html(i + 1, c) for i, c in enumerate(cards))
                + "</div>", unsafe_allow_html=True)
    st.caption("✓ done = ran after all its prerequisites · ↻ stale = a prerequisite re-ran after it · "
               "⊘ refused = the coordinator proposed it and the guardrail said no. "
               "Coordinator calls are counted apart: "
               f'{((llm.get("per_node") or {}).get("coordinator") or {}).get("calls", 0)} calls, '
               f'{money(((llm.get("per_node") or {}).get("coordinator") or {}).get("cost_usd"))}.')
    st.write("")
    tabs = st.tabs(["Coordinator timeline", "PRISMA", "Screening", "Extraction", "Risk of bias",
                    "Synthesis + critic", "Evaluation", "Cost by stage"])
    with tabs[0]:
        tab_timeline(run)
    with tabs[1]:
        tab_prisma(run)
    with tabs[2]:
        tab_screening(run)
    with tabs[3]:
        tab_extraction(run)
    with tabs[4]:
        tab_rob(run)
    with tabs[5]:
        tab_synthesis(run)
    with tabs[6]:
        tab_evaluation(run)
    with tabs[7]:
        tab_cost(run)


def tab_cost(run: D.Run) -> None:
    per_node = ((run.summary.get("llm") or {}).get("per_node")) or {}
    rows = [{"Stage": D.STAGE_LABELS.get(k, k.replace("_", " ").title()), "Billed ($)": v["cost_usd"],
             "Calls": v["calls"], "Cached": v["cached"],
             "Input tokens": v["input_tokens"], "Output tokens": v["output_tokens"]}
            for k, v in per_node.items()]
    if not rows:
        st.info("No usage recorded.")
        return
    df = pd.DataFrame(rows)
    order = list(df["Stage"])
    chart = (alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22)
             .encode(y=alt.Y("Stage:N", sort=order, title=None),
                     x=alt.X("Billed ($):Q", title="Billed cost (USD)", axis=alt.Axis(format="$.2f")),
                     tooltip=["Stage", alt.Tooltip("Billed ($):Q", format="$.4f"), "Calls", "Cached"])
             .properties(height=34 * len(df)))
    st.altair_chart(chart.configure_axis(labelFontSize=14, titleFontSize=13, labelLimit=260, tickCount=6, gridColor="#ecebe7")
                    .configure_view(stroke=None), width="stretch")
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Billed ($)": st.column_config.NumberColumn(format="$%.4f")})
    emb = run.summary.get("embeddings") or {}
    if emb:
        st.caption(f'Embeddings ({e(str(emb.get("model", "")))}): {emb.get("texts_embedded", 0):,} texts embedded, '
                   f'{emb.get("texts_cached", 0):,} from cache, {money(emb.get("cost_usd"))}. '
                   f'Included in the run total, not in the stage rows.')


# --- Page 3: cost estimator ------------------------------------------------------------------

def page_estimator() -> None:
    st.title("Cost estimator")
    st.markdown('<p class="note">Estimates a <b>fresh</b> run (empty cache) from rates measured on '
                'finished runs. Each source run is scaled to your limit with its own include rates '
                'and per-record tokens; the range is the lowest, median and highest of those '
                'projections. No API call.</p>', unsafe_allow_html=True)
    finished = [r for r in all_runs(str(runs_dir())) if r.finished]
    if not finished:
        st.info("No finished run to measure rates from.")
        return
    rates = {r.run_id: D.run_rates(D.load_run(r.path, with_state=False)) for r in finished}
    rates = {k: v for k, v in rates.items() if v}

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Your review")
        st.text_input("Research question", placeholder="e.g. Do smartwatches detect atrial fibrillation?",
                      help="Shown for reference: the estimate depends on the limit and the models.")
        topics = sorted({r.topic for r in finished if r.topic})
        topic = st.selectbox("Similar topic (picks the source runs)", ["all topics"] + topics)
        st.multiselect("Databases", ["openalex", "pubmed", "arxiv"], default=["openalex", "pubmed"],
                       help="With --limit, the number of records screened is capped; databases change "
                            "which records are found, not how many are screened.")
        limit = st.number_input("--limit (records screened)", min_value=10, max_value=10000,
                                value=500, step=50)
        models_avail = [m for m in PRICES if PRICES[m][1] > 0]
        with st.expander("Models per stage"):
            st.caption("The audit log aggregates usage per stage, so the model is chosen per stage "
                       "(screening = screeners A/B + adjudicator; extraction includes risk of bias).")
            models = {}
            for stage in D.DRIVERS:
                models[stage] = st.selectbox(D.STAGE_LABELS.get(stage, "Coordinator"), models_avail,
                                             index=models_avail.index(D.DEFAULT_MODEL), key=f"m_{stage}")
    with right:
        st.subheader("Source runs")
        default = [k for k, r in rates.items() if topic == "all topics" or r.topic == topic]
        table = pd.DataFrame([{
            "Use": k in default, "Run": k, "Topic": r.topic, "Limit": r.screen_limit,
            "TA include": 100 * r.ta_rate, "FT include": 100 * r.ft_rate,
            "Sec / call": r.sec_per_call, "Billed ($)": r.billed_usd,
        } for k, r in rates.items()])
        edited = st.data_editor(
            table, hide_index=True, width="stretch", disabled=[c for c in table.columns if c != "Use"],
            key=f"src_{topic}",
            column_config={
                "TA include": st.column_config.NumberColumn(format="%.0f%%",
                    help="Share of screened records included at title/abstract."),
                "FT include": st.column_config.NumberColumn(format="%.0f%%",
                    help="Share of full-text reports sought that were included."),
                "Sec / call": st.column_config.NumberColumn(format="%.2f",
                    help="Wall-clock seconds per non-cached LLM call. Includes downloads, and "
                         "network outages for resumed runs: untick outliers."),
                "Billed ($)": st.column_config.NumberColumn(format="$%.3f"),
            })
        used = [rates[k] for k in edited.loc[edited["Use"], "Run"]]

    if not used:
        st.warning("Tick at least one source run.")
        return
    est = D.estimate(used, limit=int(limit), models=models)
    st.divider()
    vol = est["volume"]
    kpis([
        ("Estimated cost", f'{money(est["cost"].mid)}', f'range {money(est["cost"].low)} – {money(est["cost"].high)}'),
        ("LLM calls", f'{est["calls"].mid:,.0f}', f'range {est["calls"].low:,.0f} – {est["calls"].high:,.0f}'),
        ("Time", duration(est["minutes"].mid * 60),
         f'range {duration(est["minutes"].low * 60)} – {duration(est["minutes"].high * 60)}'),
        ("Studies included", f'{vol["included"].mid:,.0f}',
         f'range {vol["included"].low:,.0f} – {vol["included"].high:,.0f} of {int(limit):,} screened'),
    ])
    rows = []
    for s in est["stages"]:
        rows.append({"Stage": D.STAGE_LABELS.get(s.stage, "Coordinator"), "Scales": D.DRIVER_TEXT[s.driver],
                     "Model": s.model, "Cost ($)": s.cost.mid, "Cost low": s.cost.low, "Cost high": s.cost.high,
                     "Calls": round(s.calls.mid), "Minutes": round(s.minutes.mid, 1),
                     "Minutes range": f"{s.minutes.low:.1f} – {s.minutes.high:.1f}" if s.minutes.high < 10
                     else f"{s.minutes.low:.0f} – {s.minutes.high:.0f}"})
    emb = est["embeddings"]
    rows.append({"Stage": "Embeddings (RAG)", "Scales": D.DRIVER_TEXT["sought"], "Model": D.EMBED_MODEL,
                 "Cost ($)": emb.mid, "Cost low": emb.low, "Cost high": emb.high, "Calls": None,
                 "Minutes": None, "Minutes range": ""})
    df = pd.DataFrame(rows)
    chart_df = df[df["Cost ($)"] > 0]
    order = list(chart_df["Stage"])
    bars = (alt.Chart(chart_df).mark_bar(color=BLUE, cornerRadiusEnd=4, height=22)
            .encode(y=alt.Y("Stage:N", sort=order, title=None), x=alt.X("Cost ($):Q", title="Estimated cost (USD)", axis=alt.Axis(format="$.2f")),
                    tooltip=["Stage", alt.Tooltip("Cost ($):Q", format="$.3f"),
                             alt.Tooltip("Cost low:Q", format="$.3f"), alt.Tooltip("Cost high:Q", format="$.3f")]))
    rule = (alt.Chart(chart_df).mark_rule(color="#52514e", strokeWidth=2)
            .encode(y=alt.Y("Stage:N", sort=order), x="Cost low:Q", x2="Cost high:Q"))
    st.altair_chart((bars + rule).properties(height=34 * len(chart_df))
                    .configure_axis(labelFontSize=14, titleFontSize=13, labelLimit=260, tickCount=6, gridColor="#ecebe7").configure_view(stroke=None),
                    width="stretch")
    st.caption("Bar = median; grey line = low–high over the source runs.")
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="$%.3f") for c in ("Cost ($)", "Cost low", "Cost high")})
    st.markdown(
        f'<p class="note">Assumptions: the search finds at least {int(limit):,} records (fewer → lower cost); '
        f'tokens per record are the same for another model (tokenizers differ slightly); time is apportioned '
        f'by LLM calls because the pipeline does not log time per stage; embedding rate measured on '
        f'{est["embed_runs"]} of {len(used)} runs (the others hit the embedding cache, median used).</p>',
        unsafe_allow_html=True)




def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
