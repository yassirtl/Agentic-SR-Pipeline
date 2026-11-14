"""Blind labels page (Part 11g): label the screening sample yourself, without seeing the AI.

The sheet is ``evaluation/labels/<run_id>/labels.csv`` (title, abstract, empty ``human``),
made by ``python -m sr_pipeline.evaluation sheet RUN``. This page shows one record at a time
with the criteria, and never loads the pipeline's decisions while labelling: the run's
state is read only after the last record, to compute the score. Every click is saved at
once. The score is the human label as reference vs the pipeline's title/abstract decision
(``evaluation.labels.score_labels``): accuracy, sensitivity, specificity, Cohen's kappa.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from sr_pipeline.evaluation import labels as LB
from sr_pipeline.evaluation.__main__ import EVAL_DIR
from sr_pipeline.ui import pages as P

e = P.e
LABELS_ROOT: Path | None = None          # tests; otherwise evaluation/labels
NAMES = {"include": "✓ Include", "exclude": "✗ Exclude", "unsure": "? Not sure"}


def root() -> Path:
    return LABELS_ROOT or EVAL_DIR / "labels"


def _topic(sheet: Path) -> str:
    run = P.runs_dir() / sheet.parent.name / "run.json"
    try:
        return json.loads(run.read_text()).get("topic") or ""
    except (OSError, ValueError):
        return ""


def page_labels() -> None:
    P.setup()
    st.title("Blind labels")
    st.markdown('<p class="todo">👉 Decide for each paper, <b>from its title and abstract '
                'only</b>, whether it should go on to full-text reading. You do <b>not</b> see '
                "the AI's answer: that is what makes this an independent check. About 1–2 "
                'minutes per paper; every click is saved, you can stop and come back.</p>',
                unsafe_allow_html=True)
    found = LB.sheets(root())
    if not found:
        st.info("No label sheet yet. Make one with `python -m sr_pipeline.evaluation sheet RUN`.")
        return
    ids = [s.parent.name for s in found]
    pick = st.selectbox("Sample", ids, key="lb_sheet",
                        format_func=lambda i: f"{i} · topic {_topic(root() / i / 'labels.csv') or '–'}")
    sheet = root() / pick / "labels.csv"
    rows = LB.sheet_rows(sheet)
    done = [r for r in rows if (r.get("human") or "").strip()]
    st.progress(len(done) / max(1, len(rows)), text=f"{len(done)} of {len(rows)} labelled")
    crit = sheet.parent / "criteria.md"
    if crit.exists():
        with st.expander("The criteria (read once, open again any time)",
                         expanded=not done):
            import re
            st.markdown(re.sub(r"^#+ (.*)$", r"**\1**", crit.read_text(), flags=re.M))

    key = f"lb_i:{pick}"
    first_open = next((i for i, r in enumerate(rows) if not (r.get("human") or "").strip()),
                      len(rows))
    i = st.session_state.get(key, first_open)
    if i >= len(rows):
        finished(pick, sheet, rows)
        return
    r = rows[i]
    current = (r.get("human") or "").strip()
    st.markdown(
        f'<div class="hero"><div class="eyebrow">Paper {i + 1} of {len(rows)}'
        + (f" · your label: {e(NAMES.get(current, current))}" if current else "")
        + f'</div><div class="q" style="font-size:20px">{e(r.get("title") or "")}</div>'
        f'<div class="meta" style="line-height:1.55">{e(r.get("abstract") or "(no abstract)")}'
        "</div></div>", unsafe_allow_html=True)
    note = st.text_input("Optional note (which criterion, why)", value=r.get("note") or "",
                         key=f"lb_note:{pick}:{i}")

    def save(label: str):
        LB.set_label(sheet, r["paper_id"], label, note)
        after = LB.sheet_rows(sheet)
        nxt = next((k for k in range(i + 1, len(after))
                    if not (after[k].get("human") or "").strip()), None)
        if nxt is None:
            nxt = next((k for k, x in enumerate(after) if not (x.get("human") or "").strip()),
                       len(after))
        st.session_state[key] = nxt
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("✓ Include", key=f"lb_in:{i}", type="primary", width="stretch"):
        save("include")
    if c2.button("✗ Exclude", key=f"lb_ex:{i}", width="stretch"):
        save("exclude")
    if c3.button("? Not sure", key=f"lb_un:{i}", width="stretch"):
        save("unsure")
    if i > 0 and c4.button("← Previous", key=f"lb_prev:{i}", width="stretch"):
        st.session_state[key] = i - 1
        st.rerun()
    st.caption("Include unless the paper clearly fails a criterion; when in doubt, include "
               "(criteria marked full text only cannot exclude here). Not sure = left out of "
               "the score.")


def finished(run_id: str, sheet: Path, rows: list[dict]) -> None:
    labels, problems = LB.read_labels(sheet)
    unsure = sum(1 for r in rows if (r.get("human") or "").strip() in LB.UNSURE)
    st.success(f"✓ All {len(rows)} papers labelled ({len(labels)} include/exclude, "
               f"{unsure} not sure). Thank you.")
    for p in problems:
        st.warning(p)
    c1, _, c3 = st.columns([1, 2, 1])
    if c1.button("← Review my labels", key="lb_review", width="stretch"):
        st.session_state[f"lb_i:{run_id}"] = 0
        st.rerun()
    state_file = P.runs_dir() / run_id / "state.json"
    if not state_file.exists():
        st.info("The run's files are not on this computer, so the score cannot be computed "
                "here. Run `python -m sr_pipeline.evaluation evaluate RUN` where the run is.")
        return
    if not c3.button("Show the AI's score →", key="lb_score", type="primary",
                     width="stretch") and not st.session_state.get("lb_scored"):
        return
    st.session_state["lb_scored"] = True
    sc = LB.score_labels(json.loads(state_file.read_text()), labels)

    def pct(x):
        return "–" if x is None else f"{100 * x:.0f}%"
    P.kpis([("Accuracy", pct(sc["accuracy"]), f"{sc['labelled']} papers compared"),
            ("Sensitivity", pct(sc["sensitivity"]),
             f"your {sc['human_includes']} includes the AI kept"),
            ("Specificity", pct(sc["specificity"]),
             f"your {sc['human_excludes']} excludes the AI removed"),
            ("Agreement (kappa)", "–" if sc["kappa"] is None else f"{sc['kappa']:.2f}",
             "you vs the AI, beyond chance")])
    c = sc["confusion"]
    st.markdown(f"You include · AI includes **{c['tp']}** · AI excludes **{c['fn']}** "
                f"(a lost study)  \nYou exclude · AI includes **{c['fp']}** · AI excludes "
                f"**{c['tn']}**")
    if sc["disagreements"]:
        st.markdown("**Where you and the AI disagree**")
        for d in sc["disagreements"]:
            st.markdown(f"- {e(d['title'] or '')}: you **{d['human']}**, AI **{d['pipeline']}**"
                        + (f" (criterion {d['criterion']}"
                           + (f": {d['label']}" if d.get("label") else "") + ")"
                           if d.get("criterion") else ""))
    st.caption("This is the independent measure for the report. Running "
               "`python -m sr_pipeline.evaluation evaluate RUN` writes it into "
               "evaluation/results.")
