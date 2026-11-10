"""Run viewer, cost estimator (Part 11a), live runs (Part 11b) and per-stage human check
(Part 11c). Start with ``sr-ui`` (or
``streamlit run src/sr_pipeline/ui/app.py``). The viewer and estimator read files only; the
Live run page starts real runs (API calls, billed)."""

import streamlit as st

from sr_pipeline.ui import pages as P

st.set_page_config(page_title="SR Pipeline · Runs", page_icon="📚", layout="wide")
P.setup()
P.PAGES["runs"] = st.Page(P.page_runs, title="Runs", icon="🗂️", default=True)
P.PAGES["detail"] = st.Page(P.page_detail, title="Run detail", icon="🔎", url_path="run")
P.PAGES["live"] = st.Page(P.page_live, title="Live run", icon="▶️", url_path="live")
P.PAGES["check"] = st.Page(P.page_check, title="Human check", icon="✅", url_path="check")
P.PAGES["estimator"] = st.Page(P.page_estimator, title="Cost estimator", icon="🧮", url_path="estimate")
st.navigation(list(P.PAGES.values())).run()
