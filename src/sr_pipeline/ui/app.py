"""Run viewer and cost estimator (Part 11a). Start with ``sr-ui`` (or
``streamlit run src/sr_pipeline/ui/app.py``). Reads files only: no API call."""

import streamlit as st

from sr_pipeline.ui import pages as P

st.set_page_config(page_title="SR Pipeline · Runs", page_icon="📚", layout="wide")
P.setup()
P.PAGES["runs"] = st.Page(P.page_runs, title="Runs", icon="🗂️", default=True)
P.PAGES["detail"] = st.Page(P.page_detail, title="Run detail", icon="🔎", url_path="run")
P.PAGES["estimator"] = st.Page(P.page_estimator, title="Cost estimator", icon="🧮", url_path="estimate")
st.navigation(list(P.PAGES.values())).run()
