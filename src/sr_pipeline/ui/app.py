"""Streamlit app. Start with ``sr-ui`` (or ``streamlit run src/sr_pipeline/ui/app.py``).

Start: Home and the guided review (Part 11e: demo replay or a real run, one step at a time),
Past reviews. Advanced tools: run viewer and cost estimator (Part 11a), live run page
(Part 11b), full per-stage human check (Part 11c). Only a real run makes API calls (billed).
"""

import streamlit as st

from sr_pipeline.ui import blind as B
from sr_pipeline.ui import guide as G
from sr_pipeline.ui import pages as P

st.set_page_config(page_title="AI Systematic Review", page_icon="📚", layout="wide")
P.PAGES["home"] = st.Page(G.page_home, title="Home", icon="🏠", default=True)
P.PAGES["guide"] = st.Page(G.page_guide, title="Guided review", icon="🧭", url_path="review")
P.PAGES["runs"] = st.Page(P.page_runs, title="Past reviews", icon="🗂️", url_path="runs")
P.PAGES["detail"] = st.Page(P.page_detail, title="Run detail", icon="🔎", url_path="run")
P.PAGES["live"] = st.Page(P.page_live, title="Live run (expert)", icon="▶️", url_path="live")
P.PAGES["check"] = st.Page(P.page_check, title="Human check (all stages)", icon="✅",
                           url_path="check")
P.PAGES["estimator"] = st.Page(P.page_estimator, title="Cost estimator", icon="🧮",
                               url_path="estimate")
P.PAGES["labels"] = st.Page(B.page_labels, title="Blind labels (accuracy)", icon="🏷️",
                            url_path="labels")
P.setup()
st.navigation({
    "Start": [P.PAGES["home"], P.PAGES["guide"], P.PAGES["runs"]],
    "Evaluate": [P.PAGES["labels"], P.PAGES["check"]],
    "Advanced tools": [P.PAGES["detail"], P.PAGES["live"], P.PAGES["estimator"]],
}).run()
