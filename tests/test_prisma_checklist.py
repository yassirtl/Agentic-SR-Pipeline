"""Part 12: PRISMA 2020 checklist validator, on a full fake-LLM run's article."""

import json

from sr_pipeline import cli
from sr_pipeline.reporting import prisma_checklist as PC

from test_report import INJECT, QUESTION, e2e_config, e2e_llm


def test_checklist_on_a_full_run(tmp_path):
    cfg = e2e_config(tmp_path)
    rid, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True, ask=lambda p: "",
                         out=lambda *a: None, llm=e2e_llm(), **INJECT)
    report = cfg.runs_dir / rid / "report"
    data = json.loads((report / "prisma_checklist.json").read_text())
    items = {i["item"]: i for i in data["items"]}
    assert data["summary"]["items"] == len(items) == 42
    assert items["1"]["status"] == "reported"             # ": a systematic review" title
    assert items["24c"]["status"] == "reported"           # amendments from the audit
    assert items["25"]["status"] == items["26"]["status"] == "not reported"   # the authors
    assert items["13f"]["where"] == "not done by the pipeline"
    md = (report / "manuscript.md").read_text()
    assert "## Appendix D. PRISMA 2020 checklist (checked by code)" in md
    assert md.splitlines()[0].endswith(": a systematic review")


def test_missing_sections_are_reported_missing():
    items = {i["item"]: i for i in PC.check("# A review\n\nText only.", {})}
    assert items["1"]["status"] == "not reported"
    assert items["3"]["status"] == "not reported" and items["3"]["where"] == "no Introduction"
