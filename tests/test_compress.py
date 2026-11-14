"""Part 12: context compression of retrieved passages (rag/compress.py), off by default."""

import json

from sr_pipeline import cli
from sr_pipeline.rag.compress import compress_hits, compress_text, sentences

from test_report import INJECT, QUESTION, e2e_config, e2e_llm

PASSAGE = ("The study was funded by a national grant. Participants aged 65 were randomized "
           "to walking or usual care. The weather was mild during the trial. Depressive "
           "symptoms fell in the walking group. Data are available on request.")


def test_keeps_the_related_sentences_in_order():
    out = compress_text(PASSAGE, "older adults randomized walking depressive symptoms", 2)
    assert out == ("Participants aged 65 were randomized to walking or usual care. … "
                   "Depressive symptoms fell in the walking group.")
    assert all(s in PASSAGE for s in out.split(" … "))     # verbatim: quotes stay findable


def test_short_passage_or_off_is_unchanged():
    assert compress_text("One sentence only.", "anything", 3) == "One sentence only."
    hits = [{"chunk_id": "c1", "text": PASSAGE}]
    assert compress_hits(hits, "walking", 0) is hits
    out = compress_hits(hits, "walking", 1)
    assert out[0]["chunk_id"] == "c1" and out[0]["compressed_from"] == len(PASSAGE)
    assert len(sentences(out[0]["text"])) == 1


def test_a_full_run_with_compression_on(tmp_path):
    cfg = e2e_config(tmp_path).model_copy(update={"rag_compress": True,
                                                  "rag_compress_sentences": 2})
    rid, state = cli.run(cfg, question=QUESTION, topic_id=None, auto=True,
                         ask=lambda p: "", out=lambda *a: None, llm=e2e_llm(), **INJECT)
    assert state.get("report")
    assert json.loads((cfg.runs_dir / rid / "run.json").read_text())["rag_compress"] == 2
