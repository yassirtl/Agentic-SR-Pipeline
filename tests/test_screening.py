"""Part 6: dual screening, adjudication, human / auto resolution, kappa, full text with RAG.

The scripted LLM answers from the prompt (each paper's title holds a marker), so the
answers do not depend on call order and the same script works with several threads.
"""

import json

from langgraph.types import Command

from sr_pipeline.agents.screening import (
    apply_resolutions, cohen_kappa, criteria, make_conflict_review, make_ft_screening,
    make_ta_screening, quote_found, screen_record, ta_evidence, ta_included)
from sr_pipeline.coordinator import summarize_state
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import OpenAILLM, ScriptedLLM
from sr_pipeline.rag import FakeEmbedder
from sr_pipeline.sources.fulltext import FullText, Section, strip_references

PROTOCOL = {
    "question": "Does exercise reduce depressive symptoms in older adults?",
    "pico": {"population": "older adults", "intervention": "exercise",
             "comparator": "no intervention, usual care or any control",
             "outcomes": ["depressive symptoms"]},
    "inclusion_criteria": ["Participants aged 60 years or older",
                           "Depressive symptoms measured"],
    "exclusion_criteria": ["Not an exercise intervention"],
    "study_designs": ["randomized controlled trial"],
}


def paper(pid: str, marker: str, abstract: str = "Older adults aged 65 were randomized to "
          "walking or usual care; depressive symptoms were measured with the GDS-15.") -> dict:
    return {"paper_id": pid, "title": f"{marker} trial of walking", "abstract": abstract,
            "round": 1}


PAPERS = [paper("p1", "AGREE_IN"), paper("p2", "AGREE_OUT"), paper("p3", "SPLIT_IN"),
          paper("p4", "SPLIT_UNRES"), paper("p5", "SPLIT_OUT", abstract="")]

QUOTE = "Older adults aged 65 were randomized"


CRITERIA = ["D", "I1", "I2", "E1"]  # deciding order of PROTOCOL's criteria


def dec(decision, criterion="all", quote=QUOTE, reason="r", chunk_id=""):
    """Scripted answer in the per-criterion format (Part 6b). 'include': every D / I
    criterion met, no exclusion applies. 'exclude' on C: C fails with ``quote``; with an
    empty quote C is 'unclear' (a strict screener's missing information). 'unresolved':
    adjudicator only."""
    ok = lambda cid: "no" if cid.startswith("E") else "yes"
    bad = lambda cid: "yes" if cid.startswith("E") else "no"
    verdicts = []
    for cid in CRITERIA:
        if decision == "exclude" and cid == criterion:
            v = {"criterion": cid, "met": bad(cid) if quote else "unclear", "quote": quote}
        else:
            v = {"criterion": cid, "met": ok(cid), "quote": QUOTE if ok(cid) == "yes" else ""}
        verdicts.append({**v, "chunk_id": chunk_id})
    return {"verdicts": verdicts, "reason": reason, "unresolved": decision == "unresolved"}


def screener(style: str):
    def answer(prompt: str):
        if "AGREE_IN" in prompt:
            return dec("include")
        if "AGREE_OUT" in prompt:
            return dec("exclude", "E1", quote="walking")
        # Split papers: A (strict) excludes on missing information, B (lenient) includes.
        return dec("exclude", "I2", quote="") if style == "a" else dec("include")
    return answer


def adjudicator(prompt: str):
    if "SPLIT_IN" in prompt:
        return dec("include", "I2", reason="abstract silent: full text decides")
    if "SPLIT_OUT" in prompt:
        return dec("exclude", "I1", quote="invented words never in the abstract")
    return dec("unresolved", "I2", quote="")


def script(**extra):
    return ScriptedLLM({"screener_a": screener("a"), "screener_b": screener("b"),
                        "adjudicator": adjudicator, **extra})


APPEND = {"ta_decisions", "ft_decisions", "completed_stages", "audit_log"}


def merge(state: dict, update: dict) -> dict:
    """Apply a node's update like LangGraph: append-only fields are concatenated."""
    return {**state, **{k: state.get(k, []) + v if k in APPEND else v
                        for k, v in update.items()}}


def ta_state(**kw):
    return {"question": PROTOCOL["question"], "protocol": PROTOCOL, "unique_records": PAPERS,
            "completed_stages": ["planner", "search_strategy", "retrieval"], **kw}


# --- statistics and checks ----------------------------------------------------------

def test_cohen_kappa():
    # p_o = 0.75; p_e = 0.5*0.25 + 0.5*0.75 = 0.5 -> kappa = 0.5
    assert cohen_kappa(["in", "in", "out", "out"], ["in", "out", "out", "out"]) == 0.5
    assert cohen_kappa(["in", "out"], ["out", "in"]) == -1.0      # worse than chance
    assert cohen_kappa(["out", "out"], ["out", "out"]) == 1.0      # p_e = 1: one shared label
    assert cohen_kappa([], []) is None


def test_quote_check_ignores_case_and_punctuation_but_not_invention():
    text = "Older adults (aged >= 65) were randomized; GDS-15 measured mood."
    assert quote_found("older adults aged 65 were RANDOMIZED", text)
    assert quote_found("Older adults ... GDS-15 measured", text)
    assert quote_found("", text)  # decision on missing information
    assert not quote_found("participants were all depressed", text)


def test_criteria_are_numbered_with_design():
    assert [c for c, _ in criteria(PROTOCOL)] == ["D", "I1", "I2", "E1"]
    assert "I1: Participants aged 60" in __import__(
        "sr_pipeline.agents.screening", fromlist=["x"]).protocol_block(PROTOCOL)


# --- one record --------------------------------------------------------------------

def test_agree_disagree_adjudicate_paths():
    llm = script()
    out = {p["paper_id"]: screen_record(llm, PROTOCOL, p, ta_evidence(p), stage="ta", rnd=1)[0]
           for p in PAPERS}
    assert (out["p1"]["final"], out["p1"]["resolved_by"]) == ("include", "agreement")
    assert (out["p2"]["final"], out["p2"]["criterion"]) == ("exclude", "E1")
    assert out["p2"]["adjudicator"] is None
    assert (out["p3"]["final"], out["p3"]["resolved_by"]) == ("include", "adjudicator")
    assert (out["p4"]["final"], out["p4"]["resolved_by"]) == ("unresolved", "pending")
    # No abstract is flagged; the adjudicator's invented quote is caught, and an exclusion
    # without a verified quote is not accepted: the record goes to a human / auto-include.
    assert "no_abstract" in out["p5"]["flags"]
    assert "quote_not_found_adjudicator" in out["p5"]["flags"]
    assert "adjudicator_exclusion_without_quote" in out["p5"]["flags"]
    assert out["p5"]["final"] == "unresolved"
    # The adjudicator is called only on disagreements (p3, p4, p5).
    assert sum(a == "adjudicator" for a, _ in llm.calls) == 3
    # Strict and lenient instructions really differ.
    prompts = {a: p for a, p in llm.calls if a.startswith("screener")}
    assert "(strict)" in prompts["screener_a"] and "(lenient)" in prompts["screener_b"]


def test_two_exclusions_without_evidence_go_to_the_adjudicator():
    # Seen live: both screeners excluded a gold study for an age the abstract did not state.
    llm = ScriptedLLM({"screener_a": [dec("exclude", "I1", quote="")],
                       "screener_b": [dec("exclude", "I1", quote="all participants were 70")],
                       "adjudicator": [dec("include", "all")]})
    item, _ = screen_record(llm, PROTOCOL, PAPERS[0], ta_evidence(PAPERS[0]), stage="ta", rnd=1)
    # B's "no" quotes words that are not in the abstract: downgraded to unclear, so B
    # includes and the adjudicator decides.
    assert "unverified_b_I1" in item["flags"] and item["b"]["decision"] == "include"
    assert (item["final"], item["resolved_by"]) == ("include", "adjudicator")
    # One verified quote is enough for an agreed exclusion on the same criterion.
    llm = ScriptedLLM({"screener_a": [dec("exclude", "E1", quote="")],
                       "screener_b": [dec("exclude", "E1", quote="randomized to walking")]})
    item, _ = screen_record(llm, PROTOCOL, PAPERS[0], ta_evidence(PAPERS[0]), stage="ta", rnd=1)
    # A's E1 'unclear' (no quote) does not exclude; A excludes on nothing -> include, so
    # this is a disagreement decided by the adjudicator (none scripted -> unresolved).
    assert item["a"]["decision"] == "include" and item["final"] == "unresolved"
    llm = ScriptedLLM({"screener_a": [dec("exclude", "E1", quote="walking")],
                       "screener_b": [dec("exclude", "E1", quote="randomized to walking")]})
    item, _ = screen_record(llm, PROTOCOL, PAPERS[0], ta_evidence(PAPERS[0]), stage="ta", rnd=1)
    assert (item["final"], item["resolved_by"], item["criterion"]) == \
        ("exclude", "agreement", "E1")
    # Part 6c: two exclusions on different criteria (A: I2 missing information, B: E1)
    # are not an agreement; the adjudicator decides.
    llm = ScriptedLLM({"screener_a": [dec("exclude", "I2", quote="")],
                       "screener_b": [dec("exclude", "E1", quote="randomized to walking")],
                       "adjudicator": [dec("include")]})
    item, _ = screen_record(llm, PROTOCOL, PAPERS[0], ta_evidence(PAPERS[0]), stage="ta", rnd=1)
    assert "different_reasons" in item["flags"]
    assert (item["final"], item["resolved_by"]) == ("include", "adjudicator")


def test_llm_error_leaves_record_unresolved_never_dropped():
    llm = ScriptedLLM({"screener_a": [RuntimeError("API down")]})
    item, _ = screen_record(llm, PROTOCOL, PAPERS[0], ta_evidence(PAPERS[0]), stage="ta", rnd=1)
    assert item["final"] == "unresolved"
    assert item["flags"][0].startswith("llm_error: RuntimeError")


# --- title / abstract stage -------------------------------------------------------------

def test_ta_stage_counts_and_kappa():
    out = make_ta_screening(script(), workers=1)(ta_state())
    c = out["prisma_counts"]["title_abstract"]
    assert (c["records_screened"], c["included"], c["records_excluded"], c["unresolved"]) == \
           (5, 2, 1, 2)
    assert c["excluded_by_criterion"] == {"E1": 1}
    assert c["no_abstract"] == 1 and c["pairs"] == 5
    assert c["agreement"] == 0.4  # A and B agree on p1 and p2 only
    # A: in,out,out,out,out  B: in,out,in,in,in -> p_e = 0.2*0.8 + 0.8*0.2 = 0.32
    assert c["kappa"] == round((0.4 - 0.32) / 0.68, 4)
    assert [d["round"] for d in out["ta_decisions"]] == [1] * 5
    assert out["audit_log"][0]["llm_calls"] == 13  # 5 x (A + B) + 3 adjudications


def test_parallel_screening_gives_the_same_result():
    one = make_ta_screening(script(), workers=1)(ta_state())
    many = make_ta_screening(script(), workers=4)(ta_state())
    strip = lambda out: [(d["paper_id"], d["final"], d["resolved_by"]) for d in
                         out["ta_decisions"]]
    assert strip(one) == strip(many)


def test_auto_mode_includes_unresolved_records():
    state = merge(ta_state(), make_ta_screening(script(), workers=1)(ta_state()))
    out = make_conflict_review("ta", human=False)(state)
    assert [(d["paper_id"], d["final"], d["resolved_by"]) for d in out["ta_decisions"]] == \
        [("p4", "include", "auto-include"), ("p5", "include", "auto-include")]
    c = out["prisma_counts"]["title_abstract"]
    assert (c["included"], c["unresolved"], c["resolved_by"]["auto-include"]) == (4, 0, 2)
    # Nothing pending: the node does nothing.
    assert make_conflict_review("ta", human=False)(merge(state, out)) == {}


def test_human_answer_validation():
    pending = [{"paper_id": "p4"}, {"paper_id": "p9"}]
    assert apply_resolutions(True, pending) == {"p4": "include", "p9": "include"}
    assert apply_resolutions({"p4": "exclude", "p9": "include"}, pending)["p4"] == "exclude"
    for bad in ({"p4": "exclude"}, {"p4": "maybe", "p9": "include"}, "yes",
                {"p4": "include", "p9": "include", "zz": "include"}):
        try:
            apply_resolutions(bad, pending)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_human_resolves_conflicts_in_the_graph():
    coordinator = ScriptedLLM({"coordinator": [{"next": s, "rationale": s} for s in STAGES]})
    screening = script()
    llm = ScriptedLLM({**coordinator.script, **screening.script})
    graph = build_graph(llm, stages={"ta_screening": make_ta_screening(llm, workers=1)},
                        human_screening=True)
    cfg = run_config("screen-human")
    first = graph.invoke({"question": PROTOCOL["question"], "protocol": PROTOCOL,
                          "unique_records": PAPERS}, cfg)
    assert graph.get_state(cfg).next == ("ta_conflicts",)
    (intr,) = first["__interrupt__"]
    assert [p["paper_id"] for p in intr.value["papers"]] == ["p4", "p5"]
    screener_calls = len([a for a, _ in llm.calls if a != "coordinator"])

    again = graph.invoke(Command(resume={"p4": "maybe"}), cfg)        # invalid: asked again
    assert "invalid" in again["__interrupt__"][0].value["error"]
    final = graph.invoke(Command(resume={"p4": "exclude", "p5": "include"}), cfg)

    assert final["completed_stages"] == list(STAGES)
    latest_p4 = [d for d in final["ta_decisions"] if d["paper_id"] == "p4"]
    assert [(d["final"], d["resolved_by"]) for d in latest_p4] == [("unresolved", "pending"),
                                                                    ("exclude", "human")]
    c = final["prisma_counts"]["title_abstract"]
    assert (c["included"], c["records_excluded"], c["unresolved"]) == (3, 2, 0)
    # Resuming re-ran only the conflict node: no screening call was repeated.
    assert len([a for a, _ in llm.calls if a != "coordinator"]) == screener_calls
    assert "Title/abstract screening (round 1): 5 screened, 3 included" in \
        summarize_state(final)


def test_dropped_paper_loses_its_old_decision():
    state = merge(ta_state(), make_ta_screening(script(), workers=1)(ta_state()))
    assert {p["paper_id"] for p in ta_included(state)} == {"p1", "p3", "p4", "p5"}
    # A new retrieval no longer returns p1: its include must not reach full text.
    state["unique_records"] = [p for p in PAPERS if p["paper_id"] != "p1"]
    assert {p["paper_id"] for p in ta_included(state)} == {"p3", "p4", "p5"}


# --- full text ---------------------------------------------------------------------------

def fake_fetch(papers, http, *, contact_email=None, env=None):
    out = {}
    for p in papers:
        if p["paper_id"] == "p1":
            out["p1"] = FullText(paper_id="p1", status="pmc_xml", sections=[
                Section(title="Methods > Participants",
                        text="We enrolled 120 participants aged 60 years or older from "
                             "community centres. " * 3),
                Section(title="Methods > Outcomes",
                        text="Depressive symptoms were measured with the Geriatric "
                             "Depression Scale at baseline and 12 weeks. " * 3),
                Section(title="Discussion", text="Walking is cheap and safe. " * 20)])
        else:
            out[p["paper_id"]] = FullText(paper_id=p["paper_id"], status="abstract_only",
                                          reason="no open-access full text")
    return out


def ft_screener(prompt: str):
    if "Full-text passages" in prompt:
        return dec("include", quote="participants aged 60 years or older",
                   chunk_id="p1::1")
    return dec("include")


def test_full_text_stage_uses_passages_and_counts_reports_not_retrieved(tmp_path):
    reranks = []

    def reranker(prompt):
        reranks.append(prompt)
        return {"scores": []}  # no scores: fused order kept

    llm = ScriptedLLM({"screener_a": ft_screener, "screener_b": ft_screener,
                       "adjudicator": adjudicator, "reranker": reranker})
    ta = ScriptedLLM({"screener_a": screener("a"), "screener_b": screener("b"),
                      "adjudicator": adjudicator})
    state = ta_state(unique_records=PAPERS[:3])  # p1 include, p2 exclude, p3 include
    state = merge(state, make_ta_screening(ta, workers=1)(state))

    node = make_ft_screening(llm, http=None, embedder=FakeEmbedder(), index_dir=tmp_path / "idx",
                             workers=1, passages=2, candidates=4, chunk_words=60,
                             chunk_overlap=10, fetch=fake_fetch)
    out = node(state)
    by_id = {d["paper_id"]: d for d in out["ft_decisions"]}
    assert set(by_id) == {"p1", "p3"}                 # only title/abstract includes
    assert by_id["p1"]["fulltext_status"] == "pmc_xml"
    assert set(by_id["p1"]["passages"]) == {"I1", "I2", "D", "E1"}
    assert all(len(v) <= 2 for v in by_id["p1"]["passages"].values())
    assert "abstract_only" in by_id["p3"]["flags"] and by_id["p3"]["passages"] == {}
    assert not any(f.startswith("quote_not_found") for f in by_id["p1"]["flags"])
    assert len(reranks) == 4                           # one per criterion, p1 only

    p1_prompt = next(p for a, p in llm.calls if a == "screener_a" and "Full-text" in p)
    assert "[p1::1] [Methods > Participants" in p1_prompt
    assert "For criterion I1:" in p1_prompt

    c = out["prisma_counts"]["full_text"]
    assert (c["reports_sought"], c["reports_not_retrieved"], c["reports_assessed"]) == (2, 1, 1)
    assert (c["studies_included"], c["included_on_abstract_only"]) == (2, 1)
    assert c["not_retrieved_reasons"] == {"no open-access full text": 1}
    assert out["included"] == ["p1", "p3"]
    assert json.loads((tmp_path / "idx" / "meta.json").read_text())["papers"] == 2


# --- LLM cache under concurrency, PDF clean-up -------------------------------------------

def test_llm_cache_write_is_atomic(tmp_path):
    from types import SimpleNamespace

    from sr_pipeline.agents.screening import ScreeningDecision

    class Chat:
        def with_structured_output(self, schema, include_raw):
            return SimpleNamespace(invoke=lambda prompt: {
                "parsed": schema.model_validate(dec("include")), "parsing_error": None,
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})})

    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m: Chat())
    llm.structured_call(ScreeningDecision, "p", agent="screener_a")
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]  # no temp file left behind
    _, usage = llm.call_with_usage(ScreeningDecision, "p", agent="screener_a")
    assert usage.cached


def test_pdf_reference_list_is_dropped():
    pages = ["Methods. See References for details.", "Results", "Discussion\nReferences\n1. X",
             "2. Y"]
    assert strip_references(pages) == ["Methods. See References for details.", "Results",
                                       "Discussion\n"]
    # A heading only in the first half is not trusted.
    assert strip_references(["References\nintro", "a", "b", "c"]) == \
        ["References\nintro", "a", "b", "c"]
