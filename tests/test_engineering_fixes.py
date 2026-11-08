"""Part 10b: engineering-topic fixes from the T2 run (runs/20261108-114756-7b0e80).

(1) setting block (PICOS) in the search, fixed by the topic's search_terms; (2) evidence
terms on a 'yes'; (3) absence at full text = not met; (4) arXiv DOI fails peer review;
(5) CLI: one progress line per stage, one line for API errors.
"""

import httpx
import openai
import pytest

from sr_pipeline import cli
from sr_pipeline.agents.screening import (check_verdicts, criterion_query, evidence_terms,
                                          ft_counts, has_evidence_term, make_ta_screening,
                                          screen_record, ta_evidence)
from sr_pipeline.agents.search_strategy import build_prompt, make_search_strategy
from sr_pipeline.evaluation.topics import load_topic, seed_protocol
from sr_pipeline.llm import ScriptedLLM, api_error_message, is_fatal
from sr_pipeline.protocol import Protocol
from sr_pipeline.reporting.prisma import flow_counts

from test_search_refine import FakeCounts, http_for

# --- (1) setting block ---------------------------------------------------------------

DEVICE = ["microcontroller", "FPGA", "wearable"]
PROTOCOL = {
    "question": "Which lightweight ML models detect abnormal ECG on constrained devices?",
    "title": "t", "rationale": "r",
    "pico": {"population": "ECG recordings", "intervention": "lightweight ML models",
             "comparator": "none", "outcomes": ["accuracy"],
             "setting": "resource-constrained devices (microcontroller, FPGA)"},
    "inclusion_criteria": [], "exclusion_criteria": [], "study_designs": [],
    "databases": ["openalex", "pubmed"], "years": {"start": 2021, "end": None},
    "languages": ["English"],
}
POP = {"name": "population", "role": "population", "keywords": ["ECG", "electrocardiogram"]}
INT = {"name": "intervention", "role": "intervention", "keywords": ["machine learning"]}
LLM_SETTING = {"name": "setting", "role": "setting", "keywords": ["device"]}


def plan(*blocks):
    return {"concepts": list(blocks), "rationale": "r"}


def test_setting_is_in_the_search_prompt_with_its_rule():
    prompt = build_prompt(PROTOCOL)
    assert "Setting: resource-constrained devices (microcontroller, FPGA)\n" in prompt
    assert "Add a setting block (role setting)" in prompt
    no_setting = {**PROTOCOL, "pico": {**PROTOCOL["pico"], "setting": ""}}
    assert "Setting:" not in build_prompt(no_setting)


def test_fixed_setting_block_replaces_the_llm_one_in_every_attempt():
    # Attempt 1 is too broad, attempt 2 (refinement) drops the setting block: code puts
    # the fixed block back both times, with the topic's keywords.
    def hits(q):
        return 50 if "fpga" in q.lower() and "classif" in q.lower() else 100_000
    protocol = {**PROTOCOL, "search_terms": {"setting": DEVICE}}
    llm = ScriptedLLM({"search_strategy": [plan(POP, INT, LLM_SETTING),
                                           {**plan(POP, {**INT, "keywords": ["classifier"]}),
                                            "removed_terms": [], "changed_terms": []}]})
    node = make_search_strategy(llm, http_for(FakeCounts(hits)), max_records=1000,
                                min_hits=20, env={})
    s = node({"protocol": protocol, "completed_stages": ["planner"]})["search_strategy"]
    assert len(s["attempts"]) == 2
    for a in s["attempts"]:
        setting = [c for c in a["concepts"] if c["role"] == "setting"]
        assert len(setting) == 1 and setting[0]["keywords"] == DEVICE and setting[0]["fixed"]
        assert [c["role"] for c in a["concepts"]] == ["population", "intervention", "setting"]
        assert "microcontroller" in a["queries"]["openalex"]
        assert not a["setting_block_missing"]
    assert s["attempts"][0]["llm_blocks_replaced"][0]["keywords"] == ["device"]
    assert s["chosen_attempt"] == 2 and s["counts"] == {"openalex": 50, "pubmed": 50}
    # the refine prompt tells the LLM the block is fixed by code
    assert "[setting] (fixed, added by code)" in llm.calls[1][1]


def test_empty_search_terms_remove_the_block_and_missing_setting_is_flagged():
    llm = ScriptedLLM({"search_strategy": [plan(POP, INT, {**LLM_SETTING, "role": "outcome"})]})
    protocol = {**PROTOCOL, "search_terms": {"outcome": []}}
    s = make_search_strategy(llm)({"protocol": protocol,
                                   "completed_stages": ["planner"]})["search_strategy"]
    assert [c["role"] for c in s["concepts"]] == ["population", "intervention"]
    assert s["attempts"][0]["setting_block_missing"]          # pico.setting, no block


def test_t2_topic_seeds_search_terms_and_evidence_terms():
    topic = load_topic("T2")
    draft = Protocol(**{**PROTOCOL, "pico": PROTOCOL["pico"]}).model_dump(mode="json")
    protocol, audit = seed_protocol(draft, topic)
    assert "microcontroller" in protocol["search_terms"]["setting"]
    assert {"search_terms", "criterion_evidence"} <= set(audit["seeded_fields"])
    ev = evidence_terms(protocol)
    assert set(ev) == {"I3", "I4"} and "FPGA" in ev["I3"] and "latency" in ev["I4"]
    # evidence terms also steer full-text retrieval for that criterion
    text = protocol["inclusion_criteria"][2]
    assert "microcontroller" in criterion_query(protocol, "I3", text)


def test_protocol_drops_evidence_for_unknown_criteria():
    p = Protocol(**{**PROTOCOL, "inclusion_criteria": ["On a device"],
                    "criterion_evidence": [{"criterion": "On a device", "terms": [" FPGA ", ""]},
                                           {"criterion": "gone", "terms": ["x"]}]})
    assert [e.model_dump() for e in p.criterion_evidence] == \
        [{"criterion": "On a device", "terms": ["FPGA"]}]


# --- (2) evidence terms ------------------------------------------------------------------

@pytest.mark.parametrize("quote", ["deployed on two FPGAs", "a 128kB model",
                                   "an ARM Cortex-M4 microcontroller", "low-power wearable",
                                   "latency of 3 ms per beat"])
def test_evidence_term_found(quote):
    assert has_evidence_term(quote, ["FPGA", "kB", "microcontroller", "low power", "ms",
                                     "wearable"])


@pytest.mark.parametrize("quote", ["prior knowledge of the signal", "social factors",
                                   "a powerful model", "a basic classifier",
                                   "accuracy, precision and recall"])
def test_evidence_term_needs_a_whole_word(quote):
    assert not has_evidence_term(quote, ["edge", "SoC", "power", "ASIC", "latency"])


T2_TEXT = ("We applied the MIT-BIH arrhythmia dataset. The model reached 98% accuracy, "
           "precision and recall. It was deployed on an STM32 microcontroller with 64 kB of RAM.")
EVIDENCE = {"I1": ["microcontroller", "FPGA"], "I2": ["memory", "kB", "latency"]}


def v(cid, met, quote=""):
    return {"criterion": cid, "met": met, "quote": quote, "chunk_id": ""}


def test_yes_without_an_evidence_term_is_unclear():
    raw = [v("D", "yes", "We applied the MIT-BIH arrhythmia dataset"),
           v("I1", "yes", "We applied the MIT-BIH arrhythmia dataset"),
           v("I2", "yes", "accuracy, precision and recall")]
    out, flags = check_verdicts(raw, ["D", "I1", "I2"], T2_TEXT, "a", evidence=EVIDENCE)
    assert [x["met"] for x in out] == ["yes", "unclear", "unclear"]
    assert out[1]["met_claimed"] == "yes"
    assert {"no_evidence_term_a_I1", "no_evidence_term_a_I2"} <= set(flags)
    good = [v("I1", "yes", "deployed on an STM32 microcontroller"), v("I2", "yes", "64 kB of RAM")]
    out, flags = check_verdicts(good, ["I1", "I2"], T2_TEXT, "a", evidence=EVIDENCE)
    assert [x["met"] for x in out] == ["yes", "yes"] and not flags


# --- (3) absence at full text --------------------------------------------------------

FT_PROTOCOL = {"question": "q", "pico": {}, "study_designs": [],
               "inclusion_criteria": ["ECG classifier", "Evaluated on a constrained device",
                                      "Peer-reviewed journal or conference publication"],
               "exclusion_criteria": [],
               "criterion_evidence": [{"criterion": "Evaluated on a constrained device",
                                       "terms": ["microcontroller", "FPGA"]}]}
PAPER = {"paper_id": "p1", "title": "CNN heartbeat classification",
         "abstract": "A CNN classifies heartbeats of the MIT-BIH arrhythmia database."}
FT_EVIDENCE = ta_evidence(PAPER) + "\nFull-text passages:\n[p1::1] We applied the MIT-BIH dataset."


def ans(i2_met, i2_quote="", i3="unclear"):
    i3_quote = "A CNN classifies heartbeats" if i3 == "yes" else ""
    return {"comparison": "", "reason": "r", "unresolved": False,
            "verdicts": [v("D", "yes", "A CNN classifies heartbeats"),
                         v("I1", "yes", "A CNN classifies heartbeats"),
                         v("I2", i2_met, i2_quote), v("I3", i3, i3_quote)]}


def test_ft_unclear_inclusion_with_full_text_is_an_agreed_exclusion():
    # Both screeners answer the device criterion 'yes' with a quote naming no device (the
    # T2 case): downgraded to unclear, then not met at full text. Same reason for A and B:
    # agreement, no adjudicator call. Peer review (I3) unclear does not exclude.
    bad = ans("yes", "We applied the MIT-BIH dataset")
    llm = ScriptedLLM({"screener_a": [bad], "screener_b": [bad]})
    item, _ = screen_record(llm, FT_PROTOCOL, PAPER, FT_EVIDENCE, stage="ft", rnd=1, mode="ft")
    assert (item["final"], item["criterion"], item["resolved_by"]) == ("exclude", "I2",
                                                                        "agreement")
    assert [c[0] for c in llm.calls] == ["screener_a", "screener_b"]
    assert "ft_absence_a_I2" in item["flags"] and "no_evidence_term_b_I2" in item["flags"]
    i2 = next(x for x in item["b"]["verdicts"] if x["criterion"] == "I2")
    assert i2["absence"] and i2["met"] == "no"
    i3 = next(x for x in item["b"]["verdicts"] if x["criterion"] == "I3")
    assert i3["met"] == "unclear" and "absence" not in i3
    assert ft_counts([{**item, "fulltext_status": "pdf"}])["ft_absence"] == 1


def test_ft_supported_yes_includes_and_abstract_only_never_uses_absence():
    good = ans("yes", "We applied the MIT-BIH dataset", i3="yes")
    ev = FT_EVIDENCE + " It runs on an FPGA."
    good["verdicts"][2]["quote"] = "It runs on an FPGA"
    llm = ScriptedLLM({"screener_a": [good], "screener_b": [good]})
    item, _ = screen_record(llm, FT_PROTOCOL, PAPER, ev, stage="ft", rnd=1, mode="ft")
    assert item["final"] == "include"
    # no full text (ft_abstract): unclear stays unclear, B includes, A (lenient there) too
    silent = ans("unclear")
    llm = ScriptedLLM({"screener_a": [silent], "screener_b": [silent]})
    item, _ = screen_record(llm, FT_PROTOCOL, PAPER, ta_evidence(PAPER), stage="ft", rnd=1,
                            mode="ft_abstract")
    assert item["final"] == "include" and not any("ft_absence" in f for f in item["flags"])


def test_ft_absence_split_goes_to_the_adjudicator_whose_absence_exclusion_stands():
    a = ans("unclear")
    b = ans("yes", "It runs on an FPGA")
    llm = ScriptedLLM({"screener_a": [a], "screener_b": [b], "adjudicator": [ans("unclear")]})
    item, _ = screen_record(llm, FT_PROTOCOL, PAPER, FT_EVIDENCE + " It runs on an FPGA.",
                            stage="ft", rnd=1, mode="ft")
    assert (item["final"], item["criterion"], item["resolved_by"]) == ("exclude", "I2",
                                                                        "adjudicator")


# --- (4) arXiv rule --------------------------------------------------------------------

def test_arxiv_doi_fails_peer_review_without_llm_call():
    paper = {**PAPER, "doi": "10.48550/arxiv.2207.06408"}
    llm = ScriptedLLM({})
    item, usages = screen_record(llm, FT_PROTOCOL, paper, ta_evidence(paper), stage="ta",
                                 rnd=1)
    assert llm.calls == [] and usages == []
    assert (item["final"], item["criterion"], item["resolved_by"]) == ("exclude", "I3",
                                                                        "arxiv_rule")
    assert item["quote"] == paper["doi"]


def test_arxiv_rule_needs_a_peer_review_criterion():
    protocol = {**FT_PROTOCOL, "inclusion_criteria": FT_PROTOCOL["inclusion_criteria"][:2]}
    ok = ans("yes", "A CNN classifies heartbeats")
    ok["verdicts"] = ok["verdicts"][:3]
    llm = ScriptedLLM({"screener_a": [ok], "screener_b": [ok]})
    paper = {**PAPER, "doi": "https://doi.org/10.48550/arXiv.1"}
    item, _ = screen_record(llm, protocol, paper, ta_evidence(paper), stage="ta", rnd=1)
    assert item["resolved_by"] != "arxiv_rule"


def test_automation_box_counts_title_and_arxiv_rules_per_criterion():
    papers = [{"paper_id": "rev", "title": "ECG on FPGAs: a systematic review",
               "abstract": "We review."},
              {"paper_id": "arx", "title": "TinyML ECG", "abstract": "x",
               "doi": "10.48550/arxiv.2412.05583"},
              {"paper_id": "ok", "title": "ECG classifier", "abstract": "A CNN classifies "
               "heartbeats on an FPGA."}]
    yes = {"comparison": "", "reason": "r", "verdicts": [
        v("D", "yes", "A CNN classifies heartbeats"), v("I1", "yes", "A CNN classifies"),
        v("I2", "yes", "on an FPGA"), v("I3", "unclear")]}
    llm = ScriptedLLM({"screener_a": [yes], "screener_b": [yes]})
    out = make_ta_screening(llm, workers=1)({"protocol": FT_PROTOCOL, "unique_records": papers,
                                             "completed_stages": ["retrieval"]})
    ta = out["prisma_counts"]["title_abstract"]
    assert ta["automation_by_criterion"] == {"D": 1, "I3": 1} and ta["arxiv_rule"] == 1
    prisma = {"identification": {"identified": 3, "not_retrieved": 0, "duplicates_removed": 0,
                                 "databases": {}},
              "title_abstract": ta,
              "full_text": {"reports_sought": 1, "reports_excluded": 0, "studies_included": 1}}
    c = flow_counts(prisma, FT_PROTOCOL)
    assert c["numbers"]["automation"] == 2
    assert all(c["checks"].values()), c["checks"]
    assert "arXiv preprint, not peer reviewed (n\xa0=\xa01)" in " ".join(c["removed"]["lines"])


# --- (5) CLI and API errors --------------------------------------------------------------

def _rate_limit(code):
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com"))
    return openai.RateLimitError("Error code: 429", response=response,
                                 body={"code": code, "message": "m"})


def test_no_credits_is_fatal_and_stops_screening():
    exc = _rate_limit("insufficient_quota")
    assert is_fatal(exc) and not is_fatal(_rate_limit("rate_limit_exceeded"))
    assert not is_fatal(ValueError("bad json"))
    assert "no credits" in (api_error_message(exc) or "")
    assert api_error_message(ValueError("x")) is None
    llm = ScriptedLLM({"screener_a": [exc]})
    with pytest.raises(openai.RateLimitError):
        screen_record(llm, FT_PROTOCOL, PAPER, ta_evidence(PAPER), stage="ta", rnd=1)
    # an ordinary error still leaves the record unresolved (unchanged)
    llm = ScriptedLLM({"screener_a": [ValueError("bad json")]})
    item, _ = screen_record(llm, FT_PROTOCOL, PAPER, ta_evidence(PAPER), stage="ta", rnd=1)
    assert item["final"] == "unresolved"


def test_progress_line_per_stage_and_none_for_coordinator():
    line = cli.progress_line("ta_screening", {"audit_log": [
        {"event": "title/abstract screening", "records_screened": 100, "included": 40,
         "records_excluded": 60, "cost_usd": 0.0123}]})
    assert line is not None and line.endswith("ta_screening: title/abstract screening, 100 screened, 40 included, "
                         "60 excluded, $0.0123")
    assert cli.progress_line("coordinator", {"audit_log": [{"event": "x"}]}) is None
    assert cli.progress_line("ta_conflicts", None) is None


def test_main_prints_one_line_for_an_api_error(monkeypatch, capsys):
    def boom(*_, **__):
        exc = _rate_limit("insufficient_quota")
        setattr(exc, "run_id", "20261108-000000-abcdef")
        raise exc
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "run", boom)
    assert cli.main(["--topic", "T2"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0 and "no credits" in err
    assert err.endswith("sr-review --resume 20261108-000000-abcdef")
