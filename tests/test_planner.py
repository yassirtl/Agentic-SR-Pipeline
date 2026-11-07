"""Part 3: config, protocol template, Planner, approval interrupt, audit writer."""

from datetime import date
from pathlib import Path

import pytest
import yaml
from langgraph.types import Command

from sr_pipeline.agents.planner import build_prompt, make_planner
from sr_pipeline.audit import AuditWriter, read_audit
from sr_pipeline.config import find_project_root, load_config
from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import OpenAILLM, ScriptedLLM
from sr_pipeline.protocol import Protocol, Template, load_template

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "protocols" / "template.yaml"
QUESTION = "Does exercise reduce depressive symptoms in older adults?"

DRAFT = {
    "title": "Exercise for depression in older adults",
    "pico": {"population": "adults aged 65 and over", "intervention": "structured exercise",
             "comparator": "usual care or no exercise",
             "outcomes": ["depressive symptoms", "quality of life"]},
    "inclusion_criteria": ["Participants aged 65 or older", "Exercise intervention",
                           "Reports a depression outcome"],
    "exclusion_criteria": ["Not a primary study", "Animal study"],
    "study_designs": ["randomized controlled trial"],
    "rationale": "The question names population, intervention and outcome.",
    # Only read when the template says databases: auto (the default).
    "databases": ["pubmed", "openalex"],
    "database_reason": "Health question: PubMed for clinical trials, OpenAlex for coverage.",
}


def scripted(*drafts) -> ScriptedLLM:
    return ScriptedLLM({"planner": list(drafts),
                        "coordinator": [{"next": s, "rationale": "r"} for s in STAGES]})


def graph_with_planner(llm, approve_protocol, template=None):
    template = template or load_template(TEMPLATE)
    return build_graph(llm, stages={"planner": make_planner(llm, template)},
                       approve_protocol=approve_protocol)


# --- config -------------------------------------------------------------------

def test_config_paths_are_absolute_under_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # working directory must not matter
    cfg = load_config(env={})
    assert find_project_root() == ROOT
    assert cfg.cache_dir == ROOT / ".cache" / "llm"
    assert cfg.runs_dir == ROOT / "runs"
    assert cfg.protocol_template == TEMPLATE
    # The real LLM built from the config uses that same absolute cache.
    assert cfg.make_llm(chat_factory=lambda m: None).cache_dir == cfg.cache_dir
    assert OpenAILLM(chat_factory=lambda m: None).cache_dir == cfg.cache_dir


def test_config_precedence_defaults_yaml_env(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"max_records": 50, "models": {"planner": "gpt-4o"},
                                    "runs_dir": "out"}))
    cfg = load_config(path, env={"SR_MAX_RECORDS": "20", "SR_MODEL_EXTRACTION": "gpt-4o",
                                 "SR_PROJECT_ROOT": str(tmp_path)})
    assert cfg.max_records == 20                       # env beats yaml
    assert cfg.model_for("planner") == "gpt-4o"        # yaml kept
    assert cfg.model_for("extraction") == "gpt-4o"     # env merged in
    assert cfg.model_for("screening") == "gpt-4o-mini"  # default
    assert cfg.runs_dir == tmp_path.resolve() / "out"  # relative -> project root


# --- template + planner ---------------------------------------------------------

def test_template_is_valid():
    t = load_template(TEMPLATE)
    assert t.databases == "auto"
    # Part 9b: last 5 years by default (start computed from this year).
    assert t.years.recent == 5 and t.years.start == date.today().year - 5


def test_planner_produces_a_valid_protocol_with_template_scope():
    template = Template(databases=["pubmed"], years={"start": 2010, "end": 2024},
                        pico={"population": "adults aged 65 and over"})
    llm = scripted(DRAFT)
    update = make_planner(llm, template)({"question": QUESTION, "completed_stages": []})

    p = Protocol.model_validate(update["protocol"])
    assert p.question == QUESTION
    assert p.pico.outcomes[0] == "depressive symptoms"
    assert p.databases == ["pubmed"]                 # scope from the template...
    assert (p.years.start, p.years.end) == (2010, 2024)
    assert update["completed_stages"] == ["planner"]
    entry = update["audit_log"][0]
    assert entry["event"] == "protocol drafted" and entry["round"] == 1 and "cost_usd" in entry

    prompt = llm.calls[0][1]                         # ...and the user's hints reach the LLM
    assert QUESTION in prompt and "adults aged 65 and over" in prompt


def test_prompt_has_no_hint_lines_for_empty_template():
    assert "required by the user" not in build_prompt(QUESTION, Template())


# --- approval interrupt -----------------------------------------------------------

def test_interrupt_pauses_then_resumes_with_edits():
    llm = scripted(DRAFT)
    graph = graph_with_planner(llm, approve_protocol=True)
    cfg = run_config("approve")

    first = graph.invoke({"question": QUESTION}, cfg)

    # Paused right after the planner; nothing after it has run.
    snapshot = graph.get_state(cfg)
    assert snapshot.next == ("protocol_approval",)
    assert first["completed_stages"] == ["planner"]
    (intr,) = first["__interrupt__"]
    assert intr.value["protocol"]["title"] == DRAFT["title"]

    final = graph.invoke(Command(resume={"edits": {"years": {"start": 2018},
                                                   "pico": {"comparator": "any"}}}), cfg)

    p = final["protocol"]
    assert p["years"] == {"start": 2018, "end": None, "end_date": None,
                          "recent": 5}  # nested merge; an explicit start wins over recent
    assert p["pico"]["comparator"] == "any"
    assert p["pico"]["population"] == DRAFT["pico"]["population"]  # untouched
    assert final["completed_stages"] == list(STAGES)          # run continued to the end
    assert len([c for c in llm.calls if c[0] == "planner"]) == 1  # resume did not re-call the LLM

    approval = next(e for e in final["audit_log"] if e["node"] == "protocol_approval")
    assert approval["edited_fields"] == ["pico", "years"]


def test_invalid_edits_ask_again():
    graph = graph_with_planner(scripted(DRAFT), approve_protocol=True)
    cfg = run_config("bad-edit")
    graph.invoke({"question": QUESTION}, cfg)

    again = graph.invoke(Command(resume={"edits": {"years": {"start": 2030, "end": 2020}}}), cfg)
    (intr,) = again["__interrupt__"]
    assert "after years.end" in intr.value["error"]
    assert graph.get_state(cfg).next == ("protocol_approval",)

    final = graph.invoke(Command(resume=True), cfg)
    assert final["protocol"]["years"]["start"] == date.today().year - 5   # template value
    assert final["completed_stages"] == list(STAGES)
    approval = next(e for e in final["audit_log"] if e["node"] == "protocol_approval")
    assert approval["attempts"] == 2


def test_auto_mode_does_not_pause():
    final = graph_with_planner(scripted(DRAFT), approve_protocol=False).invoke(
        {"question": QUESTION}, run_config("auto"))
    assert "__interrupt__" not in final
    assert final["completed_stages"] == list(STAGES)
    assert final["protocol"]["title"] == DRAFT["title"]


# --- audit writer -------------------------------------------------------------------

def test_audit_writer_appends_only_new_entries(tmp_path):
    graph = graph_with_planner(scripted(DRAFT), approve_protocol=True)
    cfg = run_config("audit")
    writer = AuditWriter(tmp_path, "run1")

    paused = graph.invoke({"question": QUESTION}, cfg)
    n1 = writer.sync(paused["audit_log"])
    final = graph.invoke(Command(resume=True), cfg)
    AuditWriter(tmp_path, "run1").sync(final["audit_log"])  # a new writer resumes the count

    lines = read_audit(tmp_path / "run1" / "audit.jsonl")
    assert n1 == len(paused["audit_log"]) > 0
    assert len(lines) == len(final["audit_log"])            # no duplicates
    assert [line["node"] for line in lines] == [e["node"] for e in final["audit_log"]]
    assert all(line["run_id"] == "run1" for line in lines)

    writer.save_state(final)
    assert (tmp_path / "run1" / "state.json").exists()


@pytest.mark.parametrize("answer", ["yes", {"approve": True}, {"edits": "years"}])
def test_unrecognised_answer_asks_again(answer):
    graph = graph_with_planner(scripted(DRAFT), approve_protocol=True)
    cfg = run_config(f"odd-{answer!r}")
    graph.invoke({"question": QUESTION}, cfg)
    again = graph.invoke(Command(resume=answer), cfg)
    assert "unrecognised answer" in again["__interrupt__"][0].value["error"]

    final = graph.invoke(Command(resume={"edits": {}}), cfg)  # empty edits = approve
    assert final["completed_stages"] == list(STAGES)


# --- Part 4b: planner stays on the question, database choice ----------------------

def test_prompt_keeps_outcomes_from_the_question_and_asks_for_a_control():
    prompt = build_prompt(QUESTION, Template())
    assert "Outcomes: only the ones the question names, primary outcome first" in prompt
    assert "Do NOT add secondary outcomes unless the user listed them" in prompt
    assert "never as a direction or result" in prompt
    assert "do not add a condition, severity or setting" in prompt
    assert "write a control" in prompt and "'none' only" in prompt


def test_auto_databases_chosen_by_planner_and_openalex_always_kept():
    llm = scripted({**DRAFT, "databases": ["pubmed"]})          # LLM forgot openalex
    update = make_planner(llm, Template())({"question": QUESTION, "completed_stages": []})
    p = update["protocol"]
    assert p["databases"] == ["openalex", "pubmed"]             # added back by code
    assert p["database_reason"] == DRAFT["database_reason"]
    assert update["audit_log"][0]["databases"] == ["openalex", "pubmed"]
    assert "Choose the databases" in llm.calls[0][1]


def test_explicit_database_list_wins_over_planner():
    llm = scripted(DRAFT)  # the draft's own choice is ignored
    update = make_planner(llm, Template(databases=["arxiv"]))(
        {"question": QUESTION, "completed_stages": []})
    assert update["protocol"]["databases"] == ["arxiv"]
    assert update["protocol"]["database_reason"] == "set by the user in the protocol template"
    prompt = llm.calls[0][1]
    assert "Choose the databases" not in prompt and "databases arxiv" in prompt


def test_human_can_edit_the_database_choice_at_approval():
    graph = graph_with_planner(scripted(DRAFT), approve_protocol=True)
    cfg = run_config("db-edit")
    first = graph.invoke({"question": QUESTION}, cfg)
    shown = first["__interrupt__"][0].value["protocol"]
    assert shown["databases"] == ["pubmed", "openalex"] and shown["database_reason"]
    final = graph.invoke(Command(resume={"edits": {"databases": ["openalex"]}}), cfg)
    assert final["protocol"]["databases"] == ["openalex"]
