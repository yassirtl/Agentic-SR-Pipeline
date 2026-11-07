"""Part 2b: LLM cache + cost, latest() per paper, coordinator fallback on LLM errors."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from sr_pipeline.graph import build_graph, run_config
from sr_pipeline.guardrail import STAGES
from sr_pipeline.llm import OpenAILLM, ScriptedLLM, compute_cost
from sr_pipeline.state import latest, stage_round


class Answer(BaseModel):
    label: str


class FakeChat:
    """Mimics ChatOpenAI.with_structured_output(..., include_raw=True)."""

    def __init__(self, model, usage=(1000, 500), parsing_error=None):
        self.model = model
        self.usage = usage
        self.parsing_error = parsing_error
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw, "usage needs the raw message"
        chat = self

        class Runnable:
            def invoke(self, prompt):
                chat.invocations += 1
                tin, tout = chat.usage
                raw = AIMessage(content="", usage_metadata={
                    "input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout})
                if chat.parsing_error:
                    return {"raw": raw, "parsed": None, "parsing_error": chat.parsing_error}
                return {"raw": raw, "parsed": schema(label=f"answer to: {prompt}"),
                        "parsing_error": None}

        return Runnable()


@pytest.fixture
def llm_and_clients(tmp_path):
    clients = {}

    def factory(model):
        clients[model] = FakeChat(model)
        return clients[model]

    llm = OpenAILLM(models={"extraction": "gpt-4o"}, cache_dir=tmp_path / "llm",
                    chat_factory=factory)
    return llm, clients


# --- Fix 1: cache + cost -----------------------------------------------------

def test_cache_hit_skips_the_model(llm_and_clients, tmp_path):
    llm, clients = llm_and_clients

    first, u1 = llm.call_with_usage(Answer, "Q1", agent="screening")
    second, u2 = llm.call_with_usage(Answer, "Q1", agent="screening")

    assert second == first
    assert clients["gpt-4o-mini"].invocations == 1  # second call never reached the model
    assert (u1.cached, u2.cached) == (False, True)
    assert u2.cost_usd == 0.0
    assert len(list((tmp_path / "llm").glob("*.json"))) == 1

    # Any change in the prompt is a different key -> a real call.
    llm.call_with_usage(Answer, "Q2", agent="screening")
    assert clients["gpt-4o-mini"].invocations == 2


def test_cost_from_usage_and_client_per_model(llm_and_clients):
    llm, clients = llm_and_clients

    _, u = llm.call_with_usage(Answer, "Q", agent="screening")
    assert (u.model, u.input_tokens, u.output_tokens) == ("gpt-4o-mini", 1000, 500)
    assert u.cost_usd == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1e6)

    # Per-agent model: extraction uses gpt-4o, priced differently.
    _, u = llm.call_with_usage(Answer, "Q", agent="extraction")
    assert u.model == "gpt-4o"
    assert u.cost_usd == pytest.approx((1000 * 2.50 + 500 * 10.00) / 1e6)

    # One client per model, reused across calls.
    llm.call_with_usage(Answer, "Q3", agent="screening")
    assert set(clients) == {"gpt-4o-mini", "gpt-4o"}

    assert compute_cost("unknown-model", 10, 10) is None


def test_parsing_error_raises_and_is_not_cached(tmp_path):
    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m: FakeChat(m, parsing_error="bad json"))
    with pytest.raises(ValueError, match="bad json"):
        llm.call_with_usage(Answer, "Q", agent="screening")
    assert not list(tmp_path.glob("*.json"))


# --- Fix 2: latest() per paper -------------------------------------------------

def test_latest_keeps_last_round_per_paper():
    history = [
        {"paper_id": "p1", "round": 1, "decision": "exclude"},
        {"paper_id": "p2", "round": 1, "decision": "include"},
        {"paper_id": "p1", "round": 2, "decision": "include"},
    ]
    assert latest(history) == [
        {"paper_id": "p1", "round": 2, "decision": "include"},
        {"paper_id": "p2", "round": 1, "decision": "include"},
    ]
    assert len(history) == 3  # history untouched


def test_stage_round_counts_previous_runs():
    done = ["planner", "search_strategy", "retrieval", "ta_screening"]
    assert stage_round("ft_screening", done) == 1
    assert stage_round("ta_screening", done) == 2


# --- Fix 3: coordinator never crashes ------------------------------------------

def test_coordinator_llm_errors_still_complete_the_review():
    llm = ScriptedLLM({"coordinator": [ValueError("bad JSON")] * len(STAGES)})
    final = build_graph(llm).invoke({"question": "Q"}, run_config("err"))

    assert final["completed_stages"] == list(STAGES)
    errors = [e for e in final["audit_log"] if e.get("outcome") == "llm_error"]
    assert len(errors) == len(STAGES)
    assert errors[0]["substituted"] == "planner"
    assert "bad JSON" in errors[0]["error"]


def test_llm_errors_count_toward_the_budget():
    llm = ScriptedLLM({"coordinator": [RuntimeError("API down")] * 10})
    final = build_graph(llm, max_steps=3).invoke({"question": "Q"}, run_config("down"))

    assert final["steps_taken"] == 3
    assert final["completed_stages"] == list(STAGES[:3])
    assert final["audit_log"][-1]["event"] == "budget exhausted"


def test_coordinator_entry_records_usage():
    llm = ScriptedLLM({"coordinator": [{"next": s, "rationale": "r"} for s in STAGES]})
    final = build_graph(llm).invoke({"question": "Q"}, run_config("usage"))
    entry = next(e for e in final["audit_log"] if e["node"] == "coordinator")
    assert {"model", "input_tokens", "output_tokens", "cost_usd", "cached"} <= entry.keys()
