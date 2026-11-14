"""Part 12: LangChain tool calling. The critic's tools, the scripted LLM running tool calls,
and OpenAILLM.call_with_tools on a fake chat model (rounds, final answer, usage, cache)."""


from langchain_core.messages import AIMessage

from sr_pipeline.agents import critic as C
from sr_pipeline.llm import OpenAILLM, ScriptedLLM

EVIDENCE = """Studies (label | first author year | design):
S1 | Ann 2020 | randomized controlled trial
S2 | Bo 2021 | randomized controlled trial
Number of studies: 2.

Outcome O1: depression (better = lower values)
Method: narrative (SWiM).
  S1 | favours intervention | between-group
  S2 | no difference | between-group
Vote count by direction of effect: favours intervention 1, no difference 1
"""
SYN = {"evidence": EVIDENCE, "narrative": [
    {"section": "O1", "text": "Both trials favoured exercise (S1, S2).",
     "studies": ["S1", "S2"], "flags": []}]}


def tools():
    return {t.name: t for t in C.critic_tools(SYN)}


def test_critic_tools_return_the_right_lines():
    t = tools()
    out = t["lookup_study"].func(label="s2")
    assert "overview: S2 | Bo 2021" in out and "O1: S2 | no difference" in out
    assert "S1" not in out
    summary = t["outcome_summary"].func(outcome_id="O1")
    assert "Vote count" in summary and "S1 |" not in summary
    assert t["outcome_summary"].func(outcome_id="O9").startswith("no outcome O9")
    assert t["check_quote"].func(text="favours intervention 1") == "found in the evidence tables"
    assert "NOT found" in t["check_quote"].func(text="favours control 2")


def test_prompt_has_sentences_and_outcomes_not_study_lines():
    p = C.critic_prompt_tools({"question": "Q?"}, SYN)
    assert "Both trials favoured exercise" in p and "Outcome O1: depression" in p
    assert "S2 | no difference" not in p             # study details only through the tools


def test_scripted_llm_runs_the_tool_calls():
    llm = ScriptedLLM({"critic": [{"tool_calls": [
        {"name": "lookup_study", "args": {"label": "S2"}},
        {"name": "check_quote", "args": {"text": "no difference"}}],
        "answer": {"issues": []}}]})
    ans, usage, trace = llm.call_with_tools(C.Review, "p", C.critic_tools(SYN), agent="critic")
    assert ans.issues == [] and [x["tool"] for x in trace] == ["lookup_study", "check_quote"]
    assert "no difference" in trace[0]["result"]


class FakeToolChat:
    """Round 1: calls lookup_study; round 2: submits the Review."""
    def __init__(self):
        self.rounds, self.choices = 0, []

    def bind_tools(self, tools, tool_choice=None):
        self.choices.append(tool_choice)
        self.names = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools]
        return self

    def invoke(self, messages):
        self.rounds += 1
        usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        if self.rounds == 1:
            return AIMessage(content="", usage_metadata=usage, tool_calls=[
                {"name": "lookup_study", "args": {"label": "S2"}, "id": "c1"}])
        assert any(getattr(m, "tool_call_id", None) == "c1" for m in messages)
        return AIMessage(content="", usage_metadata=usage, tool_calls=[
            {"name": "Review", "args": {"issues": [
                {"sentence": 1, "problem": "S2 shows no difference", "evidence":
                 "S2 | no difference | between-group", "severity": "warning"}]}, "id": "c2"}])


def test_openai_tool_calling_loop_usage_and_cache(tmp_path):
    chat = FakeToolChat()
    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m, *a: chat)
    ans, usage, trace = llm.call_with_tools(C.Review, "check", C.critic_tools(SYN),
                                            agent="critic")
    assert ans.issues[0].sentence == 1 and chat.rounds == 2
    assert chat.choices == ["any", "any"] and "Review" in chat.names
    assert [x["tool"] for x in trace] == ["lookup_study"] and "Bo 2021" in trace[0]["result"]
    assert (usage.input_tokens, usage.output_tokens, usage.cached) == (200, 20, False)
    again = llm.call_with_tools(C.Review, "check", C.critic_tools(SYN), agent="critic")
    assert chat.rounds == 2 and again[1].cached and again[2] == trace   # from the cache


def test_last_round_forces_the_answer(tmp_path):
    chat = FakeToolChat()
    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m, *a: chat)
    llm.call_with_tools(C.Review, "x", C.critic_tools(SYN), agent="critic", max_rounds=2)
    assert chat.choices == ["any", "Review"]


def test_critic_stage_records_the_tool_calls():
    state = {"protocol": {"question": "Q?", "pico": {"outcomes": ["depression"]}},
             "synthesis": {**SYN, "round": 1, "studies": [], "outcomes": [],
                           "heterogeneity": []},
             "completed_stages": ["extraction", "synthesis"], "included": []}
    llm = ScriptedLLM({"critic": [{"tool_calls": [{"name": "outcome_summary",
                                                   "args": {"outcome_id": "O1"}}],
                                   "answer": {"issues": []}}]})
    out = C.make_critic(llm)(state)
    assert [t["tool"] for t in out["qa_report"]["tool_calls"]] == ["outcome_summary"]
    assert out["audit_log"][0]["tool_calls"] == 1
    assert out["audit_log"][0]["tools_used"] == ["outcome_summary"]


def test_tool_call_limit_forces_the_answer(tmp_path):
    chat = FakeToolChat()
    llm = OpenAILLM(cache_dir=tmp_path, chat_factory=lambda m, *a: chat)
    _, _, trace = llm.call_with_tools(C.Review, "y", C.critic_tools(SYN), agent="critic",
                                      max_calls=0)
    assert trace[0]["result"].startswith("limit of 0 tool calls reached")
    assert chat.choices == ["Review", "Review"]        # no budget: the answer is due at once
