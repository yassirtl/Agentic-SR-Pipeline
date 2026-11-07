"""LLM access: one interface, a real OpenAI implementation and a scripted fake.

Every agent calls ``llm.structured_call(schema, prompt, agent=...)`` and gets back
a validated Pydantic object, never free text.
"""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"


class LLM(Protocol):
    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T: ...


class OpenAILLM:
    """Real implementation. The model is chosen per agent (cheap vs strong)."""

    def __init__(self, models: dict[str, str] | None = None, default: str = DEFAULT_MODEL):
        self.models = models or {}
        self.default = default

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default)

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T:
        # Imported and built lazily: ChatOpenAI needs an API key at construction.
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=self.model_for(agent), temperature=0)
        return chat.with_structured_output(schema).invoke(prompt)


class ScriptedLLM:
    """Test double: returns pre-written answers in order, one queue per agent.

    Answers may be schema instances or plain dicts (validated against the schema).
    Every call is recorded in ``self.calls`` as ``(agent, prompt)``.
    """

    def __init__(self, script: dict[str, list[Any]]):
        self.script = {agent: list(answers) for agent, answers in script.items()}
        self.calls: list[tuple[str, str]] = []

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T:
        self.calls.append((agent, prompt))
        queue = self.script.get(agent)
        if not queue:
            raise RuntimeError(f"ScriptedLLM: no scripted answer left for agent '{agent}'")
        answer = queue.pop(0)
        return answer if isinstance(answer, schema) else schema.model_validate(answer)
