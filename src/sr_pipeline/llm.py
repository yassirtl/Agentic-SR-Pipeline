"""LLM access: one interface, a real OpenAI implementation and a scripted fake.

Every agent calls ``llm.structured_call(schema, prompt, agent=...)`` and gets back
a validated Pydantic object, never free text. ``call_with_usage`` returns the same
object plus a ``Usage`` record (model, tokens, cost, cached) for the audit log.

Reproducibility: ``temperature=0`` is not a guarantee (the provider may change
outputs over time). The disk cache is: the same (model, schema, prompt) always
returns the stored answer, without calling the API.
"""

import hashlib
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

from .config import find_project_root

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CACHE_SUBDIR = ".cache/llm"  # under the project root (see config.py)

# USD per 1M tokens: (input, output). Edit when prices change or models are added.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),  # embeddings: input tokens only
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None = model missing from PRICES
    cached: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def cache_key(model: str, schema: type[BaseModel], prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "schema": schema.__name__,
         "json_schema": schema.model_json_schema(), "prompt": prompt},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLM(Protocol):
    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]: ...

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T: ...


class _Base:
    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]:
        raise NotImplementedError

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T:
        return self.call_with_usage(schema, prompt, agent=agent)[0]


def _default_chat_factory(model: str):
    # Imported lazily: ChatOpenAI needs an API key at construction.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)


class OpenAILLM(_Base):
    """Real implementation: per-agent model, disk cache, token usage and cost.

    ``chat_factory(model)`` builds the chat model (ChatOpenAI by default); tests
    pass a fake so no network or API key is needed.
    """

    def __init__(self, models: dict[str, str] | None = None, default: str = DEFAULT_MODEL,
                 cache_dir: str | Path | None = None,
                 chat_factory: Callable[[str], Any] = _default_chat_factory):
        self.models = models or {}
        self.default = default
        # Absolute by default, so every working directory shares one cache.
        self.cache_dir = Path(cache_dir) if cache_dir else find_project_root() / DEFAULT_CACHE_SUBDIR
        self.chat_factory = chat_factory
        self._clients: dict[str, Any] = {}  # one client per model, built once
        self._lock = threading.Lock()       # screening calls the LLM from several threads

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default)

    def _client(self, model: str):
        with self._lock:
            if model not in self._clients:
                self._clients[model] = self.chat_factory(model)
            return self._clients[model]

    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]:
        model = self.model_for(agent)
        path = self.cache_dir / f"{cache_key(model, schema, prompt)}.json"

        if path.exists():
            data = json.loads(path.read_text())
            usage = Usage(model, data["input_tokens"], data["output_tokens"],
                          cost_usd=0.0, cached=True)
            return schema.model_validate(data["parsed"]), usage

        out = self._client(model).with_structured_output(schema, include_raw=True).invoke(prompt)
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise ValueError(f"structured output failed for {schema.__name__}: "
                             f"{out.get('parsing_error')}")
        parsed = out["parsed"]
        meta = getattr(out["raw"], "usage_metadata", None) or {}
        tin, tout = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
        usage = Usage(model, tin, tout, cost_usd=compute_cost(model, tin, tout))

        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write (unique temp file, then rename): with concurrent screening a
        # reader never sees a half-written cache file.
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps({"model": model, "schema": schema.__name__,
                                   "parsed": parsed.model_dump(mode="json"),
                                   "input_tokens": tin, "output_tokens": tout}, indent=2))
        tmp.replace(path)
        return parsed, usage


class ScriptedLLM(_Base):
    """Test double: returns pre-written answers in order, one queue per agent.

    Answers may be schema instances, plain dicts (validated against the schema),
    or exceptions (raised, to simulate a failing LLM). An agent's script may
    also be a function ``prompt -> answer``: the answer then depends on the
    prompt, not on call order (needed when calls run in several threads).
    Every call is recorded in ``self.calls`` as ``(agent, prompt)``.
    """

    def __init__(self, script: dict[str, list[Any] | Callable[[str], Any]]):
        self.script = {agent: answers if callable(answers) else list(answers)
                       for agent, answers in script.items()}
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]:
        with self._lock:
            self.calls.append((agent, prompt))
            queue = self.script.get(agent)
            if callable(queue):
                answer = queue(prompt)
            elif not queue:
                raise RuntimeError(f"ScriptedLLM: no scripted answer left for agent '{agent}'")
            else:
                answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        parsed = answer if isinstance(answer, schema) else schema.model_validate(answer)
        return parsed, Usage("scripted")
