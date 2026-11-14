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
import re
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
    "gpt-4.1-mini": (0.40, 1.60),  # checked on the OpenAI pricing page, 2026-11-07
    "gpt-4.1": (2.00, 8.00),
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


def is_fatal(exc: BaseException) -> bool:
    """Part 10b: an API error that no retry or other record can fix (no credits left, bad
    or revoked key). Stages that catch per-record errors re-raise these, so the run stops
    with one message (``sr-review`` then says: fix it, then ``--resume``) instead of
    marking every remaining record as failed."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    if isinstance(exc, openai.RateLimitError):
        return getattr(exc, "code", None) == "insufficient_quota" or \
            "insufficient_quota" in str(exc)
    return False


def error_signature(exc: BaseException) -> str:
    """Part 10d: what kind of failure this was, so two runs of a stage can be compared: the
    exception type and the start of its message with numbers removed (token counts differ
    between two identical length errors)."""
    msg = re.sub(r"\d+", "#", str(exc).splitlines()[0] if str(exc) else "")
    return f"{type(exc).__name__}: {msg[:120]}"


def api_error_message(exc: BaseException) -> str | None:
    """One line for an OpenAI API error (None if ``exc`` is not one)."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(exc, openai.OpenAIError):
        return None
    if isinstance(exc, openai.RateLimitError) and is_fatal(exc):
        return "OpenAI API: no credits left (insufficient_quota). Add credits on the OpenAI billing page"
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "OpenAI API: the key was refused (check OPENAI_API_KEY in .env)"
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI API: rate limit still exceeded after retries; wait a minute"
    if isinstance(exc, openai.APIConnectionError):
        return "OpenAI API: no connection (network down or timeout)"
    return f"OpenAI API error: {type(exc).__name__}: {str(exc)[:200]}"


@dataclass
class Tool:
    """A tool the LLM may call (Part 12, tool calling): ``func(**args) -> str``, its
    arguments described by the pydantic model ``args``."""
    name: str
    description: str
    func: Callable[..., str]
    args: type[BaseModel]


def _run_tool(tools: dict[str, "Tool"], name: str, args: dict) -> str:
    t = tools.get(name)
    if t is None:
        return f"error: unknown tool {name!r}"
    try:
        return str(t.func(**t.args.model_validate(args).model_dump()))
    except Exception as exc:                          # the LLM sees the error and goes on
        return f"error: {type(exc).__name__}: {exc}"[:300]


class LLM(Protocol):
    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]: ...

    def call_with_tools(self, schema: type[T], prompt: str, tools: list[Tool], *, agent: str,
                        max_rounds: int = 5) -> tuple[T, Usage, list[dict]]: ...

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T: ...


class _Base:
    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]:
        raise NotImplementedError

    def structured_call(self, schema: type[T], prompt: str, *, agent: str) -> T:
        return self.call_with_usage(schema, prompt, agent=agent)[0]


# Output cap per call: every structured answer is short; seen live, a screening call looped
# until the 16,384-token limit (5 minutes). A cap makes such a call fail fast (-> unresolved).
DEFAULT_MAX_TOKENS = 4096
# Part 10d: the synthesis answer grows with the number of studies (each sentence lists the
# labels it cites). Seen live (T3, --limit 500, 231 studies): the narrative hit 4,096 tokens
# five times in a row (LengthFinishReasonError, no text). 16,384 = gpt-4o-mini's maximum.
MAX_TOKENS_BY_AGENT: dict[str, int] = {"synthesis": 16384}


def max_tokens_for(agent: str) -> int:
    return MAX_TOKENS_BY_AGENT.get(agent, DEFAULT_MAX_TOKENS)


def _default_chat_factory(model: str, max_tokens: int = DEFAULT_MAX_TOKENS):
    # Imported lazily: ChatOpenAI needs an API key at construction.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0, max_tokens=max_tokens)


TOOL_RESULT_CHARS = 4000     # what one tool result may add to the conversation
MAX_TOOL_CALLS = 60          # past this, a tool call gets "limit reached" and the answer is due


class OpenAILLM(_Base):
    """Real implementation: per-agent model, disk cache, token usage and cost.

    ``chat_factory(model)`` builds the chat model (ChatOpenAI by default); tests
    pass a fake so no network or API key is needed. An agent with a larger output cap
    (``MAX_TOKENS_BY_AGENT``) gets ``chat_factory(model, max_tokens)``.
    """

    def __init__(self, models: dict[str, str] | None = None, default: str = DEFAULT_MODEL,
                 cache_dir: str | Path | None = None,
                 chat_factory: Callable[[str], Any] = _default_chat_factory):
        self.models = models or {}
        self.default = default
        # Absolute by default, so every working directory shares one cache.
        self.cache_dir = Path(cache_dir) if cache_dir else find_project_root() / DEFAULT_CACHE_SUBDIR
        self.chat_factory = chat_factory
        self._clients: dict[tuple[str, int], Any] = {}  # one per (model, cap), built once
        self._lock = threading.Lock()       # screening calls the LLM from several threads

    def model_for(self, agent: str) -> str:
        return self.models.get(agent, self.default)

    def _client(self, model: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        with self._lock:
            key = (model, max_tokens)
            if key not in self._clients:
                self._clients[key] = (self.chat_factory(model) if max_tokens == DEFAULT_MAX_TOKENS
                                      else self.chat_factory(model, max_tokens))
            return self._clients[key]

    def call_with_usage(self, schema: type[T], prompt: str, *, agent: str) -> tuple[T, Usage]:
        model = self.model_for(agent)
        path = self.cache_dir / f"{cache_key(model, schema, prompt)}.json"

        if path.exists():
            data = json.loads(path.read_text())
            usage = Usage(model, data["input_tokens"], data["output_tokens"],
                          cost_usd=0.0, cached=True)
            return schema.model_validate(data["parsed"]), usage

        client = self._client(model, max_tokens_for(agent))
        out = client.with_structured_output(schema, include_raw=True).invoke(prompt)
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

    def call_with_tools(self, schema: type[T], prompt: str, tools: list[Tool], *, agent: str,
                        max_rounds: int = 5, max_calls: int = MAX_TOOL_CALLS
                        ) -> tuple[T, Usage, list[dict]]:
        """LangChain tool calling: the model gets ``tools`` plus the final answer
        (``schema``, offered as one more tool) and decides which to call, round after round,
        until it calls the final answer; the last round may only give the answer. Returns
        (answer, summed usage, trace of the tool calls). The whole exchange is cached on
        (model, schema, prompt, tool names), like a single call."""
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_core.tools import StructuredTool

        model = self.model_for(agent)
        key = cache_key(model, schema, prompt + "\n[tools: " + ", ".join(t.name for t in tools)
                        + "]")
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return (schema.model_validate(data["parsed"]),
                    Usage(model, data["input_tokens"], data["output_tokens"], cost_usd=0.0,
                          cached=True), data.get("trace", []))
        by_name = {t.name: t for t in tools}
        lc_tools = [StructuredTool.from_function(func=t.func, name=t.name,
                                                 description=t.description, args_schema=t.args)
                    for t in tools]
        client = self._client(model, max_tokens_for(agent))
        messages, trace, tin, tout, parsed = [HumanMessage(prompt)], [], 0, 0, None
        for rnd in range(1, max_rounds + 1):
            spent = len(trace) >= max_calls
            choice = schema.__name__ if rnd == max_rounds or spent else "any"
            ai = client.bind_tools([*lc_tools, schema], tool_choice=choice).invoke(messages)
            meta = getattr(ai, "usage_metadata", None) or {}
            tin, tout = tin + meta.get("input_tokens", 0), tout + meta.get("output_tokens", 0)
            messages.append(ai)
            final = next((c for c in ai.tool_calls if c["name"] == schema.__name__), None)
            if final is not None:
                parsed = schema.model_validate(final["args"])
                break
            for c in ai.tool_calls:
                result = (_run_tool(by_name, c["name"], c.get("args") or {})
                          if len(trace) < max_calls else
                          f"limit of {max_calls} tool calls reached: submit your answer now")
                trace.append({"round": rnd, "tool": c["name"], "args": c.get("args") or {},
                              "result": result[:300]})
                messages.append(ToolMessage(content=result[:TOOL_RESULT_CHARS],
                                            tool_call_id=c["id"]))
        if parsed is None:
            raise ValueError(f"tool calling: no {schema.__name__} after {max_rounds} rounds")
        usage = Usage(model, tin, tout, cost_usd=compute_cost(model, tin, tout))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps({"model": model, "schema": schema.__name__,
                                   "parsed": parsed.model_dump(mode="json"), "trace": trace,
                                   "input_tokens": tin, "output_tokens": tout}, indent=2,
                                  default=str))
        tmp.replace(path)
        return parsed, usage, trace


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

    def call_with_tools(self, schema: type[T], prompt: str, tools: list[Tool], *, agent: str,
                        max_rounds: int = 5) -> tuple[T, Usage, list[dict]]:
        """A scripted answer may be ``{"tool_calls": [{"name", "args"}...], "answer": {...}}``:
        the tool calls are really run (so a test sees the tools work), then the answer is
        returned. A plain answer means no tool call."""
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
        trace = []
        if isinstance(answer, dict) and "answer" in answer and "tool_calls" in answer:
            by_name = {t.name: t for t in tools}
            for c in answer["tool_calls"]:
                trace.append({"round": 1, "tool": c["name"], "args": c.get("args") or {},
                              "result": _run_tool(by_name, c["name"], c.get("args") or {})[:300]})
            answer = answer["answer"]
        parsed = answer if isinstance(answer, schema) else schema.model_validate(answer)
        return parsed, Usage("scripted"), trace
