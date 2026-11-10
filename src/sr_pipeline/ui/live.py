"""Live runs from the UI (Part 11b): start, answer and resume a review in a background thread.

No Streamlit import here: the pages call these functions, the worker never touches the UI.

One code path for everything. A worker thread calls ``cli.run`` (the same function as
``sr-review``, same graph, same SQLite checkpoint) and streams the graph until one of:
- a pause for a human (protocol approval, screening conflicts, budget cap): the worker's
  ``answer`` function has no answer ready, raises ``AwaitingHuman``, the run stays paused at
  its checkpoint and the worker ends with status ``waiting`` + the pause payload;
- the end (``finished`` with a report, or ``ended`` without one: step budget, repeated
  failure);
- an error (``error``, with the one-line API message when it is an API error).

Answering a pause in the UI, or resuming a stopped run from the Runs page, starts a new worker
with ``cli.run(resume=run_id)`` and the human's answer (the same payloads as the CLI).

The worker writes ``runs/<id>/live.json`` (status, pause payload, error) and ``live.log``
(the CLI's progress lines). The UI reads those and ``audit.jsonl``; it never opens the
checkpoint. Workers are kept in a module-level registry keyed by run id (Streamlit reruns the
page script, module state survives); a second worker on the same run is refused.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import cli
from ..audit import new_run_id
from ..config import Config, load_config
from ..llm import api_error_message

# Test hook: extra keyword arguments for cli.run (scripted LLM, fake search / full text /
# embedder). Empty in a real session: cli.run then builds the real clients from the config.
INJECT: dict[str, Any] = {}
CONFIG: Config | None = None     # test hook; otherwise load_config() (.env, SR_RUNS_DIR)

_WORKERS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()

NO_ANSWER = object()


def config() -> Config:
    return CONFIG or load_config()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(run_dir: Path, status: str, **extra) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = run_dir / "live.json.tmp"
    tmp.write_text(json.dumps({"status": status, "updated_at": _now(), **extra},
                              indent=2, default=str))
    tmp.replace(run_dir / "live.json")    # atomic: the UI never reads half a file


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "live.json").read_text())
    except (OSError, ValueError):
        return {}


def is_alive(run_id: str) -> bool:
    with _LOCK:
        t = _WORKERS.get(run_id)
        return bool(t and t.is_alive())


def join(run_id: str, timeout: float | None = None) -> None:
    """Wait for the run's worker (tests; the UI polls instead)."""
    t = _WORKERS.get(run_id)
    if t:
        t.join(timeout)


def answer_once(kind: str | None, value: Any):
    """``answer`` function for cli.run: gives ``value`` to the first pause of ``kind``, then
    leaves every other pause (or the same one asked again after a refused answer, which
    then carries the error) for the human: AwaitingHuman."""
    used = {"done": value is NO_ANSWER}

    def answer(payload: dict):
        if not used["done"] and payload.get("kind") == kind:
            used["done"] = True
            return value
        raise cli.AwaitingHuman(payload)

    return answer


def _worker(cfg: Config, run_id: str, kwargs: dict) -> None:
    run_dir = cfg.runs_dir / run_id
    log = run_dir / "live.log"

    def out(*parts):
        with log.open("a") as f:
            f.write(" ".join(str(p) for p in parts) + "\n")

    write_status(run_dir, "running")
    try:
        _, state = cli.run(cfg, out=out, ask=_no_terminal, **INJECT, **kwargs)
        write_status(run_dir, "finished" if state.get("report") else "ended",
                     stages=state.get("completed_stages", []))
    except cli.AwaitingHuman as exc:
        write_status(run_dir, "waiting", pending=exc.payload)
    except BaseException as exc:   # a worker must never die silently
        msg = api_error_message(exc) or f"{type(exc).__name__}: {exc}"
        out(traceback.format_exc())
        write_status(run_dir, "error", error=msg)


def _no_terminal(prompt: str) -> str:
    raise RuntimeError("the UI never asks at the terminal")


def _spawn(cfg: Config, run_id: str, kwargs: dict) -> str:
    with _LOCK:
        t = _WORKERS.get(run_id)
        if t and t.is_alive():
            raise RuntimeError(f"run {run_id} is already running")
        write_status(cfg.runs_dir / run_id, "starting")
        t = threading.Thread(target=_worker, args=(cfg, run_id, kwargs),
                             name=f"sr-run-{run_id}", daemon=True)
        _WORKERS[run_id] = t
        t.start()
    return run_id


def start(*, question: str | None, topic: str | None, auto: bool, limit: int | None,
          budget_usd: float | None, estimate: dict | None = None,
          cfg: Config | None = None) -> str:
    """Start a new run in the background; returns its run id at once."""
    cfg = cfg or config()
    run_id = new_run_id()
    return _spawn(cfg, run_id, {
        "question": question, "topic_id": topic, "auto": auto, "screen_limit": limit,
        "budget_usd": budget_usd, "run_id": run_id, "answer": answer_once(None, NO_ANSWER),
        "meta_extra": {"started_from": "ui", **({"estimate": estimate} if estimate else {})}})


def resume(run_id: str, *, kind: str | None = None, answer: Any = NO_ANSWER,
           budget_usd: float | None = None, cfg: Config | None = None) -> str:
    """Continue a paused or stopped run. ``kind`` + ``answer``: the human's answer to the
    pause the run is waiting at (same payload as the CLI); without one, a stopped run goes on
    from its last checkpoint and a paused run stays paused (status ``waiting`` again).
    ``budget_usd``: a new cap for a stopped run (runs started before Part 11b have none)."""
    cfg = cfg or config()
    if not (cfg.runs_dir / run_id / "run.json").exists():
        raise FileNotFoundError(f"no run {run_id} in {cfg.runs_dir}")
    return _spawn(cfg, run_id, {"question": None, "topic_id": None, "auto": False,
                                "resume": run_id, "answer": answer_once(kind, answer),
                                "budget_usd": budget_usd})


# --- Answers built from the UI forms (same payloads as the CLI) ---------------------------

def protocol_answer(protocol: dict, edited: dict, accept_designs: bool):
    """True when nothing changed; else {"edits": {...only changed fields...}} and/or
    {"accept_suggested_designs": True}."""
    edits = {k: v for k, v in edited.items() if _norm(_get(protocol, k), k) != _norm(v, k)}
    nested: dict = {}
    for key, value in edits.items():
        cur = nested
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    answer: dict = {}
    if nested:
        answer["edits"] = nested
    if accept_designs:
        answer["accept_suggested_designs"] = True
    return answer or True


def _norm(value, key: str = ""):
    """What a form field means, so an untouched form sends ``True``: a text box turns None
    into "", a text area splits on lines and strips, a multiselect may reorder. A spurious
    criteria edit is not harmless (it rewrites criteria_source and drops the tests and labels
    keyed on the criterion text), so only a real change counts."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return " ".join(value.split()) or None
    if isinstance(value, list):
        items = [_norm(x) for x in value]
        items = [x for x in items if x is not None]
        return sorted(items, key=str) if key == "databases" else items
    return value


def form_text(value) -> str:
    """A protocol string as a form shows it (whitespace collapsed, None -> "")."""
    return " ".join(str(value).split()) if value else ""


def _get(d: dict, dotted: str):
    for p in dotted.split("."):
        d = (d or {}).get(p) if isinstance(d, dict) else None
    return d
