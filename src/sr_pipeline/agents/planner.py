"""Planner: research question -> PRISMA protocol, then human approval.

Two graph nodes, on purpose:

- ``planner`` (a stage): one LLM call that drafts the scientific part of the
  protocol, merged with the scope from the template.
- ``protocol_approval`` (not a stage, fixed edge after planner): pauses the run
  with LangGraph ``interrupt`` so a human can approve or edit the protocol.

Why separate: on resume LangGraph re-executes the interrupted node *from the
top*. Keeping the LLM call out of the approval node means resuming never calls
the LLM again.

Resume value (``Command(resume=...)``):
  ``True``                 approve as is
  ``{"edits": {...}}``     apply edits (nested dicts merged, lists replaced)
(``None`` and ``{}`` are not usable: LangGraph treats them as "no resume value".)
Invalid answers or edits do not crash the run: the node asks again with the error.
"""

from pydantic import ValidationError
from langgraph.types import interrupt

from ..llm import LLM
from ..protocol import SCOPE_FIELDS, Protocol, ProtocolDraft, Template, deep_merge
from ..state import ReviewState, stage_round


def _hints(template: Template) -> str:
    lines = []
    pico = {k: v for k, v in template.pico.items() if v}
    if pico:
        lines.append("PICO elements fixed by the user: "
                     + "; ".join(f"{k}: {v}" for k, v in pico.items()))
    for name in ("study_designs", "inclusion_criteria", "exclusion_criteria"):
        values = getattr(template, name)
        if values:
            lines.append(f"{name.replace('_', ' ').capitalize()} required by the user: "
                         + "; ".join(values))
    return "\n".join(lines)


def build_prompt(question: str, template: Template) -> str:
    years = template.years
    span = f"{years.start or 'any'} to {years.end or 'now'}"
    hints = _hints(template)
    return (
        "You are a methodologist writing the protocol of a PRISMA 2020 systematic review.\n"
        "Frame the research question with PICO and write eligibility criteria.\n"
        "Rules:\n"
        "- Criteria must be short and testable from a title, abstract or full text.\n"
        "- One idea per criterion; do not repeat the scope (databases, years, languages).\n"
        "- Keep every element the user fixed below; refine, do not contradict.\n"
        "- If the question does not name a comparator, write 'any' or 'none'.\n\n"
        f"Research question: {question}\n"
        f"Scope (fixed): databases {', '.join(template.databases)}; years {span}; "
        f"languages {', '.join(template.languages)}.\n"
        + (f"{hints}\n" if hints else "")
    )


def make_planner(llm: LLM, template: Template):
    def planner(state: ReviewState) -> dict:
        question = state["question"]
        draft, usage = llm.call_with_usage(ProtocolDraft, build_prompt(question, template),
                                           agent="planner")
        # Scope comes from the template, never from the LLM.
        protocol = Protocol(question=question, **draft.model_dump(),
                            **template.model_dump(include=set(SCOPE_FIELDS)))
        completed = state.get("completed_stages", [])
        return {
            "protocol": protocol.model_dump(mode="json"),
            "completed_stages": ["planner"],
            "audit_log": [{"node": "planner", "step": state.get("steps_taken", 0),
                           "round": stage_round("planner", completed),
                           "event": "protocol drafted", **usage.as_dict()}],
        }

    return planner


def protocol_approval(state: ReviewState) -> dict:
    """Pause for a human decision on the protocol (see module docstring)."""
    protocol = state["protocol"]
    error = None
    asked = 0
    while True:
        asked += 1
        answer = interrupt({"kind": "protocol_approval", "protocol": protocol, "error": error,
                            "help": "resume with True to approve, or {'edits': {...}}"})
        if answer is True:
            edits = {}
        elif isinstance(answer, dict) and set(answer) <= {"edits"} \
                and isinstance(answer.get("edits"), dict):
            edits = answer["edits"]
        else:
            error = f"unrecognised answer: {answer!r}"
            continue
        try:
            approved = Protocol.model_validate(deep_merge(protocol, edits))
        except ValidationError as exc:
            error = f"invalid edits {edits!r}: {exc.errors(include_url=False)}"
            continue
        break

    return {
        "protocol": approved.model_dump(mode="json"),
        "audit_log": [{"node": "protocol_approval", "step": state.get("steps_taken", 0),
                       "event": "protocol approved", "approved_by": "human",
                       "edited_fields": sorted(edits), "edits": edits, "attempts": asked}],
    }
