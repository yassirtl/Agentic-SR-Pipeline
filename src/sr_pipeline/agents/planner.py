"""Planner: research question -> PRISMA protocol, then human approval.

Two graph nodes, on purpose:

- ``planner`` (a stage): one LLM call that drafts the scientific part of the
  protocol, merged with the scope from the template. With ``databases: auto``
  the same call also picks the databases (see ``protocol.choose_databases``).
  Study designs restrict the search (a design filter is AND-ed to it), so they
  come only from the user (template hint, used as is) or from the question
  itself; designs the LLM adds on its own are dropped by code
  (``choose_designs``) and logged.
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

import re

from pydantic import ValidationError
from langgraph.types import interrupt

from ..llm import LLM
from ..protocol import (Protocol, ProtocolDraft, ProtocolDraftAuto, Template,
                        choose_databases, deep_merge)
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


# Words that show the question itself restricts the study design.
DESIGN_WORDS = re.compile(
    r"randomi[sz]|\btrials?\b|\brcts?\b|cohort|case.?control|cross.?sectional|"
    r"observational|longitudinal|prospective|retrospective|qualitative", re.I)


def choose_designs(question: str, template: Template, drafted: list[str]
                   ) -> tuple[list[str], str, list[str]]:
    """(designs, source, dropped). Template hint wins; else the LLM's designs are kept
    only when the question names a design; else no restriction."""
    if template.study_designs:
        return list(template.study_designs), "template", []
    if DESIGN_WORDS.search(question):
        return list(drafted), "question", []
    return [], "none (any design)", list(drafted)


DATABASE_RULE = (
    "- Choose the databases: openalex always (all disciplines); add pubmed for health, "
    "medicine, psychology or life-science questions; add arxiv for computer science, "
    "physics, mathematics, statistics or other quantitative fields. Give a one-line reason.\n"
)


def build_prompt(question: str, template: Template) -> str:
    years = template.years
    span = f"{years.start or 'any'} to {years.end or 'now'}"
    auto = template.databases == "auto"
    databases = "chosen by you (see rules)" if auto else ", ".join(template.databases)
    hints = _hints(template)
    return (
        "You are a methodologist writing the protocol of a PRISMA 2020 systematic review.\n"
        "Frame the research question with PICO and write eligibility criteria.\n"
        "Rules:\n"
        "- Stay on the question. Population: exactly who the question names; do not add a "
        "condition, severity or setting the question does not state.\n"
        "- Outcomes: only the ones the question names, primary outcome first. Do NOT add "
        "secondary outcomes unless the user listed them below.\n"
        "- Name each outcome as the measured construct (e.g. 'depressive symptoms'), never as "
        "a direction or result ('reduction in ...', 'improvement in ...').\n"
        "- Comparator: if the question asks about an effect (does X reduce / improve / cause "
        "Y?), write a control, e.g. 'no intervention, usual care or any control'; write "
        "'none' only for questions without any comparison (e.g. prevalence).\n"
        "- Study designs: only if the question or the user below restricts them; otherwise "
        "an empty list (any design). Do not add designs you think are suitable, and do not "
        "restrict designs through the criteria either.\n"
        "- Criteria must be short and testable from a title, abstract or full text.\n"
        "- One idea per criterion; do not repeat the scope (databases, years, languages).\n"
        "- Keep every element the user fixed below; refine, do not contradict.\n"
        + (DATABASE_RULE if auto else "")
        + f"\nResearch question: {question}\n"
        f"Scope: databases {databases}; years {span}; "
        f"languages {', '.join(template.languages)}.\n"
        + (f"{hints}\n" if hints else "")
    )


def make_planner(llm: LLM, template: Template):
    def planner(state: ReviewState) -> dict:
        question = state["question"]
        schema = ProtocolDraftAuto if template.databases == "auto" else ProtocolDraft
        draft, usage = llm.call_with_usage(schema, build_prompt(question, template),
                                           agent="planner")
        databases, reason = choose_databases(template, draft)
        designs, design_source, dropped = choose_designs(question, template,
                                                         draft.study_designs)
        # Years and languages come from the template, never from the LLM.
        protocol = Protocol(question=question,
                            **draft.model_dump(include=set(ProtocolDraft.model_fields)
                                               - {"study_designs"}),
                            study_designs=designs,
                            databases=databases, database_reason=reason,
                            years=template.years, languages=template.languages)
        completed = state.get("completed_stages", [])
        return {
            "protocol": protocol.model_dump(mode="json"),
            "completed_stages": ["planner"],
            "audit_log": [{"node": "planner", "step": state.get("steps_taken", 0),
                           "round": stage_round("planner", completed),
                           "event": "protocol drafted", "databases": databases,
                           "database_reason": reason, "study_designs": designs,
                           "study_designs_source": design_source,
                           "study_designs_dropped": dropped, **usage.as_dict()}],
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
