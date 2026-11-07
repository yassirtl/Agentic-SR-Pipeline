"""Planner: research question -> PRISMA protocol, then human approval.

Two graph nodes, on purpose:

- ``planner`` (a stage): one LLM call that drafts the scientific part of the
  protocol, merged with the scope from the template. With ``databases: auto``
  the same call also picks the databases (see ``protocol.choose_databases``).
  Study designs restrict the search (a design filter is AND-ed to it), so they
  come only from the user (template hint, used as is) or from the question
  itself; designs the LLM adds on its own are dropped by code
  (``choose_designs``) and logged. The Planner also writes a design
  *suggestion* (``suggested_designs`` + ``design_reason``, e.g. RCTs for an
  effect question): shown at approval, applied only if the human accepts it,
  never in ``--auto``.
- ``protocol_approval`` (not a stage, fixed edge after planner): pauses the run
  with LangGraph ``interrupt`` so a human can approve or edit the protocol.

Why separate: on resume LangGraph re-executes the interrupted node *from the
top*. Keeping the LLM call out of the approval node means resuming never calls
the LLM again.

Resume value (``Command(resume=...)``):
  ``True``                                approve as is
  ``{"edits": {...}}``                    apply edits (nested dicts merged, lists replaced)
  ``{"accept_suggested_designs": True}``  restrict the search to the Planner's
                                          suggested designs (may be combined with
                                          "edits"; an explicit ``study_designs``
                                          edit wins)
(``None`` and ``{}`` are not usable: LangGraph treats them as "no resume value".)
Invalid answers or edits do not crash the run: the node asks again with the error.
"""

import re

from pydantic import ValidationError
from langgraph.types import interrupt

from ..llm import LLM
from ..protocol import (Protocol, ProtocolDraft, ProtocolDraftAuto, Template,
                        choose_databases, deep_merge)
from ..evaluation.topics import seed_protocol, topic_template
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
        "- Separately, suggest designs (suggested_designs + design_reason): the designs that "
        "best answer this type of question (e.g. an effect question 'does X reduce Y?' -> "
        "randomized controlled trial). This is only a suggestion shown to the human, who "
        "decides; it does not change study_designs. Leave it empty if no design clearly "
        "fits.\n"
        "- Criteria must be short and testable from a title, abstract or full text. List in "
        "full_text_criteria (copied word for word) those an abstract can never show.\n"
        "- One idea per criterion; do not repeat the scope (databases, years, languages).\n"
        "- Keep every element the user fixed below; refine, do not contradict.\n"
        + (DATABASE_RULE if auto else "")
        + f"\nResearch question: {question}\n"
        f"Scope: databases {databases}; years {span}; "
        f"languages {', '.join(template.languages)}.\n"
        + (f"{hints}\n" if hints else "")
    )


def make_planner(llm: LLM, template: Template, topic: dict | None = None):
    """``topic`` (an entry of evaluation/topics.yaml, see ``evaluation.topics``): evaluation
    run; its scope and criteria go into the template, and override the draft exactly."""
    if topic is not None:
        template = topic_template(template, topic)

    def planner(state: ReviewState) -> dict:
        question = topic["question"] if topic is not None else state["question"]
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
        protocol_dict, seeded = protocol.model_dump(mode="json"), {}
        if topic is not None:
            protocol_dict, seeded = seed_protocol(protocol_dict, topic)
            seeded = {"seeded": {k: v for k, v in seeded.items() if k != "event"}}
        return {
            "protocol": protocol_dict,
            "completed_stages": ["planner"],
            "audit_log": [{"node": "planner", "step": state.get("steps_taken", 0),
                           "round": stage_round("planner", completed),
                           "event": "protocol drafted", "databases": databases,
                           "database_reason": reason, "study_designs": designs,
                           "study_designs_source": design_source,
                           "study_designs_dropped": dropped,
                           "suggested_designs": protocol.suggested_designs,
                           "design_reason": protocol.design_reason,
                           "design_suggestion": suggestion_status(protocol, accepted=False),
                           "criteria_source": protocol_dict["criteria_source"],
                           "full_text_criteria": protocol_dict["full_text_criteria"],
                           # Tags the LLM wrote that match none of its criteria word for word.
                           "full_text_criteria_dropped": [
                               c for c in draft.full_text_criteria
                               if c not in protocol.full_text_criteria],
                           **seeded,
                           **usage.as_dict()}],
        }

    return planner


def suggestion_status(protocol: Protocol | dict, *, accepted: bool) -> str:
    """Audit label for the design suggestion."""
    get = protocol.get if isinstance(protocol, dict) else lambda k: getattr(protocol, k)
    if not get("suggested_designs"):
        return "none suggested"
    return "accepted by human" if accepted else "suggested, not applied"


ANSWER_KEYS = {"edits", "accept_suggested_designs"}


def apply_answer(protocol: dict, answer) -> tuple[Protocol, dict, bool]:
    """Approved protocol, applied edits and whether the design suggestion was accepted.
    Raises ValueError for an answer that cannot be used (the node asks again)."""
    if answer is True:
        return Protocol.model_validate(protocol), {}, False
    if not (isinstance(answer, dict) and answer and set(answer) <= ANSWER_KEYS
            and isinstance(answer.get("edits", {}), dict)
            and isinstance(answer.get("accept_suggested_designs", False), bool)):
        raise ValueError(f"unrecognised answer: {answer!r}")
    edits = dict(answer.get("edits", {}))
    accepted = answer.get("accept_suggested_designs", False)
    if accepted:
        if not protocol.get("suggested_designs"):
            raise ValueError("accept_suggested_designs: the Planner suggested no design")
        edits.setdefault("study_designs", list(protocol["suggested_designs"]))
    if {"inclusion_criteria", "exclusion_criteria"} & set(edits) and \
            "criteria_source" not in edits:
        base = protocol.get("criteria_source") or "Planner (LLM draft)"
        edits["criteria_source"] = f"{base}; edited by human at approval"
    try:
        return Protocol.model_validate(deep_merge(protocol, edits)), edits, accepted
    except ValidationError as exc:
        raise ValueError(f"invalid edits {edits!r}: {exc.errors(include_url=False)}") from exc


def protocol_approval(state: ReviewState) -> dict:
    """Pause for a human decision on the protocol (see module docstring)."""
    protocol = state["protocol"]
    error = None
    asked = 0
    while True:
        asked += 1
        answer = interrupt({"kind": "protocol_approval", "protocol": protocol, "error": error,
                            "suggested_designs": protocol.get("suggested_designs", []),
                            "design_reason": protocol.get("design_reason", ""),
                            "criteria_source": protocol.get("criteria_source"),
                            "help": "resume with True to approve, {'edits': {...}}, and/or "
                                    "{'accept_suggested_designs': True}"})
        try:
            approved, edits, accepted = apply_answer(protocol, answer)
        except ValueError as exc:
            error = str(exc)
            continue
        break

    return {
        "protocol": approved.model_dump(mode="json"),
        "audit_log": [{"node": "protocol_approval", "step": state.get("steps_taken", 0),
                       "event": "protocol approved", "approved_by": "human",
                       "edited_fields": sorted(edits), "edits": edits, "attempts": asked,
                       "study_designs": approved.study_designs,
                       "criteria_source": approved.criteria_source,
                       "design_suggestion": suggestion_status(approved, accepted=accepted)}],
    }
