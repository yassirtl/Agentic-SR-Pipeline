"""Review protocol (PRISMA 2020 item 24): fixed before the search starts.

Two parts:
- the *scientific* part (title, PICO, criteria, study designs), drafted by the
  Planner LLM from the question;
- the *scope* part (databases, years, languages), set by the user in the
  protocol template and never invented by the LLM. One exception: with
  ``databases: auto`` (the default) the Planner picks the databases from the
  topic, with a one-line reason; OpenAlex is always kept (code, not prompt).
The human approves (or edits) the merged protocol before any search runs.

Study designs: ``study_designs`` restricts the search and comes only from the
user or the question (see ``agents.planner.choose_designs``). The Planner may
also write ``suggested_designs`` + ``design_reason``: a suggestion the human can
accept at approval; it is never applied by itself (and never in ``--auto``).
"""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Database = Literal["openalex", "pubmed", "arxiv"]


class PICO(BaseModel):
    population: str = Field(description="Who is studied (patients, participants, systems).")
    intervention: str = Field(description="The intervention, exposure or method of interest.")
    comparator: str = Field(description="What it is compared with. For a question about an "
                            "effect, a control (e.g. 'no intervention, usual care or any "
                            "control'), not 'none'.")
    outcomes: list[str] = Field(description="Outcomes named by the question, primary outcome "
                                "first; others only if the user asked for them.")


class YearRange(BaseModel):
    start: int | None = Field(None, description="First publication year included (None = no limit).")
    end: int | None = Field(None, description="Last publication year included (None = no limit).")
    end_date: date | None = Field(
        None, description="Last publication date included (YYYY-MM-DD), when a year is not "
                          "precise enough, e.g. to match a published review's search date. "
                          "Sets ``end`` to its year.")

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_date is not None:
            if self.end is not None and self.end != self.end_date.year:
                raise ValueError(f"years.end ({self.end}) and years.end_date "
                                 f"({self.end_date}) disagree")
            self.end = self.end_date.year
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"years.start ({self.start}) is after years.end ({self.end})")
        return self


class ProtocolDraft(BaseModel):
    """What the Planner LLM writes (the scientific part)."""

    title: str = Field(description="Short review title.")
    pico: PICO
    inclusion_criteria: list[str] = Field(
        description="Short, testable criteria a record must meet, one idea each.")
    exclusion_criteria: list[str] = Field(
        description="Short, testable reasons to exclude a record, one idea each.")
    full_text_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria from the two lists above, copied word for word, that a title "
                    "and abstract can never show (e.g. 'groups comparable at baseline', "
                    "'reports mean and standard deviation'). They are checked only at "
                    "full-text screening. Empty if none.")
    study_designs: list[str] = Field(
        description="Eligible study designs ONLY if the question or the user restricts them "
                    "(e.g. 'randomized controlled trial'). Otherwise an empty list = any "
                    "design. Never add designs you think are suitable.")
    suggested_designs: list[str] = Field(
        default_factory=list,
        description="A SUGGESTION shown to the human, never applied by itself: the designs "
                    "that best answer this type of question (e.g. 'does X reduce Y?' -> "
                    "'randomized controlled trial'). Empty if no design clearly fits.")
    design_reason: str = Field(
        "", description="One line: why these designs are suggested.")
    rationale: str = Field(description="Two or three sentences on how the question was framed.")


class DatabaseChoice(BaseModel):
    """Extra fields the Planner writes when the template says ``databases: auto``."""

    databases: list[Database] = Field(
        min_length=1,
        description="Databases to search: openalex (all fields, always), pubmed (health, "
                    "medicine, life sciences), arxiv (computer science, physics, maths, "
                    "statistics, quantitative fields).")
    database_reason: str = Field(description="One line: why these databases.")


class ProtocolDraftAuto(ProtocolDraft, DatabaseChoice):
    """Planner output when the databases are chosen by the LLM."""


class Protocol(ProtocolDraft):
    """Full protocol = question + LLM draft + scope from the template."""

    question: str
    databases: list[Database] = Field(min_length=1)
    database_reason: str = ""
    years: YearRange = Field(default_factory=lambda: YearRange())
    languages: list[str] = Field(default_factory=lambda: ["English"])
    # Provenance, shown at approval: an evaluation run takes its criteria from the gold
    # review (evaluation/topics.yaml), not from the Planner.
    topic: str | None = None
    criteria_source: str = "Planner (LLM draft)"
    # Criterion text -> one positive test sentence the screeners judge instead of the
    # (displayed) wording, e.g. for the review's negative exclusion criteria (Part 6c).
    criterion_tests: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _tags_follow_criteria(self):
        """Stage tags and test sentences are keyed by criterion text: a criterion that was
        reworded or removed (e.g. at approval) loses them, nothing stale survives."""
        texts = set(self.inclusion_criteria) | set(self.exclusion_criteria)
        self.full_text_criteria = [c for c in dict.fromkeys(self.full_text_criteria)
                                   if c in texts]
        self.criterion_tests = {c: t for c, t in self.criterion_tests.items()
                                if c in texts and t.strip()}
        return self


class Template(BaseModel):
    """Contents of protocols/template.yaml. Scope fields are applied as is;
    the other fields are hints for the Planner (may be left empty)."""

    databases: list[Database] | Literal["auto"] = Field(
        "auto", description="'auto' = the Planner chooses; a list = fixed by the user.")
    years: YearRange = Field(default_factory=lambda: YearRange())
    languages: list[str] = Field(default_factory=lambda: ["English"])
    study_designs: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    pico: dict[str, str | list[str]] = Field(default_factory=dict)


def choose_databases(template: Template, draft: ProtocolDraft) -> tuple[list[str], str]:
    """Final database list + reason. An explicit template list always wins."""
    if template.databases != "auto":
        return list(template.databases), "set by the user in the protocol template"
    chosen = list(dict.fromkeys(draft.databases))  # dedupe, keep order
    if "openalex" not in chosen:                    # multidisciplinary: always searched
        chosen.insert(0, "openalex")
    return chosen, draft.database_reason


def load_template(path: str | Path) -> Template:
    return Template.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def deep_merge(base: dict, edits: dict) -> dict:
    """Return ``base`` with ``edits`` applied; nested dicts are merged, lists replaced."""
    out = dict(base)
    for k, v in edits.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
