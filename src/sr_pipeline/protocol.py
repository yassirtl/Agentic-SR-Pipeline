"""Review protocol (PRISMA 2020 item 24): fixed before the search starts.

Two parts:
- the *scientific* part (title, PICO, criteria, study designs), drafted by the
  Planner LLM from the question;
- the *scope* part (databases, years, languages), set by the user in the
  protocol template and never invented by the LLM.
The human approves (or edits) the merged protocol before any search runs.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Database = Literal["openalex", "pubmed", "arxiv"]
SCOPE_FIELDS = ("databases", "years", "languages")


class PICO(BaseModel):
    population: str = Field(description="Who is studied (patients, participants, systems).")
    intervention: str = Field(description="The intervention, exposure or method of interest.")
    comparator: str = Field(description="What it is compared with; 'any' or 'none' if not applicable.")
    outcomes: list[str] = Field(description="Outcomes that matter for the question, most important first.")


class YearRange(BaseModel):
    start: int | None = Field(None, description="First publication year included (None = no limit).")
    end: int | None = Field(None, description="Last publication year included (None = no limit).")

    @model_validator(mode="after")
    def _ordered(self):
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
    study_designs: list[str] = Field(
        description="Eligible study designs, e.g. 'randomized controlled trial', 'cohort study'.")
    rationale: str = Field(description="Two or three sentences on how the question was framed.")


class Protocol(ProtocolDraft):
    """Full protocol = question + LLM draft + scope from the template."""

    question: str
    databases: list[Database] = Field(min_length=1)
    years: YearRange = Field(default_factory=lambda: YearRange())
    languages: list[str] = Field(default_factory=lambda: ["English"])


class Template(BaseModel):
    """Contents of protocols/template.yaml. Scope fields are applied as is;
    the other fields are hints for the Planner (may be left empty)."""

    databases: list[Database] = Field(default_factory=lambda: ["openalex", "pubmed", "arxiv"],
                                      min_length=1)
    years: YearRange = Field(default_factory=lambda: YearRange())
    languages: list[str] = Field(default_factory=lambda: ["English"])
    study_designs: list[str] = Field(default_factory=list)
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    pico: dict[str, str | list[str]] = Field(default_factory=dict)


def load_template(path: str | Path) -> Template:
    return Template.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def deep_merge(base: dict, edits: dict) -> dict:
    """Return ``base`` with ``edits`` applied; nested dicts are merged, lists replaced."""
    out = dict(base)
    for k, v in edits.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
