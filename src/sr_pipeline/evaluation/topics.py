"""Evaluation topics (``evaluation/topics.yaml``): seed the protocol with the gold review's scope.

An evaluation run must use the same question, designs, search window, languages
and eligibility criteria as the published review it is compared with; otherwise
recall measures the difference between two protocols, not the pipeline. Seen in
Part 6: the Planner wrote "aged 65 or older" for D0, then T3 (the review: over 60) and
that one criterion drove most exclusions.

Two places use a topic:
- ``topic_template``: before the Planner, the topic's scope (years, languages,
  designs) goes into the template, and its criteria become hints, so the PICO
  the Planner drafts fits them;
- ``seed_protocol``: after the draft, the topic's values replace the Planner's
  exactly (question, study designs, years, languages, and the criteria when the
  topic has them). The protocol records where its criteria come from
  (``criteria_source``), which the approval step shows; the human can still
  edit everything there.

A criterion in ``topics.yaml`` is a string, or ``{text, stage, test, label}`` (Part 6c;
``label``: Part 9b, the short exclusion reason of the PRISMA diagram):
``stage: full_text`` = an abstract can never show it (checked only at full text);
``test`` = one positive sentence the screeners judge, while the review's wording
stays the one displayed. ``load_topic`` flattens them into the text lists plus
``full_text_criteria`` and ``criterion_tests``.

Part 10: a topic may also give ``extraction_fields`` ({key: question}) and
``checklist_domains`` (the review's own risk-of-bias instrument), copied into the protocol.

T1 has no criteria in ``topics.yaml`` yet: their runs keep the Planner's
criteria, and ``criteria_source`` says so.
"""

from pathlib import Path

import yaml

from ..protocol import Protocol, Template, YearRange

TOPICS_FILE = Path(__file__).resolve().parents[3] / "evaluation" / "topics.yaml"
PLANNER_SOURCE = "Planner (LLM draft)"


def load_topic(topic_id: str, path: str | Path | None = None) -> dict:
    """One topic of topics.yaml, with its id. Raises KeyError for an unknown id."""
    topics = yaml.safe_load(Path(path or TOPICS_FILE).read_text()) or {}
    # Keys starting with "_" hold shared YAML anchors (e.g. _engineering_fields), not topics.
    topics = {k: v for k, v in topics.items() if not str(k).startswith("_")}
    if topic_id not in topics:
        raise KeyError(f"unknown topic {topic_id!r} (known: {', '.join(topics)})")
    topic = {"id": topic_id, **topics[topic_id]}
    full_text, tests, labels = [], {}, {}
    for key in ("inclusion_criteria", "exclusion_criteria"):
        texts = []
        for c in topic.get(key) or []:
            if isinstance(c, str):
                texts.append(c)
                continue
            texts.append(c["text"])
            if c.get("stage", "any") == "full_text":
                full_text.append(c["text"])
            elif c.get("stage", "any") != "any":
                raise ValueError(f"{topic_id}: stage must be 'any' or 'full_text': {c}")
            if c.get("test"):
                tests[c["text"]] = c["test"]
            if c.get("label"):
                labels[c["text"]] = c["label"]
        if key in topic:
            topic[key] = texts
    topic["full_text_criteria"], topic["criterion_tests"] = full_text, tests
    topic["criterion_labels"] = labels
    return topic


def topic_template(template: Template, topic: dict) -> Template:
    """Template with the topic's scope and criteria (scope applied as is, criteria as hints)."""
    update = {"years": YearRange.model_validate(topic.get("years") or {}),
              "study_designs": list(topic.get("study_designs") or [])}
    if topic.get("languages"):
        update["languages"] = list(topic["languages"])
    for key in ("inclusion_criteria", "exclusion_criteria"):
        if topic.get(key):
            update[key] = list(topic[key])
    return template.model_copy(update=update)


def criteria_source(topic: dict) -> str:
    if topic.get("inclusion_criteria") or topic.get("exclusion_criteria"):
        return f"topics.yaml {topic['id']} (review {topic.get('review', '?')})"
    return f"{PLANNER_SOURCE}; topics.yaml {topic['id']} has no criteria"


def seed_protocol(protocol: dict, topic: dict) -> tuple[dict, dict]:
    """(seeded protocol, audit entry). Overrides only what the topic defines."""
    out = dict(protocol)
    fields = ["question", "study_designs", "years"]
    out["question"] = topic["question"]
    out["study_designs"] = list(topic.get("study_designs") or [])
    out["years"] = YearRange.model_validate(topic.get("years") or {}).model_dump(mode="json")
    if topic.get("languages"):
        out["languages"] = list(topic["languages"])
        fields.append("languages")
    for key in ("inclusion_criteria", "exclusion_criteria"):
        if topic.get(key):
            out[key] = list(topic[key])
            fields.append(key)
    if topic.get("inclusion_criteria") or topic.get("exclusion_criteria"):
        # The topic's criteria come with their own tags: the Planner's do not apply.
        out["full_text_criteria"] = list(topic.get("full_text_criteria") or [])
        out["criterion_tests"] = dict(topic.get("criterion_tests") or {})
        out["criterion_labels"] = [{"criterion": t, "label": lab} for t, lab in
                                   (topic.get("criterion_labels") or {}).items()]
        fields += ["full_text_criteria", "criterion_tests", "criterion_labels"]
    if topic.get("outcome_directions"):
        given = {str(k): str(v) for k, v in topic["outcome_directions"].items()}
        out["outcome_directions"] = [{"outcome": k, "better": v} for k, v in given.items()] + [
            d for d in out.get("outcome_directions") or [] if d["outcome"] not in given]
        fields.append("outcome_directions")
    # Part 10 (engineering topics): extra extraction fields and the review's own
    # risk-of-bias domains (``[id, name, question]`` triples in topics.yaml).
    if topic.get("extraction_fields"):
        out["extraction_fields"] = {str(k): str(v) for k, v in
                                    topic["extraction_fields"].items()}
        fields.append("extraction_fields")
    if topic.get("checklist_domains"):
        out["checklist_name"] = str(topic.get("checklist_name") or
                                    f"Risk-of-bias domains of {topic.get('review', 'the review')}")
        out["checklist_domains"] = [{"id": str(d[0]), "name": str(d[1]),
                                     "question": str(d[2])}
                                    for d in topic["checklist_domains"]]
        fields += ["checklist_name", "checklist_domains"]
    out["topic"] = topic["id"]
    out["criteria_source"] = criteria_source(topic)
    out = Protocol.model_validate(out).model_dump(mode="json")
    audit = {"event": "protocol seeded from topics.yaml", "topic": topic["id"],
             "review": topic.get("review"), "seeded_fields": fields,
             "criteria_source": out["criteria_source"]}
    return out, audit
