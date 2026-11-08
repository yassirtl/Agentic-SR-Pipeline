"""Critic: quality assurance of the synthesis -> ``qa_report``.

Checks by code (deterministic):

- **protocol adherence**: every PICO outcome has at least one sentence (blocker); a study
  whose verified design does not match the protocol's designs (warning: screening let it
  through on D, the reader should know);
- **unsupported claims**: the synthesis check flags (``check_narrative``): a number not
  printed in the evidence tables or an unknown study label (blocker), a sentence without a
  citation (warning);
- **count consistency**: studies in the synthesis = ``included`` = PRISMA full-text
  "studies included" (blocker);
- **missing studies**: an included study never cited (blocker);
- **heterogeneity** (Part 8b): every outcome has a heterogeneity statement, giving
  I2 / tau2 when the outcome was pooled (blocker);
- **metric conflicts** (Part 8b): a study whose direction depends on the metric is named
  in a sentence that says so (blocker); a study against the majority direction is a
  warning (a human checks the source; values are never corrected);
- **wording vs heterogeneity** (Part 9b): when a pooled result has I2 >= 75% (not
  interpretable as a single effect), an effect-size word about it ("negligible", "small",
  "moderate", "large") or "no effect" is a blocker; on the narrative path with a sign test
  that is not significant, "no effect" is a blocker ("no evidence of a direction");
- **abstract-only studies flagged**: when some studies were extracted from the abstract
  only, the text must say so (warning; the citations already carry a dagger);
- **synthesis errors**: an LLM error left no text (blocker).

Then one LLM pass (``agent="critic"``) reads the sentences next to the evidence tables and
lists claims the tables do not support (e.g. "exercise was effective" from vote counting,
a direction reversed). **An LLM objection is a warning by default** (Part 9 rule): blockers
come from code checks. It is promoted to a blocker only when it names a value code can
check and code confirms the mismatch (``confirm_mismatch``): the value the sentence gets
wrong (a number or a count, digits or words) is written in the sentence, the evidence
quote is found in the tables, it prints the value the LLM says is right, and it does not
print the sentence's value. Reason: in the Part 8b run both extra loops were spent on LLM
objections whose quote was a real line that actually supported the sentence; the old
rule (verified quote = blocker) could not tell. Objections about direction, wording or
nuance stay warnings and go to the report's QA section.

**Loop-back** (plan: only blockers loop, at most 2 loops): ``loop_back`` is True when there
are blockers and the synthesis has been re-run fewer than ``max_loops`` times since the
last extraction. The Coordinator is told; it cannot loop when ``loop_back`` is False (code
rule in ``coordinator.py``). After the last loop the issues stay in the report.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..llm import LLM, is_fatal
from ..state import ReviewState, stage_round
from .extraction import numbers, outcomes
from .screening import _sum_usage, quote_found
from .synthesis import number_supported, wording_problems

MAX_LOOPS = 2

CONFLICT_WORDS = re.compile(r"conflict|opposite|change from baseline|change score|"
                            r"depend|disagree|inconsistent", re.I)

I2_WORDS = re.compile(r"\bI\s?(?:2|²|\^2|-squared)|\btau", re.I)

NON_RCT = {"non-randomized intervention study", "diagnostic accuracy study", "other"}


class Issue(BaseModel):
    sentence: int = Field(description="Number of the sentence (as listed).")
    problem: str = Field(description="One sentence: what the text claims that the evidence "
                         "does not support.")
    evidence: str = Field("", description="Exact words copied from the evidence tables that "
                          "contradict the sentence; empty if the problem is a missing "
                          "nuance rather than a contradiction.")
    wrong_value: str = Field("", description="If the sentence states a wrong number or "
                             "count: that number exactly as the sentence writes it (e.g. "
                             "'one', '0.48'); else empty.")
    table_value: str = Field("", description="The number the evidence line gives instead "
                             "(it must be printed in 'evidence'); else empty.")
    severity: Literal["blocker", "warning"] = Field(
        description="blocker = the sentence is wrong (contradicted, invented, reversed); "
                    "warning = imprecise or incomplete.")


class Review(BaseModel):
    issues: list[Issue] = Field(default_factory=list, description="Empty if every sentence "
                                "is supported by the evidence tables.")


def critic_prompt(protocol: dict, synthesis: dict) -> str:
    sents = "\n".join(f"{i}. [{s['section']}] {s['text']} (studies: "
                      f"{', '.join(s['studies']) or 'none'})"
                      for i, s in enumerate(synthesis["narrative"], 1))
    return (
        "You are the critic of a PRISMA 2020 systematic review. Check the synthesis "
        "sentences against the evidence tables only.\n"
        f"Review question: {protocol.get('question', '')}\n\n"
        "List a sentence only if it claims something the tables do not support, e.g.: a "
        "result in the wrong direction; an effect called significant or effective from "
        "individual p values or from vote counting without a significant sign test; a "
        "pooled result when no meta-analysis was done; a study described with the wrong "
        "result or risk of bias; a number not in the tables. For a contradiction, copy the "
        "exact words of the table line that contradicts it in 'evidence'. If the sentence "
        "gets a number or a count wrong, give the sentence's number in 'wrong_value' and "
        "the table's number in 'table_value'. A line that agrees with the sentence is not "
        "a contradiction. Do not list style issues. An empty list is a correct answer.\n\n"
        f"Sentences:\n{sents or '(none)'}\n\nEvidence tables:\n{synthesis['evidence']}"
    )


_LABEL = re.compile(r"\bS\d+\b")


def _nums(text: str) -> list[float]:
    """Numbers in digits or words, study labels (S12) left out, signs ignored."""
    return numbers(_LABEL.sub(" ", text or ""))


def confirm_mismatch(sentence: str, wrong_value: str, table_value: str, evidence: str,
                     tables: str) -> tuple[bool, str]:
    """Code check of an LLM objection (see module docstring). Returns (confirmed, why).

    Conservative: when the quote also prints the sentence's value (e.g. "high 2, some
    concerns 1" against "one study was high risk") code cannot tell which number the
    sentence meant, so the objection stays a warning."""
    if not (wrong_value.strip() and table_value.strip()):
        return False, "no checkable value named"
    wrong, right = _nums(wrong_value), _nums(table_value)
    if len(wrong) != 1 or len(right) != 1:
        return False, "value is not a single number"
    w, t = wrong[0], right[0]
    if abs(w - t) < 1e-9:
        return False, "sentence value equals the table value"
    if not any(abs(w - n) < 1e-9 for n in _nums(sentence)):
        return False, f"{w:g} is not written in the sentence"
    if not (evidence.strip() and quote_found(evidence, tables)):
        return False, "evidence quote not found in the tables"
    quoted = _nums(evidence)
    if not number_supported(f"{t:g}", quoted):
        return False, f"{t:g} is not printed in the quote"
    if number_supported(f"{w:g}", quoted):
        return False, f"the quote also prints {w:g}"
    return True, f"sentence says {w:g}, the quoted line gives {t:g}"


def _issue(severity: str, kind: str, detail: str, **extra) -> dict:
    return {"severity": severity, "kind": kind, "detail": detail, **extra}


def code_checks(state: ReviewState) -> list[dict]:
    protocol, syn = state["protocol"], state["synthesis"]
    issues = []
    sents = syn["narrative"]
    if syn.get("flags"):
        issues.append(_issue("blocker", "synthesis_error", "; ".join(syn["flags"])))

    # Protocol adherence
    for oid, outcome in outcomes(protocol):
        if not any(s["section"] == oid for s in sents):
            issues.append(_issue("blocker", "outcome_not_addressed",
                                 f"outcome {oid} ({outcome}) has no sentence"))
    designs = " ".join(protocol.get("study_designs") or []).lower()
    if "randomi" in designs and "non-randomi" not in designs:
        for s in syn["studies"]:
            if s["design_verified"] and s["design_category"] in NON_RCT:
                issues.append(_issue("warning", "design_mismatch",
                                     f"{s['label']} is a {s['design_category']} (protocol: "
                                     f"{', '.join(protocol['study_designs'])})"))

    # Heterogeneity statement per outcome (SWiM item 6, Part 8b fix 3)
    het = syn.get("heterogeneity") or []
    for oid, outcome in outcomes(protocol):
        mine = [h for h in het if h["section"] == oid]
        if not mine:
            issues.append(_issue("blocker", "heterogeneity_missing",
                                 f"outcome {oid} ({outcome}) has no heterogeneity statement"))
            continue
        pooled = next((g for g in syn["outcomes"] if g["id"] == oid and g.get("meta")), None)
        if pooled and not any(I2_WORDS.search(h["text"]) for h in mine):
            m = pooled["meta"]
            issues.append(_issue("blocker", "heterogeneity_without_i2",
                                 f"outcome {oid} was pooled: its heterogeneity statement must "
                                 f"give I2 {100 * m['i2']:.0f}% / tau2 {m['tau2']:.3f}"))

    # Unsupported claims (code check of the text)
    for i, s in enumerate(sents, 1):
        for f in s["flags"]:
            kind = f.split(":")[0]
            sev = "warning" if kind in ("no_citation", "unknown_section") else "blocker"
            issues.append(_issue(sev, kind, f"sentence {i} ('{s['text'][:120]}'): {f}",
                                 sentence=i))
    for h in het:
        for f in h["flags"]:
            kind = f.split(":")[0]
            sev = "warning" if kind in ("no_citation", "unknown_section") else "blocker"
            issues.append(_issue(sev, kind, f"heterogeneity {h['section']} "
                                 f"('{h['text'][:120]}'): {f}"))

    # Metric conflicts must be named in the text (Part 8b fix 2); a study whose direction
    # is opposite to the majority is shown for a human check of the source (never fixed).
    for g in syn["outcomes"]:
        for label in g.get("conflicts") or []:
            if not any(label in x["studies"] and CONFLICT_WORDS.search(x["text"])
                       for x in [*sents, *het]):
                issues.append(_issue("blocker", "conflict_not_mentioned",
                                     f"{label} ({g['id']}): its direction depends on the "
                                     "metric (METRIC CONFLICT in the tables) and no sentence "
                                     "citing it says so"))
        for label in g.get("against_majority") or []:
            issues.append(_issue("warning", "direction_against_majority",
                                 f"{label} ({g['id']}) points the other way from the majority "
                                 f"({g['majority']}): check the values in the source"))

    # Wording the numbers do not allow (Part 9b fix 3): this outcome's sentences, its
    # heterogeneity statement and the 'overall' sentences.
    for g in syn["outcomes"]:
        texts = ([x for x in sents if x["section"] in (g["id"], "overall")]
                 + [h for h in het if h["section"] == g["id"]])
        for w in wording_problems(g, texts):
            issues.append(_issue("blocker", w["kind"], f"{g['id']}: '{w['text'][:120]}' "
                                 f"says '{w['words']}'; {w['reason']}"))

    # Count consistency
    n_syn = len(syn["studies"])
    n_inc = len(dict.fromkeys(state.get("included") or []))
    prisma = (state.get("prisma_counts") or {}).get("full_text", {}).get("studies_included")
    if n_syn != n_inc or (prisma is not None and prisma != n_inc):
        issues.append(_issue("blocker", "count_mismatch",
                             f"synthesis {n_syn} studies, included {n_inc}, PRISMA full-text "
                             f"studies included {prisma}"))

    # Missing studies
    cited = {c for s in sents for c in s["studies"]}
    for s in syn["studies"]:
        if s["label"] not in cited:
            issues.append(_issue("blocker", "study_not_cited",
                                 f"{s['label']} ({s['author']} {s['year'] or ''}, "
                                 f"{s['title'][:60]}) is included but never cited"))

    # Abstract-only studies flagged in the text
    abstract = [s["label"] for s in syn["studies"] if s["abstract_only"]]
    if abstract and not any("abstract" in s["text"].lower() for s in sents):
        issues.append(_issue("warning", "abstract_only_not_mentioned",
                             f"{len(abstract)} studies were extracted from the abstract only "
                             f"({', '.join(abstract)}); the text does not say so"))
    return issues


def loops_used(completed: list[str]) -> int:
    """Synthesis re-runs since the last extraction (0 after the first synthesis)."""
    last_ext = max((i for i, s in enumerate(completed) if s == "extraction"), default=-1)
    return max(0, completed[last_ext + 1:].count("synthesis") - 1)


def make_critic(llm: LLM, *, max_loops: int = MAX_LOOPS):
    def critic(state: ReviewState) -> dict:
        protocol, syn = state["protocol"], state["synthesis"]
        completed = state.get("completed_stages", [])
        rnd = stage_round("critic", completed)
        issues = code_checks(state)
        usage = []
        if syn["narrative"]:
            try:
                ans, u = llm.call_with_usage(Review, critic_prompt(protocol, syn),
                                             agent="critic")
                usage.append(u)
                for it in ans.issues:
                    if not 1 <= it.sentence <= len(syn["narrative"]):
                        continue
                    verified = bool(it.evidence.strip()) and quote_found(it.evidence,
                                                                         syn["evidence"])
                    text = syn["narrative"][it.sentence - 1]["text"]
                    confirmed, why = confirm_mismatch(text, it.wrong_value, it.table_value,
                                                      it.evidence, syn["evidence"])
                    # Warning by default; blocker only when code confirms a wrong value.
                    sev = "blocker" if confirmed else "warning"
                    issues.append(_issue(sev, "llm_unsupported_claim",
                                         f"sentence {it.sentence} ('{text[:120]}'): "
                                         f"{it.problem}", sentence=it.sentence,
                                         evidence=it.evidence, evidence_verified=verified,
                                         wrong_value=it.wrong_value,
                                         table_value=it.table_value,
                                         confirmed_by_code=confirmed, check=why,
                                         severity_claimed=it.severity))
            except Exception as exc:  # the code checks still stand
                if is_fatal(exc):
                    raise
                issues.append(_issue("warning", "critic_llm_error",
                                     f"{type(exc).__name__}: {exc}"[:300]))
        blockers = [i for i in issues if i["severity"] == "blocker"]
        used = loops_used(completed)
        report = {
            "round": rnd, "synthesis_round": syn["round"], "passed": not blockers,
            "blockers": len(blockers), "warnings": len(issues) - len(blockers),
            "issues": issues, "loops_used": used, "max_loops": max_loops,
            "loop_back": bool(blockers) and used < max_loops,
        }
        if blockers and not report["loop_back"]:
            report["note"] = (f"loop budget spent ({used}/{max_loops}): the remaining "
                              "issues go to the report")
        return {
            "qa_report": report,
            "completed_stages": ["critic"],
            "audit_log": [{"node": "critic", "step": state.get("steps_taken", 0),
                           "round": rnd, "event": "quality check",
                           "synthesis_round": syn["round"], "passed": report["passed"],
                           "blockers": report["blockers"], "warnings": report["warnings"],
                           "kinds": sorted({i["kind"] for i in issues}),
                           "loop_back": report["loop_back"], "loops_used": used,
                           **_sum_usage(usage)}],
        }

    return critic


def describe_qa(state: ReviewState) -> list[str]:
    """Lines for the Coordinator summary."""
    qa = state.get("qa_report")
    if not qa:
        return []
    lines = [f"Critic (round {qa['round']}, on synthesis round {qa['synthesis_round']}): "
             f"passed={qa['passed']}, blockers {qa['blockers']}, warnings {qa['warnings']}; "
             f"loop_back={qa['loop_back']} (loops used {qa['loops_used']}/{qa['max_loops']})"]
    blockers = [i for i in qa["issues"] if i["severity"] == "blocker"][:5]
    if blockers:
        lines.append("Blocker issues: " + "; ".join(f"{i['kind']}: {i['detail'][:100]}"
                                                    for i in blockers))
    return lines
