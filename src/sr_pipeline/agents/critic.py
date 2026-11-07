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
- **abstract-only studies flagged**: when some studies were extracted from the abstract
  only, the text must say so (warning; the citations already carry a dagger);
- **synthesis errors**: an LLM error left no text (blocker).

Then one LLM pass (``agent="critic"``) reads the sentences next to the evidence tables and
lists claims the tables do not support (e.g. "exercise was effective" from vote counting,
a direction reversed). An LLM issue is a blocker only if it quotes the evidence line that
contradicts the sentence and code finds that quote in the tables; otherwise it is a
warning (the model may be wrong; a warning never triggers a loop).

**Loop-back** (plan: only blockers loop, at most 2 loops): ``loop_back`` is True when there
are blockers and the synthesis has been re-run fewer than ``max_loops`` times since the
last extraction. The Coordinator is told; it cannot loop when ``loop_back`` is False (code
rule in ``coordinator.py``). After the last loop the issues stay in the report.
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..llm import LLM
from ..state import ReviewState, stage_round
from .extraction import outcomes
from .screening import _sum_usage, quote_found

MAX_LOOPS = 2

NON_RCT = {"non-randomized intervention study", "diagnostic accuracy study", "other"}


class Issue(BaseModel):
    sentence: int = Field(description="Number of the sentence (as listed).")
    problem: str = Field(description="One sentence: what the text claims that the evidence "
                         "does not support.")
    evidence: str = Field("", description="Exact words copied from the evidence tables that "
                          "contradict the sentence; empty if the problem is a missing "
                          "nuance rather than a contradiction.")
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
        "exact words of the table line that contradicts it in 'evidence'. Do not list "
        "style issues. An empty list is a correct answer.\n\n"
        f"Sentences:\n{sents or '(none)'}\n\nEvidence tables:\n{synthesis['evidence']}"
    )


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

    # Unsupported claims (code check of the text)
    for i, s in enumerate(sents, 1):
        for f in s["flags"]:
            kind = f.split(":")[0]
            sev = "warning" if kind in ("no_citation", "unknown_section") else "blocker"
            issues.append(_issue(sev, kind, f"sentence {i} ('{s['text'][:120]}'): {f}",
                                 sentence=i))

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
                    sev = "blocker" if it.severity == "blocker" and verified else "warning"
                    text = syn["narrative"][it.sentence - 1]["text"]
                    issues.append(_issue(sev, "llm_unsupported_claim",
                                         f"sentence {it.sentence} ('{text[:120]}'): "
                                         f"{it.problem}", sentence=it.sentence,
                                         evidence=it.evidence, evidence_verified=verified,
                                         severity_claimed=it.severity))
            except Exception as exc:  # the code checks still stand
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
