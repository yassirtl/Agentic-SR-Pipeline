"""PRISMA 2020 checklist validator (Part 12): checks the finished article against the 27
items (42 with their parts) of the PRISMA 2020 checklist (Page et al., BMJ 2021;372:n71).

Code only. Each item is checked on the article's Markdown (its sections and the words that
show the item is there) and, where the article cannot show it, on the run's state. Status:
``reported``, ``partly`` or ``not reported``, with the section where it is (or why not).
The result is printed as the article's last appendix and written to
``prisma_checklist.json``. Items the pipeline does not do (sensitivity analyses,
reporting-bias assessment, certainty of evidence) are reported as such, never hidden; items
only the authors can write (funding, competing interests) stay "not reported" until they do.
"""

from __future__ import annotations

import re

REPORTED, PARTLY, MISSING = "reported", "partly", "not reported"


def sections(md: str) -> dict[str, str]:
    """{heading text (lower case): its text up to the next heading of any level}."""
    out, cur, buf = {}, "_top", []
    for line in md.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)$", line)
        if m:
            out[cur] = "\n".join(buf)
            cur, buf = m.group(1).strip().lower(), []
        else:
            buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def _find(secs: dict[str, str], *names: str) -> str | None:
    """The first heading that starts with one of ``names`` (lower case)."""
    for n in names:
        for h in secs:
            if h.startswith(n):
                return h
    return None


def _has(text: str | None, *patterns: str) -> bool:
    return bool(text) and any(re.search(p, text, re.I) for p in patterns)


def check(md: str, state: dict) -> list[dict]:
    secs = sections(md)
    title = next((l[2:] for l in md.splitlines() if l.startswith("# ")), "")
    sec = lambda *n: _find(secs, *n)                          # noqa: E731
    txt = lambda h: secs.get(h, "") if h else ""              # noqa: E731
    abstract, intro = sec("abstract"), sec("introduction")
    elig, sources = sec("protocol and eligibility"), sec("information sources")
    select, data_items = sec("selection process"), sec("data items")
    collect, synth_m = sec("data collection"), sec("synthesis methods")
    selection_r, chars = sec("study selection"), sec("study characteristics")
    rob_r, synth_r = sec("risk of bias"), sec("synthesis of results")
    disc, lim, agenda = sec("discussion"), sec("limitations"), sec("research agenda")
    other, gaps = sec("other information"), sec("evidence gaps")
    syn = state.get("synthesis") or {}
    pooled = any(g.get("meta") for g in syn.get("outcomes") or [])
    items: list[dict] = []

    def add(item, section, topic, status, where):
        items.append({"item": item, "section": section, "topic": topic, "status": status,
                      "where": where})

    def at(h, status=REPORTED, why=""):
        """``status`` at section ``h``; no such section: not reported, with ``why``."""
        return (status, h.capitalize()) if h else (MISSING, why)

    add("1", "Title", "Identify the report as a systematic review",
        *((REPORTED, "Title") if _has(title, r"systematic review") else (MISSING, "title")))
    abs_t = txt(abstract)
    add("2", "Abstract", "Structured abstract (PRISMA 2020 for Abstracts)",
        *at(abstract, REPORTED if all(_has(abs_t, rf"\*\*{w}") for w in
                                      ("Background", "Methods", "Results", "Conclusions"))
            else PARTLY, "no abstract"))
    add("3", "Introduction", "Rationale", *at(intro, why="no Introduction"))
    add("4", "Introduction", "Objectives or question",
        *at(intro if _has(txt(intro), r"\?|aim|objective|asks") else None,
            why="no objective stated"))
    add("5", "Methods", "Eligibility criteria", *at(elig, why="no criteria section"))
    add("6", "Methods", "Information sources and dates searched",
        *at(sources if _has(txt(sources), r"Searched") else None, why="no sources section"))
    add("7", "Methods", "Full search strategies",
        *at(sources if _has(txt(sources), r"\bAND\b|\+") else None, why="no query shown"))
    add("8", "Methods", "Selection process (reviewers, automation tools)",
        *at(select, why="no selection section"))
    add("9", "Methods", "Data collection process", *at(collect, why="no data collection"))
    add("10a", "Methods", "Data items: outcomes",
        *at(elig if _has(txt(elig), r"outcome") else None, why="outcomes not listed"))
    add("10b", "Methods", "Data items: other variables",
        *at(data_items or collect, why="variables not listed"))
    add("11", "Methods", "Study risk of bias assessment",
        *at(collect if _has(txt(collect), r"risk of bias") else None, why="no method"))
    add("12", "Methods", "Effect measures",
        *at(synth_m if _has(txt(synth_m), r"Hedges|direction of effect|sensitivity") else None,
            why="no effect measure"))
    add("13a", "Methods", "Which studies were eligible for each synthesis",
        *at(synth_m, PARTLY, "no synthesis methods"))
    add("13b", "Methods", "Data preparation for synthesis",
        *at(synth_m if _has(txt(synth_m), r"metric order|per-arm|verified") else None,
            why="not described"))
    add("13c", "Methods", "Tabulation and visual display",
        *at(synth_m if _has(md, r"Evidence map|Forest plot|forest") else None,
            REPORTED, "no display described"))
    add("13d", "Methods", "Synthesis methods and rationale", *at(synth_m, why="none"))
    add("13e", "Methods", "Exploring causes of heterogeneity",
        *at(synth_m if _has(txt(synth_m), r"heterogeneity") else None, why="not described"))
    add("13f", "Methods", "Sensitivity analyses", MISSING, "not done by the pipeline")
    add("14", "Methods", "Reporting bias assessment", MISSING, "not done by the pipeline")
    add("15", "Methods", "Certainty assessment (e.g. GRADE)", MISSING,
        "not done by the pipeline")
    add("16a", "Results", "Study selection and flow diagram",
        *at(selection_r if _has(txt(selection_r), r"PRISMA") else None, why="no diagram"))
    add("16b", "Results", "Excluded studies that seemed to meet the criteria",
        *((REPORTED, "Appendix C (every full-text exclusion with its criterion)")
          if _has(md, r"Appendix C\. Studies excluded at full text")
          else (PARTLY, "Study selection (exclusion reasons by criterion in the diagram)")))
    add("17", "Results", "Study characteristics", *at(chars, why="no table"))
    add("18", "Results", "Risk of bias in studies",
        *at(rob_r if _has(md, r"Appendix B") else None, why="no per-study judgements"))
    add("19", "Results", "Results of individual studies",
        *at(chars, PARTLY, "no per-study results"))
    add("20a", "Results", "Characteristics and risk of bias of each synthesis",
        *at(synth_r, PARTLY, "no synthesis"))
    add("20b", "Results", "Results of syntheses",
        *at(synth_r if pooled or _has(txt(synth_r), r"Vote count|narrative") else None,
            why="no synthesis results"))
    add("20c", "Results", "Investigations of heterogeneity",
        *at(synth_r if _has(md, r"Heterogeneity") else None, PARTLY, "none"))
    add("20d", "Results", "Sensitivity analyses", MISSING, "not done by the pipeline")
    add("21", "Results", "Reporting biases", MISSING, "not assessed")
    add("22", "Results", "Certainty of evidence", MISSING, "not assessed")
    add("23a", "Discussion", "Interpretation in the context of other evidence",
        *at(disc, PARTLY if not _has(txt(disc), r"prior|published|review") else REPORTED,
            "no Discussion"))
    add("23b", "Discussion", "Limitations of the evidence",
        *at(lim if _has(txt(lim), r"abstract only|risk of bias") else None, why="none"))
    add("23c", "Discussion", "Limitations of the review processes", *at(lim, why="none"))
    add("23d", "Discussion", "Implications for practice, policy and research",
        *at(agenda or (disc if _has(txt(disc), r"future|research") else None),
            why="none"))
    oth = txt(other)
    add("24a", "Other information", "Registration",
        *at(other if _has(oth, r"Registration") else None, why="not stated"))
    add("24b", "Other information", "Access to the protocol",
        *at(other if _has(oth, r"protocol") else None, why="not stated"))
    add("24c", "Other information", "Amendments to the protocol",
        *at(other if _has(oth, r"amend") else None, why="not stated"))
    add("25", "Other information", "Support (funding)",
        *((MISSING, "to be completed by the authors") if _has(oth, r"Support.*\[to be")
          else at(other if _has(oth, r"Support") else None, why="not stated")))
    add("26", "Other information", "Competing interests",
        *((MISSING, "to be completed by the authors") if _has(oth, r"competing interests.*\[to be")
          else at(other if _has(oth, r"competing") else None, why="not stated")))
    add("27", "Other information", "Availability of data, code and other materials",
        *at(other if _has(oth, r"Availability") else None, why="not stated"))
    if gaps:
        for it in items:
            if it["item"] == "23d" and it["status"] == REPORTED and agenda:
                it["where"] = "Research agenda (from the Evidence gaps)"
    return items


def summary(items: list[dict]) -> dict:
    n = len(items)
    return {"items": n, **{s: sum(i["status"] == s for i in items)
                           for s in (REPORTED, PARTLY, MISSING)}}


def appendix(items: list[dict]) -> list[str]:
    s = summary(items)
    rows = "\n".join(f"| {i['item']} | {i['section']} | {i['topic']} | {i['status']} | "
                     f"{i['where']} |" for i in items)
    return ["## Appendix D. PRISMA 2020 checklist (checked by code)", "",
            f"Each of the {s['items']} PRISMA 2020 checklist items (27 items with their parts) "
            f"was checked in this article by code: {s[REPORTED]} reported, {s[PARTLY]} partly, "
            f"{s[MISSING]} not reported.", "",
            "| Item | Section | Topic | Status | Where / why |", "|---|---|---|---|---|", rows]
