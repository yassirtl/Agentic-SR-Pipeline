"""Study-design filters written in code, not by the LLM.

When the protocol restricts the eligible designs, the search gets one extra
block, AND-ed with the topic blocks. Design filters are a solved problem:
information specialists have built and tested them, so they are copied here
instead of being improvised by the LLM (which wrote exact phrases such as
"randomized controlled trial"[tiab], missing "randomised" and "RCT", and
"Cohort Studies"[pt], a MeSH heading used as a publication type: matches nothing).

- Randomized trials: Cochrane Highly Sensitive Search Strategy (Cochrane
  Handbook, Technical Supplement to Chapter 4, sensitivity- and
  precision-maximizing version, PubMed format):
  randomized controlled trial[pt] OR controlled clinical trial[pt] OR
  randomized[tiab] OR placebo[tiab] OR randomly[tiab] OR trial[ti]
  ("randomised" and "RCT" added: free-text fields are not spelling-normalised).
  The original's NOT animals[mh] line is left out: our blocks are only AND-ed.
- Cohort, case-control, cross-sectional: the MeSH heading ([mh] also matches
  the narrower headings, e.g. Cohort Studies -> Prospective / Longitudinal
  Studies) OR the words authors use.

Every listed design appears in the block. A design without a validated filter
is searched by its own name and flagged ``unvalidated`` so the report says so.
OpenAlex and arXiv have no publication types or MeSH: they get the keywords.
"""

import re

from .base import Concept

# key -> (pattern recognising the design in the protocol, filter fields)
FILTERS: dict[str, tuple[str, dict]] = {
    "randomized controlled trial": (
        r"\brandomi[sz]ed\b|\brct\b",
        {"keywords": ["randomized", "randomised", "randomly", "placebo", "RCT"],
         "title_terms": ["trial"],
         "publication_types": ["randomized controlled trial", "controlled clinical trial"]}),
    "cohort study": (
        r"cohort|longitudinal|prospective",
        {"keywords": ["cohort", "cohorts", "longitudinal", "prospective", "retrospective",
                      "follow-up study"],
         "mesh_terms": ["Cohort Studies"]}),
    "case-control study": (
        r"case.?control",
        {"keywords": ["case-control", "case control", "case-referent"],
         "mesh_terms": ["Case-Control Studies"]}),
    "cross-sectional study": (
        r"cross.?sectional",
        {"keywords": ["cross-sectional", "cross sectional", "prevalence study"],
         "mesh_terms": ["Cross-Sectional Studies"]}),
}
# Checked first: "non-randomized trial" must not get the RCT filter.
NOT_VALIDATED = r"non.?randomi[sz]ed|quasi"


def match_design(name: str) -> str | None:
    """Key of the validated filter for a protocol design name, or None."""
    text = name.lower()
    if re.search(NOT_VALIDATED, text):
        return None
    return next((key for key, (pattern, _) in FILTERS.items() if re.search(pattern, text)),
                None)


def design_block(designs: list[str]) -> tuple[Concept | None, list[dict]]:
    """One design block for all the protocol's designs, plus how each was mapped.

    No designs -> (None, []): no design block at all.
    """
    if not designs:
        return None, []
    fields: dict[str, list[str]] = {"keywords": [], "title_terms": [], "mesh_terms": [],
                                    "publication_types": []}
    mapping, used = [], set()
    for name in designs:
        key = match_design(name)
        mapping.append({"design": name, "filter": key or "unvalidated"})
        if key is None:
            fields["keywords"].append(name)       # searched by its own name
        elif key not in used:
            used.add(key)
            for f, values in FILTERS[key][1].items():
                fields[f] += values
    return Concept(name="design: " + ", ".join(designs), role="design", **fields), mapping
