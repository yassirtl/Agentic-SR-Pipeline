"""Recall against a gold standard, stage by stage (Part 10).

A published review's included studies (``evaluation/gold/T*.csv``) are followed
through our run, and each one that is lost gets the stage and the reason:

  gold → covered    exists in a database we search (OpenAlex or PubMed): coverage CSV
       → found      among the run's unique records (the search + retrieval found it)
       → screened   inside the screening limit (``--limit``): has a title/abstract decision
       → ta_included   kept at title/abstract
       → included   kept at full text: in the run's final included studies

So a loss has one cause: "not covered" is the database choice, "not found" is the
query (or the download cap), "not screened" is the screening limit (Part 9b: a
record above the limit is never an exclusion), and the two screening stages carry
the deciding criterion. Recall is given against all gold studies and against the
covered ones (the reviews searched Embase, Cochrane, CNKI...).

Precision against the gold list is a **lower bound**: a study we include that the
review did not may still be eligible (other search date edge, other reading of the
criteria), so it is reported as "gold share of our includes", not as an error rate.

Matching a gold row to a paper: DOI, then PMID (both also from the coverage lookup,
``found_doi`` / ``found_pmid``), then title (fuzzy ≥ 90, year ±1, no conflicting DOI,
the same rule as the coverage lookup). ``matched_by`` says which.
"""

from pathlib import Path

from ..agents.screening import criterion_labels
from ..reporting.tables import screening_rows
from ..sources.base import normalize_doi
from .gold import _doi_ok, _year_ok, load_gold, title_matches

STAGES = ("gold", "covered", "found", "screened", "ta_included", "included")


def gold_file(path: str | Path) -> Path:
    """The ``*_coverage.csv`` next to a gold file when it exists (it has the coverage
    columns and the identifiers the review did not print), else the file itself."""
    path = Path(path)
    if path.stem.endswith("_coverage"):
        return path
    cov = path.with_name(f"{path.stem}_coverage.csv")
    return cov if cov.exists() else path


def is_covered(row: dict) -> bool | None:
    """True / False from the coverage columns; None when the file has none."""
    if "in_openalex" not in row and "in_pubmed" not in row:
        return None
    return row.get("in_openalex") == "yes" or row.get("in_pubmed") == "yes"


def match_paper(row: dict, papers: list[dict]) -> tuple[dict | None, str | None]:
    """(paper, matched_by) for one gold row, or (None, None)."""
    doi = normalize_doi(row.get("found_doi") or row.get("doi"))
    pmid = str(row.get("found_pmid") or row.get("pmid") or "")
    if doi:
        for p in papers:
            if p.get("doi") and normalize_doi(p["doi"]) == doi:
                return p, "doi"
    if pmid:
        for p in papers:
            if str(p.get("pmid") or "") == pmid:
                return p, "pmid"
    if row.get("title"):
        for p in papers:
            if title_matches(p.get("title") or "", row["title"]) and \
                    _year_ok(row, p.get("year")) and _doi_ok(row, p.get("doi")):
                return p, "title"
    return None, None


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def gold_funnel(state: dict, gold_rows: list[dict]) -> dict:
    """Per gold study: the last stage reached and why it went no further; counts per
    stage; recall vs all and vs covered; precision lower bound."""
    papers = state.get("unique_records") or []
    screening = {r["paper_id"]: r for r in screening_rows(state)}
    included = set(state.get("included") or [])
    labels = criterion_labels(state.get("protocol") or {})
    ta_counts = (state.get("prisma_counts") or {}).get("title_abstract") or {}
    limit = ta_counts.get("screen_limit")
    dbs = ((state.get("prisma_counts") or {}).get("identification") or {}).get("databases") or {}
    capped = sorted(d for d, v in dbs.items() if v.get("capped"))
    not_found = ("not among the run's records: the query missed it" if not capped else
                 f"not among the run's records: query, or the download cap of {', '.join(capped)}")

    studies = []
    for row in gold_rows:
        s = {"row": row.get("row"), "first_author": row.get("first_author"),
             "year": row.get("year"), "title": row.get("title"),
             "type": row.get("type"), "covered": is_covered(row)}
        paper, by = match_paper(row, papers)
        s.update(paper_id=paper and paper["paper_id"], matched_by=by,
                 rank=paper and paper.get("rank"))
        sc = screening.get(paper["paper_id"]) if paper else None
        reached, lost, reason = "gold", None, None
        if s["covered"] is False and not paper:
            lost, reason = "covered", "not in OpenAlex or PubMed"
        else:
            reached = "covered"
            if not paper:
                lost, reason = "found", not_found
            elif not sc or sc["ta"] == "not screened":
                reached = "found"
                lost = "screened"
                # rank = best rank in one database's relevance order; the limit takes
                # the N best ranks over all databases, so rank 56 can be outside 100.
                reason = (f"outside the {limit} records screened (best database rank "
                          f"{s['rank']})" if limit else "not screened")
            elif sc["ta"] != "include":
                reached = "screened"
                lost = "ta_included"
                reason = _criterion_reason(sc["ta_criterion"], sc["ta_resolved_by"], labels)
            elif paper["paper_id"] not in included:
                reached = "ta_included"
                lost = "included"
                reason = _criterion_reason(sc["ft_criterion"], sc["ft_resolved_by"], labels)
                if sc.get("fulltext_status"):
                    reason += f"; text: {sc['fulltext_status']}"
            else:
                reached = "included"
            if s["covered"] is False and paper:  # the lookup missed it, the search did not
                s["coverage_note"] = "found by the search although the coverage lookup missed it"
        s.update(reached=reached, lost_at=lost, reason=reason,
                 ta=sc and sc["ta"], ft=sc and sc["ft"],
                 fulltext_status=sc and sc.get("fulltext_status"))
        studies.append(s)

    order = {st: i for i, st in enumerate(STAGES)}
    counts = {st: sum(order[s["reached"]] >= order[st] for s in studies) for st in STAGES}
    n_all = counts["gold"]
    n_cov = sum(s["covered"] is not False for s in studies)
    gold_included = counts["included"]
    screened_gold = counts["screened"]
    return {
        "studies": studies,
        "counts": counts,
        "screen_limit": limit,
        "recall": {
            "search_vs_all": _ratio(counts["found"], n_all),
            "search_vs_covered": _ratio(counts["found"], n_cov),
            "final_vs_all": _ratio(gold_included, n_all),
            "final_vs_covered": _ratio(gold_included, n_cov),
            # Screening on its own: of the gold studies the screeners saw, how many kept.
            "screening_ta": _ratio(counts["ta_included"], screened_gold),
            "screening_ta_ft": _ratio(gold_included, screened_gold),
        },
        "precision_lower_bound": _ratio(gold_included, len(included)),
        "included": len(included),
        "lost_by_stage": {st: [f"{s['first_author']} {s['year']}" for s in studies
                               if s["lost_at"] == st] for st in STAGES[1:]},
    }


def _criterion_reason(cid: str | None, resolved_by: str | None, labels: dict) -> str:
    if not cid:
        return f"excluded ({resolved_by or 'no criterion'})"
    label = labels.get(cid)
    return f"excluded on {cid}" + (f" ({label})" if label else "") + \
        (f", by {resolved_by}" if resolved_by else "")


def load_gold_rows(path: str | Path) -> list[dict]:
    return load_gold(gold_file(path))
