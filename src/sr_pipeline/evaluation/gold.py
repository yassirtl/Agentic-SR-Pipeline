"""Gold standards for evaluation: the included studies of a published review, and their coverage.

A gold list (``evaluation/gold/T1.csv`` ...) is Table 1 of a published PRISMA
review, one row per included article, with the identifiers taken from the
review's own reference list (DOI, PMID, PMCID). See ``evaluation/gold/README.md``.

**Coverage** asks: does each gold article exist in the databases we search
(OpenAlex, PubMed) at all? It is not recall. The published reviews also used
Embase, Cochrane, Scopus, CNKI...; an article that is in none of our databases
can never be found by our pipeline, whatever the query. So Part 10 reports
recall twice: against all gold articles and against the covered ones.

Lookup order (cached through ``HttpClient``, so a re-run sends no request):
  OpenAlex: DOI (``/works/doi:...``), else title search (fuzzy match + year +-1)
  PubMed:   PMID (esummary), else DOI (``[doi]`` esearch), else title (``[ti]``)
A title match also yields the identifiers the review did not print (DOI, PMID):
Part 10 needs them to recognise a gold article among our screened papers.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from ..dedup import normalize_title
from ..sources.base import clean_term, normalize_doi
from ..sources.http import HttpClient, HttpError
from ..sources.openalex import BASE_URL as OPENALEX, _pmcid, _strip
from ..sources.pubmed import EUTILS, TOOL

TITLE_MATCH = 90  # rapidfuzz ratio on normalized titles
COVERAGE_COLS = ["in_openalex", "openalex_by", "openalex_id", "in_pubmed", "pubmed_by",
                 "found_doi", "found_pmid"]


def load_gold(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def title_matches(a: str, b: str) -> bool:
    return fuzz.ratio(normalize_title(a), normalize_title(b)) >= TITLE_MATCH


def _doi_ok(row: dict, doi: str | None) -> bool:
    """A title match is rejected when both sides have a DOI and they differ: that is another
    report (e.g. the 2025 journal version of a 2023 preprint), as in ``dedup.py``."""
    mine = normalize_doi(row.get("doi"))
    return not (mine and doi and normalize_doi(doi) != mine)


def _summary_doi(summary: dict) -> str | None:
    for aid in summary.get("articleids", []):
        if aid.get("idtype") == "doi":
            return aid.get("value")
    return None


def _year_ok(row: dict, year) -> bool:
    if not row.get("year") or not year:
        return True
    return abs(int(row["year"]) - int(year)) <= 1


# --- OpenAlex --------------------------------------------------------------------

def _work_ids(work: dict) -> dict:
    ids = work.get("ids") or {}
    return {"openalex_id": _strip(work.get("id"), "https://openalex.org/"),
            "doi": normalize_doi(work.get("doi")),
            "pmid": _strip(ids.get("pmid"), "https://pubmed.ncbi.nlm.nih.gov/"),
            "pmcid": _pmcid(ids.get("pmcid"))}


def openalex_lookup(row: dict, http: HttpClient, mailto: str | None = None) -> dict | None:
    """``{openalex_id, doi, pmid, pmcid, by}`` or None if OpenAlex does not have it."""
    select = "id,doi,display_name,publication_year,ids"
    doi = normalize_doi(row.get("doi"))
    if doi:
        try:
            work = http.get_json(f"{OPENALEX}/doi:{doi}", {"select": select, "mailto": mailto})
            return {**_work_ids(work), "by": "doi"}
        except HttpError as exc:
            if exc.status != 404:
                raise
    title = clean_term(row.get("title") or "")
    if not title:
        return None
    found = http.get_json(OPENALEX, {"filter": f"title.search:{title}", "per-page": 10,
                                     "select": select, "mailto": mailto})
    for work in found.get("results", []):
        if title_matches(work.get("display_name") or "", title) and \
                _year_ok(row, work.get("publication_year")) and _doi_ok(row, work.get("doi")):
            return {**_work_ids(work), "by": "title"}
    return None


# --- PubMed ----------------------------------------------------------------------

def _esearch(http: HttpClient, term: str, api_key: str | None) -> list[str]:
    out = http.get_json(f"{EUTILS}/esearch.fcgi", {
        "db": "pubmed", "term": term, "retmode": "json", "retmax": 10, "tool": TOOL,
        "api_key": api_key})
    return out.get("esearchresult", {}).get("idlist", [])


def _esummary(http: HttpClient, pmids: list[str], api_key: str | None) -> dict:
    if not pmids:
        return {}
    out = http.get_json(f"{EUTILS}/esummary.fcgi", {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "json", "tool": TOOL,
        "api_key": api_key}).get("result", {})
    return {uid: out[uid] for uid in out.get("uids", []) if "error" not in out[uid]}


def pubmed_lookup(row: dict, http: HttpClient, api_key: str | None = None) -> dict | None:
    """``{pmid, by}`` or None if PubMed does not have it."""
    if row.get("pmid"):
        if _esummary(http, [row["pmid"]], api_key):
            return {"pmid": row["pmid"], "by": "pmid"}
    doi = normalize_doi(row.get("doi"))
    if doi:
        ids = _esearch(http, f"{doi}[doi]", api_key)
        if len(ids) == 1:
            return {"pmid": ids[0], "by": "doi"}
    title = clean_term(row.get("title") or "")
    if title:
        summaries = _esummary(http, _esearch(http, f"{title}[ti]", api_key), api_key)
        for uid, s in summaries.items():
            if title_matches(s.get("title") or "", title) and \
                    _year_ok(row, (s.get("pubdate") or "")[:4] or None) and \
                    _doi_ok(row, _summary_doi(s)):
                return {"pmid": uid, "by": "title"}
    return None


# --- Coverage --------------------------------------------------------------------

def coverage(rows: list[dict], http: HttpClient, *, mailto: str | None = None,
             ncbi_key: str | None = None) -> list[dict]:
    """Each gold row + coverage columns (see ``COVERAGE_COLS``)."""
    http.set_min_interval("eutils.ncbi.nlm.nih.gov", 0.11 if ncbi_key else 0.34)
    out = []
    for row in rows:
        oa = openalex_lookup(row, http, mailto)
        pm = pubmed_lookup(row, http, ncbi_key)
        out.append({**row,
                    "in_openalex": "yes" if oa else "no",
                    "openalex_by": oa["by"] if oa else "",
                    "openalex_id": oa["openalex_id"] if oa else "",
                    "in_pubmed": "yes" if pm else "no",
                    "pubmed_by": pm["by"] if pm else "",
                    "found_doi": normalize_doi(row.get("doi")) or (oa or {}).get("doi") or "",
                    "found_pmid": row.get("pmid") or (pm or {}).get("pmid")
                                  or (oa or {}).get("pmid") or ""})
    return out


def summarize(rows: list[dict]) -> dict:
    """Counts for the report: per database, either, and by publication type."""
    either = [r for r in rows if r["in_openalex"] == "yes" or r["in_pubmed"] == "yes"]
    by_type = Counter(r["type"] for r in rows)
    covered_type = Counter(r["type"] for r in either)
    return {"gold": len(rows),
            "openalex": sum(r["in_openalex"] == "yes" for r in rows),
            "pubmed": sum(r["in_pubmed"] == "yes" for r in rows),
            "either": len(either),
            "by_type": {t: f"{covered_type[t]}/{n}" for t, n in by_type.items()}}


def main(argv: list[str] | None = None) -> int:
    """python -m sr_pipeline.evaluation.gold evaluation/gold/T1.csv [...]"""
    import os

    from ..config import load_config

    ap = argparse.ArgumentParser(description="Coverage of gold lists in OpenAlex and PubMed")
    ap.add_argument("gold", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="folder for *_coverage.csv "
                    "(default: next to each gold file)")
    args = ap.parse_args(argv)
    cfg = load_config()
    http = cfg.make_http()
    for path in args.gold:
        rows = coverage(load_gold(path), http, mailto=cfg.contact_email,
                        ncbi_key=os.environ.get("NCBI_API_KEY"))
        dest = (args.out or path.parent) / f"{path.stem}_coverage.csv"
        with dest.open("w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        s = summarize(rows)
        print(f"{path.stem}: {s['gold']} gold, OpenAlex {s['openalex']}, PubMed {s['pubmed']}, "
              f"either {s['either']}  by type {s['by_type']}")
        for r in rows:
            if r["in_openalex"] == "no" or r["in_pubmed"] == "no":
                print(f"   row {r['row']:>2} {r['first_author']} {r['year']} ({r['type']}): "
                      f"OpenAlex {r['in_openalex']}, PubMed {r['in_pubmed']}")
    print(f"HTTP requests sent: {http.network_calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --- Matching gold articles to our papers ------------------------------------------------

def match_gold(rows: list[dict], papers: list[dict]) -> dict[str, str]:
    """{gold row: paper_id} for gold articles present among ``papers`` (by DOI, else PMID).

    Uses the identifiers found by the coverage lookup (``found_doi`` / ``found_pmid``)
    when the gold file is a ``*_coverage.csv``, so articles the review cited without
    a DOI are recognised too."""
    by_doi = {normalize_doi(p.get("doi")): p["paper_id"] for p in papers if p.get("doi")}
    by_pmid = {str(p["pmid"]): p["paper_id"] for p in papers if p.get("pmid")}
    out = {}
    for r in rows:
        doi = normalize_doi(r.get("found_doi") or r.get("doi"))
        pmid = r.get("found_pmid") or r.get("pmid")
        pid = by_doi.get(doi) if doi else None
        pid = pid or (by_pmid.get(str(pmid)) if pmid else None)
        if pid:
            out[r["row"]] = pid
    return out
