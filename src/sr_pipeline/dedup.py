"""Deduplication: records from several databases -> one entry per paper (PRISMA "duplicates removed").

The same trial is usually indexed by OpenAlex *and* PubMed. Screening both copies
doubles the work and could count one study twice in the synthesis.

Rules, from most to least trustworthy (records are grouped with a union-find, so
A=B by DOI and B=C by PMID puts A, B, C in one group):

1. **Shared identifier**: same record id (a database returning one record twice,
   e.g. a paging overlap), then same DOI, PMID, PMC id. Exact.
   (The plan named DOI only; PMID/PMCID catch records without a DOI, since
   OpenAlex also stores PubMed ids.)
2. **Fuzzy title** for the rest: normalized titles (lowercase, no accents or
   punctuation) with a similarity of at least ``TITLE_THRESHOLD`` / 100
   (``rapidfuzz.fuzz.ratio``), guarded because a wrong merge silently loses a study:
   - both titles have at least ``MIN_TITLE_WORDS`` words (short titles such as
     "Exercise and depression" are shared by unrelated papers);
   - publication years differ by at most one (online-first vs print year);
   - a notice ("Correction:", "Erratum:", "Retraction:", "Reply:", ...) never
     merges with an article.

Guard for every rule except DOI: two groups whose DOIs differ are never merged
(e.g. a trial protocol and its results paper share most of a title but have
two DOIs; so are a conference abstract and the article, a preprint and the
journal version: different reports of one study, linked later at study level).
Such near-misses are kept apart and listed in ``conflicts``.

DOIs that do not name a separate report (seen in the development data):
- *deposits* (Figshare, Zenodo, OSF, Dryad): the supplementary files or a copy
  of an article. They do not block a title merge, skip the year guard (often
  deposited years later), and the paper keeps the journal DOI;
- *versions*: ``figshare.123.v1`` / ``.v2`` and F1000Research ``.27123.1`` /
  ``.27123.2`` are compared on the base DOI.

Each merge is logged ``{paper_id, record, matched, rule}``, so the PRISMA count can be audited.

The merged paper takes its fields from the best record (one with an abstract,
PubMed before OpenAlex before arXiv) and fills gaps from the others; it keeps
every record id and source. ``paper_id`` is built from the identifiers (DOI,
else PMID, else PMC id, else the smallest record id), so it stays the same when
retrieval runs again: later stages match decisions to papers by this id.
"""

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

TITLE_THRESHOLD = 95
MIN_TITLE_WORDS = 4
SOURCE_PRIORITY = {"pubmed": 0, "openalex": 1, "arxiv": 2}
ID_RULES = ("id", "doi", "pmid", "pmcid")  # "id": a database returned one record twice
NOTICE = re.compile(r"^(correction|corrigendum|erratum|retraction|retracted|reply|response"
                    r"|comment|expression of concern)\b")
DEPOSIT = re.compile(r"^10\.6084/|^10\.5281/zenodo|^10\.17605/osf|^10\.5061/dryad|figshare")
UNUSABLE_PDF_HOSTS = ("pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov")  # bot wall; PMC text via efetch


def normalize_title(title: str | None) -> str:
    """'Exercise &amp; Depression: a Randomised Trial.' -> 'exercise depression a randomised trial'."""
    text = unicodedata.normalize("NFKD", title or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"&\w+;", " ", text)  # leftover HTML entities
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def base_doi(doi: str) -> str:
    """Drop a version suffix: '...figshare.1.v2' -> '...figshare.1', F1000 '....27123.2' -> '....27123'."""
    doi = re.sub(r"\.v\d+$", "", doi)
    return re.sub(r"^(10\.12688/f1000research\.\d+)\.\d+$", r"\1", doi)


def is_deposit(doi: str | None) -> bool:
    return bool(doi and DEPOSIT.search(doi))


class _Groups:
    """Union-find over record indices, remembering each group's report DOIs (deposits left out)."""

    def __init__(self, records: list[dict]):
        self.parent = list(range(len(records)))
        self.dois = [{base_doi(r["doi"])} if r.get("doi") and not is_deposit(r["doi"]) else set()
                     for r in records]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def doi_conflict(self, a: int, b: int) -> bool:
        da, db = self.dois[self.find(a)], self.dois[self.find(b)]
        return bool(da and db and da != db)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        keep, drop = min(ra, rb), max(ra, rb)  # the earlier record's group survives
        self.parent[drop] = keep
        self.dois[keep] |= self.dois[drop]
        return True


def _title_pairs(norm: list[str]) -> list[tuple[int, int, int]]:
    """Index pairs (i < j) whose normalized titles score >= TITLE_THRESHOLD, with the score."""
    if len(norm) < 2:
        return []
    scores = cdist(norm, norm, scorer=fuzz.ratio, score_cutoff=TITLE_THRESHOLD,
                   dtype=np.uint8, workers=-1)
    i, j = np.nonzero(np.triu(scores, k=1))
    return [(int(a), int(b), int(scores[a, b])) for a, b in zip(i, j)]


def _title_guard(a: dict, b: dict, na: str, nb: str) -> str | None:
    """Reason not to merge two title matches, or None."""
    if min(len(na.split()), len(nb.split())) < MIN_TITLE_WORDS:
        return "title too short"
    deposit = is_deposit(a.get("doi")) or is_deposit(b.get("doi"))
    if not deposit and a.get("year") and b.get("year") and abs(a["year"] - b["year"]) > 1:
        return "years differ"
    if bool(NOTICE.match(na)) != bool(NOTICE.match(nb)):
        return "notice vs article"
    return None


def _merge(group: list[dict]) -> dict:
    """One paper from the records of a group (best record first, gaps filled from the others)."""
    ranked = sorted(group, key=lambda r: (not r.get("abstract"),
                                          SOURCE_PRIORITY.get(r["source"], 9)))
    best = ranked[0]

    def first(field: str, ok=lambda _: True):
        return next((r[field] for r in ranked if r.get(field) and ok(r[field])), None)

    paper = {
        "title": best["title"],
        "abstract": first("abstract"),
        "authors": first("authors") or [],
        "year": first("year"),
        "venue": first("venue"),
        "doi": first("doi", lambda d: not is_deposit(d)) or first("doi"),
        "pmid": first("pmid"),
        "pmcid": first("pmcid"),
        "url": first("url"),
        # PMC's own PDF links hit a bot wall: prefer a publisher/arXiv PDF when there is one.
        "oa_pdf_url": first("oa_pdf_url", lambda u: not u.split("/")[2] in UNUSABLE_PDF_HOSTS)
        or first("oa_pdf_url"),
        "sources": sorted({r["source"] for r in group}, key=lambda s: SOURCE_PRIORITY.get(s, 9)),
        "record_ids": [r["id"] for r in group],
        # Part 9b: best position in any database's relevance order (1 = top hit), for the
        # screening limit; None when the records carry no rank.
        "rank": min((r["rank"] for r in group if r.get("rank")), default=None),
    }
    if paper["doi"]:
        pid = f"doi:{paper['doi']}"
    elif paper["pmid"]:
        pid = f"pmid:{paper['pmid']}"
    elif paper["pmcid"]:
        pid = f"pmcid:{paper['pmcid']}"
    else:
        pid = min(paper["record_ids"])
    return {"paper_id": pid, **paper}


def deduplicate(records: list[dict]) -> dict:
    """Group ``records`` (``Record`` dumps) into papers.

    Returns ``{"papers", "duplicates", "conflicts", "counts"}``:
    - ``papers``: merged papers, in order of their first record (databases keep
      their relevance order);
    - ``duplicates``: one ``{paper_id, record, matched, rule}`` per merge (``record`` joined
      the paper of ``matched``);
    - ``conflicts``: candidate pairs kept apart, with the reason;
    - ``counts``: ``records``, ``duplicates_removed``, ``by_rule``, ``unique``.
    """
    groups = _Groups(records)
    merges: list[tuple[int, int, str]] = []  # (a, b, rule) of successful unions
    conflicts: list[dict] = []

    for rule in ID_RULES:
        seen: dict[str, int] = {}
        for i, r in enumerate(records):
            value = r.get(rule)
            if not value:
                continue
            key = base_doi(value) if rule == "doi" else str(value).lower()
            if key not in seen:
                seen[key] = i
                continue
            j = seen[key]
            if rule not in ("id", "doi") and groups.doi_conflict(i, j):
                conflicts.append({"records": [records[j]["id"], r["id"]], "rule": rule,
                                  "reason": "different DOIs"})
            elif groups.union(j, i):
                merges.append((j, i, rule))

    norm = [normalize_title(r.get("title")) for r in records]
    for a, b, score in _title_pairs(norm):
        if groups.find(a) == groups.find(b):
            continue
        reason = _title_guard(records[a], records[b], norm[a], norm[b])
        if reason is None and groups.doi_conflict(a, b):
            reason = "different DOIs"
        if reason:
            conflicts.append({"records": [records[a]["id"], records[b]["id"]],
                              "rule": f"title {score}", "reason": reason})
        elif groups.union(a, b):
            merges.append((a, b, "title"))

    members: dict[int, list[int]] = {}
    for i in range(len(records)):
        members.setdefault(groups.find(i), []).append(i)
    papers, root_pid, used = [], {}, set()
    for root in sorted(members):  # order of first record
        paper = _merge([records[i] for i in members[root]])
        pid, n = paper["paper_id"], 2
        while pid in used:  # cannot happen with consistent identifiers; never collide anyway
            pid, n = f"{paper['paper_id']}#{n}", n + 1
        paper["paper_id"] = pid
        used.add(pid)
        root_pid[root] = pid
        papers.append(paper)

    duplicates = [{"paper_id": root_pid[groups.find(a)], "record": records[b]["id"],
                   "matched": records[a]["id"], "rule": rule} for a, b, rule in merges]
    by_rule = {rule: sum(1 for *_, r in merges if r == rule) for rule in (*ID_RULES, "title")}
    counts = {"records": len(records), "duplicates_removed": len(merges),
              "by_rule": by_rule, "unique": len(papers)}
    return {"papers": papers, "duplicates": duplicates, "conflicts": conflicts, "counts": counts}
