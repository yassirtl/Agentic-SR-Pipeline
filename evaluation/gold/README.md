# Gold lists

One CSV per evaluation topic (see `../topics.yaml`): the studies included by a published
systematic review, one row per article of its Table 1, in Table 1 order.

| Column | Meaning |
|---|---|
| `row` | position in Table 1 |
| `ref` | reference number in the review (check a row against the paper with it) |
| `first_author`, `year`, `title`, `venue` | from the review's reference list |
| `doi`, `pmid`, `pmcid` | as printed in the review's reference list (empty if not printed) |
| `type` | journal article, conference abstract, conference paper, preprint, non-English journal article |
| `note` | discrepancies between Table 1 and the reference list |

How they were made (2026-11-07): T1 and T3 from the PMC JATS XML of the review
(`efetch db=pmc`: Table 1 rows link to reference elements that carry DOI / PMID / PMCID);
T2 from the journal's HTML (not in PMC; reference links give DOI and PMID). Nothing typed
by hand except the T2 Table 1 order (author, year, reference number).

Known discrepancies:
- T1 row 14 (Rajakariar): Table 1 says 2020, the reference is a 2019 conference abstract.
- T2 row 10 (Rai): Table 1 says 2023, the reference says ISPOR 2024.
- T3: Table 1 has 15 rows; the text says 13 articles / 18 studies. Lyu 2012 has no
  numbered reference.

## Coverage (`*_coverage.csv`)

`python -m sr_pipeline.evaluation.gold evaluation/gold/T*.csv` looks every row up in
OpenAlex and PubMed (DOI or PMID first, then title with a fuzzy match, year +-1 and no
conflicting DOI). Coverage is **not recall**: it says whether an article can be found by
our pipeline at all. Columns: `in_openalex`, `openalex_by` (doi/title), `openalex_id`,
`in_pubmed`, `pubmed_by` (pmid/doi/title), `found_doi`, `found_pmid` (identifiers the
review did not print, needed to recognise the article among our records).

Result (2026-11-07):

| Topic | Gold | OpenAlex | PubMed | Either | Not covered |
|---|---|---|---|---|---|
| T1 | 18 | 18 | 17 | 18 | none (Rajakariar abstract: OpenAlex only) |
| T2 | 14 | 13 | 6 | 13 | Royer 2023 (ISPOR Europe abstract) |
| T3 | 15 | 11 | 11 | 11 | An 2019, Guan 2017, Lyu 2012, Ma 2012 (Chinese journals) |

Checked by hand: the 4 ISPOR abstracts matched by title are the Value in Health
supplement records (`10.1016/j.jval.2024.03.*`); Huotala is the ACM EASE paper. A first
version matched Cai (medRxiv 2023) in PubMed to its 2025 BMC journal version (another
report, published after the search date): title matches with a different DOI are now
rejected.
