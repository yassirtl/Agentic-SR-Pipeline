# Gold lists

One CSV per evaluation topic with a published review (see `../topics.yaml`): the studies
included by that review, one row per article, in the order of the review's table. T2 is a
new review with no published answer: no gold list. Topic ids were revised on 2026-11-08:
`D0.csv` is the development topic (the old T3, exercise and depression), `T3.csv` the
embedded-ECG review; `dropped_kim2025*.csv` is the old T2 (Kim 2025), kept but unused.

| Column | Meaning |
|---|---|
| `row` | position in the review's table (Table 1; T3: Table 2) |
| `ref` | reference number in the review (check a row against the paper with it) |
| `first_author`, `year`, `title`, `venue` | from the review's reference list |
| `doi`, `pmid`, `pmcid` | as printed in the review's reference list (empty if not printed) |
| `type` | journal article, conference abstract, conference paper, preprint, non-English journal article |
| `note` | discrepancies between Table 1 and the reference list |

How they were made (2026-11-07): T1 and T3 from the PMC JATS XML of the review
(`efetch db=pmc`: Table 1 rows link to reference elements that carry DOI / PMID / PMCID);
Kim 2025 from the journal's HTML (not in PMC; reference links give DOI and PMID). Nothing
typed by hand except the Kim 2025 Table 1 order (author, year, reference number).

T3 (2026-11-08): Arredondo-Velazquez 2026 (Electronics 15(21):5046, open access). Its
Table 2 (risk of bias of "the 30 implementations", read in the MDPI HTML) lists references
[27]-[56], the same 30 as Table 5; author, year, title, venue and DOI come from the
review's reference list as deposited in Crossref (`api.crossref.org/works/10.3390/
electronics15215046`, 99 references; OpenAlex lists 87). `row` = reference number - 26.
The 24 algorithmic studies (no implementation) are not in the gold list.

Known discrepancies:
- T1 row 14 (Rajakariar): Table 1 says 2020, the reference is a 2019 conference abstract.
- Kim 2025 row 10 (Rai): Table 1 says 2023, the reference says ISPOR 2024.
- D0: Table 1 has 15 rows; the text says 13 articles / 18 studies. Lyu 2012 has no
  numbered reference.
- T3 row 12 (Cao, ref 38): Table 5 says '26, the reference list 2025. Row 5 (Cai, ref 31):
  no DOI printed, found by title in OpenAlex (10.1109/tcsii.2021.3091198, checked).
  3 rows are older than the 2021-2026 window (Raj 2018, Wu 2019, Saadatnejad 2020).

## Coverage (`*_coverage.csv`)

`python -m sr_pipeline.evaluation.gold evaluation/gold/T*.csv` looks every row up in
OpenAlex and PubMed (DOI or PMID first, then title with a fuzzy match, year +-1 and no
conflicting DOI). Coverage is **not recall**: it says whether an article can be found by
our pipeline at all. Columns: `in_openalex`, `openalex_by` (doi/title), `openalex_id`,
`in_pubmed`, `pubmed_by` (pmid/doi/title), `found_doi`, `found_pmid` (identifiers the
review did not print, needed to recognise the article among our records).

Result (2026-11-07; T3 2026-11-08):

| Topic | Gold | OpenAlex | PubMed | Either | Not covered |
|---|---|---|---|---|---|
| T1 | 18 | 18 | 17 | 18 | none (Rajakariar abstract: OpenAlex only) |
| T3 | 30 | 30 | 16 | 30 | none (14 engineering-venue papers, IEEE circuits/VLSI and Elsevier, are OpenAlex only) |
| D0 | 15 | 11 | 11 | 11 | An 2019, Guan 2017, Lyu 2012, Ma 2012 (Chinese journals) |
| Kim 2025 (dropped) | 14 | 13 | 6 | 13 | Royer 2023 (ISPOR Europe abstract) |

T3: the review also searched IEEE Xplore, Scopus, Web of Science and ScienceDirect, none of
which we use; every gold article is still in OpenAlex, so a miss will be the query's, the
screening limit's or the screeners', not the database's.

Checked by hand: the 4 ISPOR abstracts matched by title are the Value in Health
supplement records (`10.1016/j.jval.2024.03.*`); Huotala is the ACM EASE paper. A first
version matched Cai (medRxiv 2023) in PubMed to its 2025 BMC journal version (another
report, published after the search date): title matches with a different DOI are now
rejected.
