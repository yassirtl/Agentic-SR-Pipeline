# Agentic Systematic Review Pipeline: technical report

Draft, 2026-11-08. Code: https://github.com/yassirtl/Agentic-SR-Pipeline. <span class="pending">[pending] marks results that need the remaining runs.</span>

## Abstract

We built a multi-agent LangGraph system that runs a PRISMA 2020 systematic review from a single research question: protocol, search of three free databases, deduplication, dual title/abstract and full-text screening, data extraction, risk of bias, synthesis, quality check and report. An LLM coordinator chooses each next stage; code enforces PRISMA prerequisites and refuses unjustified re-runs. Every exclusion, extracted value and risk-of-bias judgment must rest on a quote that code finds in the source text, and every number in the narrative must appear in tables computed by code. We evaluate the pipeline against three published reviews, following each of their included studies through our pipeline stage by stage. In the first full run (exercise and depression in older adults, gpt-4o-mini, $0.37), the search found 7 of the 11 gold studies present in our databases, but only 1 reached the final included set: four of the seven found were outside the screening limit, so no screener saw them, and two to the same screening error on three-arm trials (judged by their extra non-exercise arm). <span class="pending">[pending: T1, T2, unlimited run, human-labelled sample]</span>

## 1. Introduction

A systematic review takes months of expert time, most of it in screening and extraction. LLMs can read abstracts and full texts quickly, but they invent quotes and numbers and their decisions are hard to audit. The goal of this project is a pipeline that is (a) end to end: from a question to a PRISMA report; (b) transparent: each decision is logged with its evidence; (c) reproducible: a re-run of the same inputs gives the same output at no cost; and (d) evaluated against human reviews, so that its errors can be measured and explained.

## 2. System design

### 2.1 Orchestration

The state is one typed dictionary (`ReviewState`); lists written by several stages are append-only, and each item is tagged with its round so that a re-run never erases history. Nine stages follow the PRISMA order: `planner, search_strategy, retrieval, ta_screening, ft_screening, extraction, synthesis, critic, report`. After every stage the **coordinator** (an LLM) sees a summary of the state (counts, flags, critic issues, never the raw records) and proposes the next stage with a rationale. A code **guardrail** accepts the proposal only if all its prerequisites are *fresh* (they ran after their own prerequisites); otherwise it substitutes the earliest stale stage and logs the refusal. Three rules added after the first real runs also refuse re-running a stage whose inputs did not change, searching again before retrieval has used the current strategy, and searching again once screening has included records. A step budget (40) ends any loop. Two **human pauses** (LangGraph `interrupt`): protocol approval and screening conflicts the adjudicator cannot resolve; both are skipped with `--auto`. Runs are checkpointed in SQLite and can be resumed after a crash.

### 2.2 Agents

| Stage | What the LLM does | What code does |
|---|---|---|
| Planner | drafts PICO, criteria, outcome directions, a design suggestion | applies the template's scope; for evaluation topics, replaces the criteria with the gold review's |
| Search strategy | gives concepts (keywords, MeSH) and refines them on hit counts | builds each database's Boolean query, adds validated design filters, counts hits before downloading, stops a refinement that does not move the counts |
| Retrieval | none | downloads (OpenAlex, PubMed, arXiv), keeps each record's relevance rank, deduplicates (identifiers, then fuzzy title with guards) |
| Title/abstract screening | strict screener A, lenient screener B, adjudicator on disagreement; one verdict per criterion with a quote | verifies quotes, decides from the verdicts (an exclusion needs a verified quote), title rule for reviews/protocols, Cohen's kappa |
| Full-text screening | same, on retrieved passages | fetches open-access full text (PMC XML, PDF), chunks it, hybrid retrieval (FAISS + BM25, rank fusion) and LLM re-ranking per criterion |
| Extraction, risk of bias | fills fields and per-arm results with quotes; judges RoB domains | accepts a value only if its quote is found and prints every number of the value; chooses the RoB tool from the verified design; computes overall judgments |
| Synthesis | writes the narrative as cited sentences | effect sizes, random-effects meta-analysis (≥ 3 studies) or SWiM vote count with sign test, forest plots; checks every number and citation of the text |
| Critic | looks for unsupported claims | code checks (coverage of outcomes, heterogeneity, conflicts, wording at high I²); an LLM objection becomes a blocker only when code confirms the wrong value; at most 2 loops back to synthesis |
| Report | none | PRISMA 2020 diagram with arithmetic checks, manuscript (Markdown, DOCX), CSV tables, BibTeX, run summary |

### 2.3 Reproducibility and cost

Every LLM call goes through one function that caches the answer on disk, keyed on model, schema and prompt, and records tokens and cost in the audit log. Database answers and embeddings are cached in the same way. A finished run can be re-run or its report rebuilt (`--report-only`) for $0. The model is configurable per agent; all agents used gpt-4o-mini (temperature 0).

## 3. Evaluation method

### 3.1 Topics and gold standards

| # | Gold review | Question | Designs | Search end | Gold |
|---|---|---|---|---|---|
| T1 | Nazarian 2021, J Med Internet Res | How accurate are smartwatches at detecting cardiac arrhythmias? | any | 2021-02-28 | 18 |
| T2 | Kim 2025, J Med Artif Intell | How accurate are LLMs at automated title and abstract screening for systematic reviews? | any | 2024-08-31 | 14 |
| T3 | Li 2024, Front Psychol | Does exercise reduce depressive symptoms in older adults? | RCT | 2024-01-07 | 15 |

Each run uses the review's question, design restriction, search end date and (for T3) eligibility criteria. The gold lists were built from each review's Table 1 and reference list, identifiers copied from the references.

**Disclosure.** T3 was the development topic: the search and screening rules were tuned on a 50-record T3 sample (tuning log in `docs/PLAN.md`, "Screening freeze"), which contained two gold studies. The full T3 gold list was not used for tuning. T1 and T2 are held out: after seeing their results only bug fixes are allowed, and each is reported.

### 3.2 Metrics

The brief asks for completeness, accuracy, inter-agent consistency and time savings. All metrics are computed by code from a finished run (`python -m sr_pipeline.evaluation evaluate RUN`).

- **Completeness: the gold funnel.** Each gold study is followed through the stages *in our databases* (coverage lookup by DOI, PMID or title in OpenAlex and PubMed), *found by the search*, *inside the screening limit*, *kept at title/abstract*, *kept at full text*. A lost study gets one cause: the database choice, the query, the screening limit (a record above the limit is never counted as an exclusion), or the deciding criterion. Recall is reported against all gold studies and against covered ones. Also: share of extraction fields reported with a verified quote; PRISMA arithmetic checks; critic outcome.
- **Accuracy.** Share of our includes that are gold studies: a lower bound on precision, since an include absent from the gold list may still be eligible. Title/abstract screening against a blind human-labelled random sample of 30 records: accuracy, sensitivity, specificity, Cohen's kappa.
- **Inter-agent consistency.** Kappa between screeners A and B per stage (low by design: A is strict, B lenient), share sent to the adjudicator, coordinator refusals.
- **Time savings.** Wall-clock time, LLM calls and cost per run, against a manual estimate from per-task rates. <span class="pending">[pending: sourced per-task rates in evaluation/manual_effort.yaml]</span> For scale, Haddaway and Westgate (2019) estimate about 164 full-time-equivalent days for an average environmental systematic review.

## 4. Results

### 4.1 T3: exercise and depression in older adults

Run `20261108-091009-f03123` (`--topic T3 --limit 100`): 4,256 records identified (OpenAlex 2,037, PubMed 2,219), 3,631 after deduplication, the 100 most relevant screened (by each database's relevance rank), 28 studies included (19 on abstract only: no open-access full text). The PRISMA counts add up (figure 1). 1,420 LLM calls (686 from cache, after resumes), $0.37, 1.4 h wall clock (including a crash and its resume).

![PRISMA 2020 flow diagram, T3](figures/prisma_T3.png)

| Stage | Gold studies still in |
|---|---|
| gold list | 15 |
| in OpenAlex or PubMed | 11 |
| found by the search | 7 |
| inside the screening limit (100 records) | 3 |
| kept at title/abstract | 2 |
| kept at full text | 1 |

- **Coverage (4 lost):** An 2019, Guan 2017, Lyu 2012, Ma 2012 are in Chinese journals indexed by neither database.
- **Search (4 lost):** Laredo-Aguilera 2018, Apóstolo 2019, Kim 2021, Cunha 2022 are in the databases but not returned by our query. Search recall: 64% of covered gold studies (47% of all).
- **Screening limit (4 lost):** Kim 2019, Chen 2015, Antunes 2005 and Prakhinkit 2014 were found (best database ranks 56, 65, 142, 382) but were outside the 100 records screened (the limit takes the best ranks over both databases, so a rank of 56 in one database can fall outside it).
- **Screening (2 lost of 3 seen):** Shahidi 2011 (laughter yoga vs group exercise) was excluded at title/abstract, and Huang 2015 (exercise vs CBT, abstract only) at full text, both on E2 "interventions beyond exercise" by the adjudicator. Both are three-arm trials with a no-exercise control arm, so the exercise-vs-control pair meets the review's criteria. The adjudicator even named that pair ("exercise therapy group vs control group") and still answered E2 "yes", quoting the other arm (laughter yoga; CBT). These are screening errors: the multi-arm rule (judge only the pair used) is in the prompt but gpt-4o-mini does not follow it. A code check (an E quote that names only an arm outside the stated pair is not evidence) would catch both. Screening is frozen, so this is reported, not fixed. Makizako 2019 was included.
- **Precision:** 1 of our 28 includes is a gold study (3.6%, lower bound).
- **Consistency:** title/abstract kappa A vs B 0.25 (agreement 59%, 82 pairs; 44% adjudicated; 18 records removed by the title rule); full text kappa 0.19 (agreement 63%). The guardrail refused 3 coordinator proposals, all attempts to search again.
- **Extraction and synthesis:** 258 of 392 fields (66%) reported with a verified quote (full text 71%, abstract only 64%). No study had verified per-arm n, mean and SD, so no meta-analysis was possible; the vote count has 1 study favouring exercise and 27 not extractable. The critic passed with 3 warnings; one of them concerns an "overall" sentence that reads a positive effect into a single study, which the code rules do not block.

### 4.2 T1 and T2

<span class="pending">[pending: runs `sr-review --topic T1` and `--topic T2`, after copying their eligibility criteria into topics.yaml]</span>

### 4.3 Cost of the screening limit

<span class="pending">[pending: `sr-review --topic T3 --limit 0` (or a larger limit), then `python -m sr_pipeline.evaluation compare RUN_100 RUN_ALL`]</span>

### 4.4 Screening accuracy against human labels

<span class="pending">[pending: 30 labelled abstracts, `runs/20261108-091009-f03123/evaluation/labels.csv`]</span>

## 5. Discussion

The funnel turns one recall number into causes that can be acted on. In T3, the losses come mostly before any LLM judgment: the databases (4) and the query (4), then the screening limit (4). Of the three gold studies the screeners saw, two were lost to the same error: the adjudicator judged a three-arm trial by its extra non-exercise arm, although it had itself named the exercise-vs-control pair. A rule written in the prompt was not enough; like the quote checks, it needs a code check. The low gold share of our includes (1 of 28) is not yet explained: whether the other 27 are eligible studies the review missed, or screening errors (19 were judged on the abstract only), needs the human-labelled sample and a check of the includes against the review's criteria. <span class="pending">[pending]</span>

The guardrail never had to block an illegal PRISMA move in the real runs. What it caught were legal but useless moves: gpt-4o-mini read "0 records" before retrieval as a failed search and rebuilt the search 39 times in the first attempt. The rules added for this (no re-search before retrieval, no re-run of a fresh stage) are a finding in themselves: an LLM supervisor needs code limits on *repetition*, not only on order.

Requiring a verified quote for every exclusion and value makes the pipeline conservative: it reports "not reported" rather than invent, which explains why no T3 study had complete per-arm data and why the synthesis fell back to a vote count.

## 6. Limitations

- Free databases only; open-access full text only (68% of T3 includes were judged on the abstract).
- One LLM (gpt-4o-mini) for every agent; stronger models were tried only for the adjudicator.
- T3 is a development topic (disclosure in 3.1); T1/T2 are the held-out tests.
- The default screening limit trades recall for cost.
- RoB 2 is assessed for the first outcome only; diagnostic pooling is univariate.

## 7. Individual contribution

<span class="pending">[to write by the author: solo project; list the work packages (design, each part of docs/PLAN.md, evaluation, runs, video) in your own words]</span>

## References

- Page MJ et al. The PRISMA 2020 statement. BMJ 2021;372:n71.
- Campbell M et al. Synthesis without meta-analysis (SWiM) in systematic reviews: reporting guideline. BMJ 2020;368:l6890.
- Haddaway NR, Westgate MJ. Predicting the time needed for environmental systematic reviews and systematic maps. Conserv Biol 2019;33(2):434-443.
- Nazarian S et al. Diagnostic accuracy of smartwatches for the detection of cardiac arrhythmia: systematic review and meta-analysis. J Med Internet Res 2021;23(8):e28974.
- Kim JK et al. Evaluating large language models for title/abstract screening: a systematic review and meta-analysis & development of new tool. J Med Artif Intell 2025;8:34.
- Li X et al. Impact of exercise type, duration, and intensity on depressive symptoms in older adults: a systematic review and meta-analysis. Front Psychol 2024;15:1484172.
