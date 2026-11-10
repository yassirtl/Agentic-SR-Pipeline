# Agentic SR Pipeline

A multi-agent LangGraph pipeline that runs a PRISMA 2020 systematic review from one research question: protocol, search, deduplication, title/abstract and full-text screening, data extraction, risk of bias, synthesis, quality check and report. Every decision is logged and every number in the report is computed by code from the run's data.

The design, decisions, tuning log and build history are in [docs/PLAN.md](docs/PLAN.md).

## How it works

```
                 ┌──────────── Coordinator (LLM) ────────────┐
                 │  picks the next stage after every step;   │
                 │  a code guardrail enforces PRISMA order   │
                 └───────────────────────────────────────────┘
 planner → search_strategy → retrieval → ta_screening → ft_screening → extraction → synthesis → critic → report
 (protocol)  (Boolean queries)  (OpenAlex,   (2 screeners +   (open-access    (fields + RoB,   (meta-analysis   (checks the  (PRISMA diagram,
             count-checked      PubMed,       adjudicator,     full text, RAG   each value with  or SWiM, text    text against manuscript, tables)
                                arXiv; dedup) human on ties)   passages)        a verified quote) with citations) the tables)
```

- **Coordinator + guardrail.** An LLM chooses the next stage from a summary of the state; code refuses a move whose prerequisites are not fresh, a re-run of a stage whose inputs did not change, and searches beyond a budget. Refusals are logged with the substituted stage.
- **Human in the loop.** Two pauses: protocol approval (edit anything, accept a suggested design restriction) and screening conflicts the adjudicator cannot resolve. `--auto` skips both.
- **Nothing invented.** A screening exclusion, an extracted value and a risk-of-bias judgment each need a quote that code finds in the text; numbers in the narrative must appear in the evidence tables; the critic's objections become blockers only when code confirms them.
- **Reproducible.** LLM answers, database answers and embeddings are cached on disk: re-running a finished run costs $0 and gives the same output.

## Install

Python 3.11, macOS or Linux.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q            # 348 tests, no network, no API key (scripted fake LLM)
```

Embeddings use the OpenAI API (`text-embedding-3-small`), so no PyTorch is needed.

## Configure

Put your keys in a `.env` file at the project root (git-ignored):

```
OPENAI_API_KEY=sk-...
NCBI_API_KEY=...          # optional: faster PubMed
OPENALEX_API_KEY=...      # optional: larger OpenAlex quota
SR_CONTACT_EMAIL=you@example.org   # optional: polite pool for OpenAlex / NCBI
```

- `config.yaml`: model per agent (`default_model`, `models:`), `max_records` per database, refinement limits, RAG and screening settings. Any key can be overridden by an `SR_*` environment variable (e.g. `SR_MODEL_ADJUDICATOR=gpt-4.1-mini`).
- `protocols/template.yaml`: the review's scope (databases, years, languages) and optional hints for the Planner (PICO, designs, criteria). Years: `recent: 5` = last 5 years; `start: 2015` fixes the first year; `start: null` = no lower limit.

## Run

```bash
.venv/bin/sr-review "Does exercise reduce depressive symptoms in older adults?"
.venv/bin/sr-review --topic T3            # an evaluation topic (question + criteria from evaluation/topics.yaml)
.venv/bin/sr-review --topic T3 --limit 0  # screen every record (default limit: the 100 most relevant)
.venv/bin/sr-review "..." --auto          # no human pauses
.venv/bin/sr-review --resume RUN_ID       # continue after a crash or Ctrl-C
.venv/bin/sr-review --report-only RUN_ID  # rebuild the report of a finished run (no LLM)
```

At the protocol pause press Enter to approve, `s` to accept the suggested design restriction, or type JSON edits (e.g. `{"edits": {"years": {"start": 2010}}}`). At a screening conflict answer `i` or `e` per record.

Each run writes `runs/<run_id>/`:

| File | Content |
|---|---|
| `report/manuscript.md`, `manuscript.docx` | PRISMA-style manuscript: methods, results, risk of bias, synthesis, QA, limitations |
| `report/prisma_flow.png` | PRISMA 2020 flow diagram (with arithmetic checks) |
| `report/tables/*.csv` | studies, extraction (with quotes), results, risk of bias, every screening decision |
| `report/references.bib`, `protocol.json`, `run_summary.json` | references, final protocol, time / calls / cost per agent |
| `audit.jsonl` | every node's decisions, prompts' outcomes, refusals, costs |
| `state.json`, `checkpoints.sqlite` | final state, resume checkpoint |

## Evaluate

The pipeline is compared with three published reviews (`evaluation/topics.yaml`, gold lists in `evaluation/gold/`, see its README). No LLM and no network are used.

```bash
.venv/bin/python -m sr_pipeline.evaluation evaluate RUN_ID            # metrics → runs/<id>/evaluation/ + evaluation/results/
.venv/bin/python -m sr_pipeline.evaluation sample RUN_ID              # blind sheet of 30 abstracts → evaluation/labels/<id>/
.venv/bin/python -m sr_pipeline.evaluation compare RUN_A RUN_B        # side by side (e.g. two --limit values)
```

Metrics, grouped by the four criteria of the brief (`src/sr_pipeline/evaluation/metrics.py`; a topic can choose its own list with `metrics:` and new ones register with `@metric(name, criterion)`):

| Criterion | Metrics |
|---|---|
| Completeness | gold studies kept at each stage (coverage, search, screening limit, title/abstract, full text) with the reason each one was lost; recall vs all gold and vs gold present in our databases; extraction fields reported with a verified quote (T2/T3: only the topic's own engineering fields; all fields shown apart); PRISMA arithmetic checks; critic outcome |
| Accuracy | gold share of our includes (a lower bound on precision); title/abstract screening vs a human-labelled sample when one is filled in (accuracy, sensitivity, specificity, kappa, disagreements) |
| Inter-agent consistency | kappa and agreement of screener A vs B per stage, share adjudicated; coordinator decisions and guardrail refusals |
| Time savings | run time, LLM calls, cost; next to a manual estimate in person-days from the sourced rates in `evaluation/manual_effort.yaml` (PredicTER defaults) |

## Results

Full analysis, figures and limitations: [docs/report/report.pdf](docs/report/report.pdf). All numbers come from `python -m sr_pipeline.evaluation evaluate RUN` (copies in `evaluation/results/`); gpt-4o-mini for every agent. v1 and D0 runs had a human at the pauses (protocol approved without edits, T3 conflicts included); later runs used `--auto`, which reproduces those answers.

| Topic | Gold review | Version | Gold: all → in our databases → found by search → screened → kept TA → kept FT | Final recall | Included (gold share) | Cost, time |
|---|---|---|---|---|---|---|
| T1 smartwatches, arrhythmia | Nazarian 2021 (18 studies) | **v1, held out**, limit 100 | 18 → 18 → 9 → 9 → 9 → 8 | **44.4%** | 27 (29.6%) | $0.27, 0.14 h |
| | | v2 + D fix, limit 500 (post hoc) | 18 → 18 → 13 → 13 → 13 → 12 | 66.7% | 252 (4.8%) | $1.04*, 0.46 h |
| T3 ML ECG classification on embedded devices | Arredondo-Velázquez 2026 (30 embedded implementations) | **v1, held out**, limit 100 | 30 → 30 → 8 → 5 → 5 → 5 | **16.7%** | 55 (9.1%) | $0.49, 0.23 h |
| | | v2 + D fix, limit 500 (post hoc) | 30 → 30 → 17 → 17 → 17 → 16 | 53.3% | 293 (5.5%) | $0.52*, 0.45 h |
| T2 lightweight ML, abnormal vs normal ECG, constrained devices | none: new review | v2, limit 100 | not applicable | not measured | 55 | $0.57, 0.23 h |
| D0 exercise, depression, older adults (development topic) | Li 2024 (15 articles) | limit 100 | 15 → 11 → 7 → 3 → 2 → 1 | 6.7% | 28 (3.6%) | $0.37, 1.4 h |

\* marginal cost: most LLM answers came from the cache of earlier runs. "Gold share" is a lower bound on precision (an include outside the gold list may still be eligible).

- **Held out vs post hoc.** v1 is the pipeline with screening frozen and the 10b rules (prompted by T2), before any change prompted by T1/T3 results. v2 (search fixes) and the D fix (a bug in the design check) were made after seeing the v1 and limit-500 results, so the final numbers are post hoc (report §3.1).
- **What raised recall.** Search fixes: gold found 9 → 13 (T1), 8 → 17 (T3). Screening every record instead of the top 100: +16.7 points on both topics. D fix: no gold study lost on the design criterion (T1 2 → 0, T3 5 → 0), +5.6 and +13.3 points.
- **What it cost.** Precision fell to about 5%: about 90% of the final includes were judged on the abstract only (no open-access full text), where the rules never exclude for missing information. Title/abstract screener agreement is near 0 (T3 in every version; T1 after the D fix, 0.42 → 0.07), so the adjudicator decides most records.
- **What is still lost.** Mostly the query (T1 5, T3 13 gold studies never returned, though all are in OpenAlex); one full-text judgement per topic. Screening kept 28 of the 30 gold studies it saw.
- **Accuracy against human labels: not measured** (labelling skipped, 2026-11-10); T2's accuracy is unknown.
- **Time.** For the same workload (screening, extraction, risk of bias), the PredicTER default rates (Haddaway & Westgate 2019; `evaluation/manual_effort.yaml`) give 69 (T1) and 80 (T3) person-days; the pipeline took under half an hour. Most of those days are extraction of includes a human would have excluded first (report §4.6).

## Layout

```
src/sr_pipeline/   state, guardrail, coordinator, llm (cache + cost), graph, cli, pipeline
  agents/          planner, search_strategy, retrieval, screening, extraction, synthesis, critic
  sources/         OpenAlex, PubMed, arXiv connectors, HTTP cache, design filters, full text
  rag/             chunking, embeddings, FAISS + BM25 hybrid index, LLM re-ranker
  reporting/       PRISMA diagram, manuscript (Markdown + DOCX), tables, run summary
  evaluation/      gold lists + coverage, topics, recall funnel, labels, metrics, CLI
protocols/template.yaml   evaluation/   tests/   docs/PLAN.md
```

## Limitations

- Only free databases (OpenAlex, PubMed, arXiv): articles indexed only in Embase, Scopus, Cochrane or Chinese databases cannot be found.
- Full text only when open access (PMC XML or an open PDF); other studies are screened and extracted from the abstract, and flagged.
- gpt-4o-mini by default: screener agreement is modest and criteria wording drives exclusions (see the screening tuning log in docs/PLAN.md).
- The screening limit (default 100 records) trades recall for cost (+17 recall points at 500 on T1 and T3); use `--limit 0` for a complete review.
- Low precision (about 5% gold share at limit 500) and title/abstract screener agreement near 0: the includes need a human check.
