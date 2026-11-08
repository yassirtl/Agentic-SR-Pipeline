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
.venv/bin/pytest -q            # 284 tests, no network, no API key (scripted fake LLM)
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
.venv/bin/python -m sr_pipeline.evaluation evaluate RUN_ID            # metrics → runs/<id>/evaluation/
.venv/bin/python -m sr_pipeline.evaluation sample RUN_ID              # blind sheet of 30 abstracts to label
.venv/bin/python -m sr_pipeline.evaluation compare RUN_A RUN_B        # side by side (e.g. two --limit values)
```

Metrics, grouped by the four criteria of the brief (`src/sr_pipeline/evaluation/metrics.py`; a topic can choose its own list with `metrics:` and new ones register with `@metric(name, criterion)`):

| Criterion | Metrics |
|---|---|
| Completeness | gold studies kept at each stage (coverage, search, screening limit, title/abstract, full text) with the reason each one was lost; recall vs all gold and vs gold present in our databases; extraction fields reported with a verified quote; PRISMA arithmetic checks; critic outcome |
| Accuracy | gold share of our includes (a lower bound on precision); title/abstract screening vs a human-labelled sample: accuracy, sensitivity, specificity, kappa, disagreements |
| Inter-agent consistency | kappa and agreement of screener A vs B per stage, share adjudicated; coordinator decisions and guardrail refusals |
| Time savings | run time, LLM calls, cost; against a manual estimate when `evaluation/manual_effort.yaml` is filled with sourced rates |

## Results so far

| Topic | Gold review | Run | Gold: all → in our databases → found by search → screened → kept TA → kept FT | Included | Cost |
|---|---|---|---|---|---|
| T1 smartwatches, arrhythmia | Nazarian 2021 (18 studies) | pending | | | |
| T2 LLMs for abstract screening | Kim 2025 (14 studies) | pending | | | |
| T3 exercise, depression, older adults | Li 2024 (15 articles) | `--topic T3 --limit 100` | 15 → 11 → 7 → 3 → 2 → 1 | 28 | $0.37 |

T3 (first full run, 2026-11-08): the search found 7 of the 11 gold articles our databases hold (64%); 4 gold articles are in Chinese journals indexed by neither OpenAlex nor PubMed. With the default limit of the 100 most relevant records (of 3,631), 4 of the 7 found were never screened (best database ranks 56–382). Of the 3 screened, Makizako 2019 was included, Shahidi 2011 (laughter yoga vs exercise) and Huang 2015 (exercise vs CBT, abstract only) were excluded on "interventions beyond exercise". So the losses come from the screening limit and the search first, then from how the criteria are read. Screener agreement was low (kappa 0.25 at title/abstract, 0.19 at full text: the strict and lenient screeners are designed to differ), the critic passed with 3 warnings, and the PRISMA counts add up. These results are from one topic; T1/T2 runs, a run without the screening limit and a human-labelled sample are still to come.

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
- The screening limit (default 100 records) trades recall for cost; use `--limit 0` for a complete review.
